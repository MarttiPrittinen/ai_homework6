"""
Demo 8 – Resumable AI Procurement Agent (LangGraph Persistence + Interrupt)
"""

import sys
import os
import re
import sqlite3
import time
from typing import TypedDict

import requests
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# ─── State ────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict):
    request: str
    vendors: list[dict]
    quotes: list[dict]
    best_quote: dict
    approval_status: str
    po_number: str
    notification: str
    quantity: int          # TASK 1: store parsed quantity
    rejection_reason: str  # TASK 3: clear reason when rejected


# ─── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# ─── Helpers / Tools ──────────────────────────────────────────────────────────

# TASK 1: parse quantity from request text like "Order 30 laptops..."
def extract_quantity(request: str) -> int:
    match = re.search(r"(\d+)", request)
    return int(match.group(1)) if match else 1


# TASK 4: vendor -> brand mapping for DummyJSON data
VENDOR_BRANDS = {
    "Dell": "Dell",
    "Lenovo": "Lenovo",
    "HP": "HP",   # HP may not exist in DummyJSON laptops, so fallback matters
}

DEFAULT_PRICES = {
    "Dell": 248.0,
    "Lenovo": 235.0,
    "HP": 259.0,
}


# TASK 4: simple "within 2 weeks" check
def ships_within_2_weeks(shipping_info: str) -> bool:
    text = (shipping_info or "").lower()
    good = ["overnight", "1 week", "2 weeks", "1-2 weeks", "few days", "3 days", "5 days", "7 days"]
    bad = ["month", "months"]
    return any(x in text for x in good) and not any(x in text for x in bad)


# TASK 1 + TASK 4:
# LLM can call this tool once per vendor.
# It returns price + product info so later nodes can use the real product name.
@tool
def get_unit_price(vendor: str) -> dict:
    """Get the current unit price and product info for a laptop vendor."""
    brand = VENDOR_BRANDS.get(vendor, vendor)

    try:
        data = requests.get(
            "https://dummyjson.com/products/category/laptops",
            timeout=10,
        ).json()

        products = data.get("products", [])
        matches = [
            p for p in products
            if brand.lower() in p.get("brand", "").lower()
            and p.get("availabilityStatus", "").lower() == "in stock"
            and ships_within_2_weeks(p.get("shippingInformation", ""))
        ]

        if matches:
            best = min(matches, key=lambda p: p.get("price", 999999))
            return {
                "vendor": vendor,
                "unit_price": float(best["price"]),
                "product_name": best["title"],
                "delivery_info": best.get("shippingInformation", "Unknown"),
            }

        print(f"WARNING: No matching live product for {vendor}, using fallback price.")
    except Exception as e:
        print(f"WARNING: Live pricing failed for {vendor}: {e}. Using fallback price.")

    return {
        "vendor": vendor,
        "unit_price": DEFAULT_PRICES[vendor],
        "product_name": f"{vendor} laptop",
        "delivery_info": "Unknown delivery",
    }


tool_llm = llm.bind_tools([get_unit_price])


# ─── Node functions ──────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    print("\n[Step 1] Looking up approved vendors...")
    time.sleep(1)
    vendors = [
        {"name": "Dell", "id": "V-001", "category": "laptops", "rating": 4.5},
        {"name": "Lenovo", "id": "V-002", "category": "laptops", "rating": 4.3},
        {"name": "HP", "id": "V-003", "category": "laptops", "rating": 4.1},
    ]
    for v in vendors:
        print(f"   Found vendor: {v['name']} (rating {v['rating']})")
    return {"vendors": vendors}


def fetch_pricing(state: ProcurementState) -> dict:
    print("\n[Step 2] Fetching pricing from suppliers...")

    # TASK 1: quantity comes from the request instead of being hardcoded
    quantity = extract_quantity(state["request"])

    # TASK 1: make the LLM call the tool once per vendor
    vendor_names = [v["name"] for v in state["vendors"]]
    prompt = (
        f"The user requested {quantity} laptops. "
        f"Call get_unit_price once for each of these vendors: {', '.join(vendor_names)}. "
        f"Do not skip any vendor."
    )
    ai_msg = tool_llm.invoke(prompt)

    tool_calls = getattr(ai_msg, "tool_calls", []) or []
    results = {}

    for call in tool_calls:
        if call["name"] == "get_unit_price":
            vendor = call["args"]["vendor"]
            results[vendor] = get_unit_price.invoke({"vendor": vendor})

    # Fallback in case the model skips a vendor
    for vendor in vendor_names:
        results.setdefault(vendor, get_unit_price.invoke({"vendor": vendor}))

    quotes = []
    for vendor in vendor_names:
        data = results[vendor]
        quote = {
            "vendor": vendor,
            "product_name": data["product_name"],   # TASK 4: keep product forward in state
            "unit_price": data["unit_price"],
            "total": round(data["unit_price"] * quantity, 2),
            "delivery_days": 14,  # simple fixed value for the original UI text
            "delivery_info": data["delivery_info"],
        }
        quotes.append(quote)
        print(
            f"   {vendor}: €{quote['unit_price']}/unit x {quantity} = €{quote['total']:,.2f} "
            f"({quote['product_name']})"
        )

    return {"quotes": quotes, "quantity": quantity}


def compare_quotes(state: ProcurementState) -> dict:
    print("\n[Step 3] Comparing quotes...")
    time.sleep(0.5)
    best = min(state["quotes"], key=lambda q: q["total"])
    print(f"   Best quote: {best['vendor']} at €{best['total']:,.2f}")
    return {"best_quote": best}


# TASK 2: only ask approval when total > €10,000
def route_after_compare(state: ProcurementState) -> str:
    return "request_approval" if state["best_quote"]["total"] > 10_000 else "submit_purchase_order"


def request_approval(state: ProcurementState) -> dict:
    best = state["best_quote"]
    qty = state["quantity"]

    print("\n[Step 4] Order exceeds €10,000 — manager approval required!")
    print("   Sending approval request to manager...")

    decision = interrupt({
        "message": (
            f"Approve purchase of {qty} x {best['product_name']} "
            f"from {best['vendor']} for €{best['total']:,.2f}?"
        ),
        "vendor": best["vendor"],
        "product_name": best["product_name"],  # TASK 4
        "amount": best["total"],
    })

    print(f"\n[Step 4] Manager responded: {decision}")
    return {"approval_status": decision}


# TASK 3: approved continues, rejected skips PO creation
def route_after_approval(state: ProcurementState) -> str:
    return "notify_employee" if "reject" in state["approval_status"].lower() else "submit_purchase_order"


def submit_purchase_order(state: ProcurementState) -> dict:
    print("\n[Step 5] Submitting purchase order to ERP system...")
    time.sleep(1)
    po_number = "PO-2026-00342"
    print(f"   Purchase order created: {po_number}")
    print(f"   Vendor: {state['best_quote']['vendor']}")
    print(f"   Product: {state['best_quote']['product_name']}")
    print(f"   Amount: €{state['best_quote']['total']:,.2f}")
    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    print("\n[Step 6] Notifying employee...")

    qty = state["quantity"]
    best = state["best_quote"]

    # TASK 3: clear rejected path with reason
    if "reject" in state.get("approval_status", "").lower():
        reason = state["approval_status"]
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request for {qty} laptops was rejected by the manager. "
            f"Include this reason: {reason}. Be empathetic but concise."
        )
    else:
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request has been approved and processed. "
            f"Details: {qty} x {best['product_name']} from {best['vendor']}, "
            f"€{best['total']:,.2f}, PO number {state['po_number']}."
        )

    response = llm.invoke(prompt)
    notification = response.content
    print(f'   "{notification}"')
    return {"notification": notification}


# ─── Build the graph ─────────────────────────────────────────────────────────

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("fetch_pricing", fetch_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "fetch_pricing")
builder.add_edge("fetch_pricing", "compare_quotes")

# TASK 2: conditional interrupt
builder.add_conditional_edges(
    "compare_quotes",
    route_after_compare,
    {
        "request_approval": "request_approval",
        "submit_purchase_order": "submit_purchase_order",
    },
)

# TASK 3: conditional path after manager response
builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    {
        "submit_purchase_order": "submit_purchase_order",
        "notify_employee": "notify_employee",
    },
)

builder.add_edge("submit_purchase_order", "notify_employee")
builder.add_edge("notify_employee", END)


# ─── Checkpointer ────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints.db")
THREAD_ID = "procurement-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_first_invocation(graph):
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits purchase request")
    print("=" * 60)

    request = "Order 5 laptops for the new engineering team"
    print(f'\nEmployee request: "{request}"')

    graph.invoke({"request": request}, config)

    print("\n" + "=" * 60)
    print("AGENT SUSPENDED OR FINISHED")
    print("=" * 60)
    print(f"\nCheckpoint DB: {DB_PATH}")
    print(f"Thread ID: {THREAD_ID}")
    print(f"\nTo resume, run:")
    print(f"  python {os.path.basename(__file__)} --resume")
    print(f'  python {os.path.basename(__file__)} --resume "Rejected — over budget"\n')


def run_second_invocation(graph, resume_text: str):
    print("=" * 60)
    print("  SECOND INVOCATION — Manager responds")
    print("=" * 60)

    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found! Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Vendors found: {len(saved_state.values.get('vendors', []))}")
    print(f"  ✓ Quotes received: {len(saved_state.values.get('quotes', []))}")
    best = saved_state.values.get("best_quote", {})
    print(f"  ✓ Best quote: {best.get('vendor', 'N/A')} at €{best.get('total', 0):,.2f}")

    result = graph.invoke(Command(resume=resume_text), config)

    print("\n" + "=" * 60)
    print("PROCUREMENT COMPLETE")
    print("=" * 60)
    print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
    print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
    print(f"  Product:      {result.get('best_quote', {}).get('product_name', 'N/A')}")
    print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,.2f}")
    print(f"  Approval:     {result.get('approval_status', 'N/A')}")
    print()


if __name__ == "__main__":
    resume_mode = "--resume" in sys.argv

    if not resume_mode and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("(Cleaned up old checkpoint DB)")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    try:
        if resume_mode:
            resume_text = "Approved — go ahead with the purchase."
            extra_args = [arg for arg in sys.argv[1:] if arg != "--resume"]
            if extra_args:
                resume_text = " ".join(extra_args)
            run_second_invocation(graph, resume_text)
        else:
            run_first_invocation(graph)
    finally:
        conn.close()