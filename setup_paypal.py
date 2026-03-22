"""
setup_paypal.py — Run ONCE to create PayPal products + plans
Usage: py setup_paypal.py

Copy the Plan IDs from output → paste into .env
"""

import json
import requests
from dotenv import load_dotenv

load_dotenv()

from core.billing import _headers, BASE_URL

def pp(data):
    print(json.dumps(data, indent=2))

# ── Step 1: Create Product ────────────────────────────────────────────────────
print("\n=== Creating Product ===")
r = requests.post(f"{BASE_URL}/v2/catalogs/products", headers=_headers(), json={
    "name":        "RecoMind",
    "description": "AI-powered recommendation engine",
    "type":        "SERVICE",
    "category":    "SOFTWARE",
})
product = r.json()
pp(product)

product_id = product.get("id")
if not product_id:
    print("❌ Product creation failed — check error above")
    exit(1)
print(f"\n✓ Product ID: {product_id}")

# ── Step 2: Create Pro Plan ───────────────────────────────────────────────────
print("\n=== Creating Pro Plan ($19/month) ===")
r = requests.post(f"{BASE_URL}/v1/billing/plans", headers=_headers(), json={
    "product_id":  product_id,
    "name":        "RecoMind Pro",
    "description": "Unlimited queries, 100 documents, Recommendation mode",
    "status":      "ACTIVE",
    "billing_cycles": [{
        "frequency":      {"interval_unit": "MONTH", "interval_count": 1},
        "tenure_type":    "REGULAR",
        "sequence":       1,
        "total_cycles":   0,
        "pricing_scheme": {
            "fixed_price": {"value": "19", "currency_code": "USD"}
        }
    }],
    "payment_preferences": {
        "auto_bill_outstanding":     True,
        "setup_fee_failure_action":  "CONTINUE",
        "payment_failure_threshold": 1,
    }
})
pro_plan = r.json()
pp(pro_plan)
pro_plan_id = pro_plan.get("id")
print(f"\n✓ Pro Plan ID: {pro_plan_id}")

# ── Step 3: Create Team Plan ──────────────────────────────────────────────────
print("\n=== Creating Team Plan ($79/month) ===")
r = requests.post(f"{BASE_URL}/v1/billing/plans", headers=_headers(), json={
    "product_id":  product_id,
    "name":        "RecoMind Team",
    "description": "Unlimited queries, unlimited documents, up to 10 users",
    "status":      "ACTIVE",
    "billing_cycles": [{
        "frequency":      {"interval_unit": "MONTH", "interval_count": 1},
        "tenure_type":    "REGULAR",
        "sequence":       1,
        "total_cycles":   0,
        "pricing_scheme": {
            "fixed_price": {"value": "79", "currency_code": "USD"}
        }
    }],
    "payment_preferences": {
        "auto_bill_outstanding":     True,
        "setup_fee_failure_action":  "CONTINUE",
        "payment_failure_threshold": 1,
    }
})
team_plan = r.json()
pp(team_plan)
team_plan_id = team_plan.get("id")
print(f"\n✓ Team Plan ID: {team_plan_id}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("✅ Add these to your .env:")
print("="*60)
print(f"PAYPAL_PLAN_PRO={pro_plan_id}")
print(f"PAYPAL_PLAN_TEAM={team_plan_id}")
print("="*60)