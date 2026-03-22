"""
setup_razorpay.py — Run ONCE to create Razorpay subscription plans
Usage: py setup_razorpay.py

Copy Plan IDs from output → paste into .env
"""

import json
from dotenv import load_dotenv
load_dotenv()

from core.razorpay_billing import create_plan, _auth, BASE_URL
import requests

print("\n=== Creating Razorpay Plans ===\n")

# Pro Plan — ₹1,599/month
print("Creating Pro Plan (₹1,599/month)...")
pro = create_plan(
    name="RecoMind Pro",
    amount=159900,    # in paise
    period="monthly",
    interval=1,
)
print(json.dumps(pro, indent=2))
pro_plan_id = pro["id"]
print(f"\n✓ Pro Plan ID: {pro_plan_id}")

# Team Plan — ₹6,599/month
print("\nCreating Team Plan (₹6,599/month)...")
team = create_plan(
    name="RecoMind Team",
    amount=659900,
    period="monthly",
    interval=1,
)
print(json.dumps(team, indent=2))
team_plan_id = team["id"]
print(f"\n✓ Team Plan ID: {team_plan_id}")

print("\n" + "="*60)
print("✅ Add these to your .env:")
print("="*60)
print(f"RAZORPAY_PLAN_PRO={pro_plan_id}")
print(f"RAZORPAY_PLAN_TEAM={team_plan_id}")
print("="*60)