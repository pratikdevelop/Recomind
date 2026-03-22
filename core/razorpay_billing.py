"""
core/razorpay_billing.py — Razorpay Subscriptions (India)
Supports: UPI, Cards, Net Banking, Wallets

Setup (free):
1. Sign up at https://razorpay.com (free, instant approval for Indian businesses)
2. Dashboard → Settings → API Keys → Generate Key
3. Add RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to .env
4. Create plans in Razorpay Dashboard → Subscriptions → Plans
"""

import os
import hmac
import hashlib
import logging
import requests
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID",     "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_PLAN_PRO   = os.getenv("RAZORPAY_PLAN_PRO",   "")   # plan_XXXXXXXX
RAZORPAY_PLAN_TEAM  = os.getenv("RAZORPAY_PLAN_TEAM",  "")

BASE_URL = "https://api.razorpay.com/v1"

# Plan prices in INR paise (1 INR = 100 paise)
PLAN_PRICES_INR = {
    "pro":  {"amount": 159900,  "label": "₹1,599/month"},   # ₹1,599
    "team": {"amount": 659900,  "label": "₹6,599/month"},   # ₹6,599
}


def _auth():
    return (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)


def _check_config():
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise RuntimeError(
            "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET not set.\n"
            "Sign up free at https://razorpay.com and get API keys from Dashboard → Settings → API Keys"
        )


# ════════════════════════════════════════════════════════════════════════════
# CREATE PLAN (run once via setup_razorpay.py)
# ════════════════════════════════════════════════════════════════════════════

def create_plan(name: str, amount: int, interval: int = 1,
                period: str = "monthly") -> Dict:
    """Create a Razorpay subscription plan."""
    _check_config()
    payload = {
        "period":   period,
        "interval": interval,
        "item": {
            "name":        name,
            "amount":      amount,
            "currency":    "INR",
            "description": name,
        }
    }
    print(f"Sending to Razorpay: {payload}")
    r = requests.post(f"{BASE_URL}/plans", auth=_auth(), json=payload)
    print(f"Razorpay response {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json()


# ════════════════════════════════════════════════════════════════════════════
# CREATE SUBSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

def create_subscription(
    plan_key:  str,
    user_id:   str,
    user_email: str,
    user_name:  str = "",
) -> Dict[str, Any]:
    """
    Create a Razorpay subscription.
    Returns subscription_id + short_url for redirect.
    """
    _check_config()

    plan_id = RAZORPAY_PLAN_PRO if plan_key == "pro" else RAZORPAY_PLAN_TEAM
    if not plan_id:
        raise ValueError(
            f"RAZORPAY_PLAN_{plan_key.upper()} not set. "
            "Run setup_razorpay.py to create plans."
        )

    payload = {
        "plan_id":         plan_id,
        "total_count":     120,          # max billing cycles (10 years)
        "quantity":        1,
        "notes": {
            "user_id":    user_id,
            "plan_key":   plan_key,
            "user_email": user_email,
        },
        "notify_info": {
            "notify_phone": None,
            "notify_email": user_email,
        },
    }

    r = requests.post(f"{BASE_URL}/subscriptions", auth=_auth(), json=payload)
    if not r.ok:
        raise RuntimeError(f"Razorpay {r.status_code}: {r.text}")

    data = r.json()
    return {
        "subscription_id": data["id"],
        "status":          data["status"],
        "short_url":       data.get("short_url", ""),
        "key_id":          RAZORPAY_KEY_ID,   # needed by frontend checkout
    }


def get_subscription(subscription_id: str) -> Dict:
    r = requests.get(f"{BASE_URL}/subscriptions/{subscription_id}", auth=_auth())
    r.raise_for_status()
    return r.json()


def cancel_subscription(subscription_id: str) -> bool:
    try:
        r = requests.post(
            f"{BASE_URL}/subscriptions/{subscription_id}/cancel",
            auth=_auth(),
            json={"cancel_at_cycle_end": 0},
        )
        return r.ok
    except Exception as exc:
        log.error(f"Razorpay cancel failed: {exc}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# WEBHOOK VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

def verify_webhook(body: bytes, signature: str) -> bool:
    """Verify Razorpay webhook signature."""
    if not RAZORPAY_WEBHOOK_SECRET:
        log.warning("RAZORPAY_WEBHOOK_SECRET not set — skipping verification")
        return True
    expected = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def get_config() -> Dict[str, Any]:
    return {
        "configured":  bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET),
        "key_id":      RAZORPAY_KEY_ID,
        "plans":       PLAN_PRICES_INR,
        "plan_pro":    bool(RAZORPAY_PLAN_PRO),
        "plan_team":   bool(RAZORPAY_PLAN_TEAM),
    }