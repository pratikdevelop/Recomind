"""
core/billing.py — PayPal Subscriptions + One-time payments
Uses PayPal REST API v2 (works in India, no Stripe needed)

Setup:
1. Create app at https://developer.paypal.com/dashboard/applications
2. Get Client ID + Secret (Sandbox for testing, Live for production)
3. Create subscription plans in PayPal dashboard or via API
4. Add PAYPAL_CLIENT_ID, PAYPAL_SECRET, PAYPAL_PLAN_* to .env
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET    = os.getenv("PAYPAL_SECRET",    "")
PAYPAL_MODE      = os.getenv("PAYPAL_MODE",      "sandbox")   # sandbox | live

# PayPal Plan IDs — create these in your PayPal dashboard
# Dashboard → Subscriptions → Plans → Create Plan
PAYPAL_PLAN_PRO  = os.getenv("PAYPAL_PLAN_PRO",  "")   # e.g. P-XXXXXXXXXXXXXXXX
PAYPAL_PLAN_TEAM = os.getenv("PAYPAL_PLAN_TEAM", "")

# Prices shown in UI (must match your PayPal plan prices)
PLAN_PRICES = {
    "pro":  {"price": "19.00", "currency": "USD", "label": "Pro",  "billing": "monthly"},
    "team": {"price": "79.00", "currency": "USD", "label": "Team", "billing": "monthly"},
}

BASE_URL = (
    "https://api-m.sandbox.paypal.com" if PAYPAL_MODE == "sandbox"
    else "https://api-m.paypal.com"
)

# ── Auth token (cached) ───────────────────────────────────────────────────────
_token: Optional[str]   = None
_token_expiry: datetime = datetime.utcnow()


def _get_access_token() -> str:
    global _token, _token_expiry
    if _token and datetime.utcnow() < _token_expiry:
        return _token

    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
        raise RuntimeError(
            "PAYPAL_CLIENT_ID and PAYPAL_SECRET not set in .env\n"
            "Get them at: https://developer.paypal.com/dashboard/applications"
        )

    resp = requests.post(
        f"{BASE_URL}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    resp.raise_for_status()
    data          = resp.json()
    _token        = data["access_token"]
    _token_expiry = datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60)
    return _token


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_access_token()}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


# ════════════════════════════════════════════════════════════════════════════
# SUBSCRIPTION FLOW
# Step 1: create_subscription_link()  → send user to PayPal approval page
# Step 2: PayPal redirects back       → capture_subscription() verifies & saves
# ════════════════════════════════════════════════════════════════════════════

def create_subscription_link(
    plan_key:    str,          # "pro" | "team"
    user_id:     str,
    return_url:  str,          # e.g. https://yourdomain.com/billing/success
    cancel_url:  str,          # e.g. https://yourdomain.com/pricing
) -> Dict[str, Any]:
    """
    Create a PayPal subscription and return the approval URL.
    Redirect the user to data['approve_url'] to complete payment.
    """
    plan_id = PAYPAL_PLAN_PRO if plan_key == "pro" else PAYPAL_PLAN_TEAM
    if not plan_id:
        raise ValueError(
            f"PAYPAL_PLAN_{plan_key.upper()} not set in .env. "
            "Create a plan in your PayPal dashboard first."
        )

    payload = {
        "plan_id":   plan_id,
        "custom_id": user_id,          # we send user_id so webhook can find the user
        "application_context": {
            "brand_name":          "RecoMind",
            "locale":              "en-IN",
            "shipping_preference": "NO_SHIPPING",
            "user_action":         "SUBSCRIBE_NOW",
            "payment_method": {
                "payer_selected":   "PAYPAL",
                "payee_preferred":  "IMMEDIATE_PAYMENT_REQUIRED",
            },
            "return_url": return_url,
            "cancel_url": cancel_url,
        },
    }

    log.info(f"Creating PayPal subscription — plan_id={plan_id}, user_id={user_id}")
    log.info(f"Payload: {payload}")
    resp = requests.post(
        f"{BASE_URL}/v1/billing/subscriptions",
        headers=_headers(),
        json=payload,
        timeout=20,
    )
    log.info(f"PayPal response: {resp.status_code} — {resp.text[:500]}")
    if not resp.ok:
        error_body = resp.json() if resp.content else {}
        raise RuntimeError(
            f"PayPal {resp.status_code}: {error_body.get('message', resp.text)}"
            f" | details: {error_body.get('details', '')}"
            f" | debug_id: {resp.headers.get('PayPal-Debug-Id','')}"
        )
    data = resp.json()

    approve_url = next(
        (link["href"] for link in data.get("links", []) if link["rel"] == "approve"),
        None,
    )
    return {
        "subscription_id": data["id"],
        "status":          data["status"],
        "approve_url":     approve_url,
    }


def get_subscription_details(subscription_id: str) -> Dict[str, Any]:
    """Fetch subscription status from PayPal — call after user returns from approval."""
    resp = requests.get(
        f"{BASE_URL}/v1/billing/subscriptions/{subscription_id}",
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def cancel_subscription(subscription_id: str, reason: str = "User requested") -> bool:
    """Cancel a PayPal subscription."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/billing/subscriptions/{subscription_id}/cancel",
            headers=_headers(),
            json={"reason": reason},
            timeout=15,
        )
        return resp.status_code == 204
    except Exception as exc:
        log.error(f"Cancel subscription failed: {exc}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# WEBHOOK VERIFICATION
# PayPal calls your /api/billing/webhook endpoint on payment events
# ════════════════════════════════════════════════════════════════════════════

def verify_webhook(
    headers: Dict[str, str],
    body:    bytes,
    webhook_id: str,
) -> bool:
    """
    Verify that a webhook came from PayPal (not a spoofed request).
    webhook_id: from PayPal dashboard → Webhooks → your endpoint ID
    """
    payload = {
        "auth_algo":         headers.get("paypal-auth-algo", ""),
        "cert_url":          headers.get("paypal-cert-url",  ""),
        "transmission_id":   headers.get("paypal-transmission-id",  ""),
        "transmission_sig":  headers.get("paypal-transmission-sig", ""),
        "transmission_time": headers.get("paypal-transmission-time",""),
        "webhook_id":        webhook_id,
        "webhook_event":     body.decode("utf-8"),
    }
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/notifications/verify-webhook-signature",
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        return resp.json().get("verification_status") == "SUCCESS"
    except Exception as exc:
        log.error(f"Webhook verification failed: {exc}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# PLAN HELPERS
# ════════════════════════════════════════════════════════════════════════════

def plan_key_from_paypal_plan_id(paypal_plan_id: str) -> Optional[str]:
    if paypal_plan_id == PAYPAL_PLAN_PRO:
        return "pro"
    if paypal_plan_id == PAYPAL_PLAN_TEAM:
        return "team"
    return None


def get_plan_info() -> Dict[str, Any]:
    return {
        "mode":   PAYPAL_MODE,
        "plans":  PLAN_PRICES,
        "configured": bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET),
    }