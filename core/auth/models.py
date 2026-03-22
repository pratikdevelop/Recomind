"""
core/auth/models.py — User model + DB schema
Uses Motor (async MongoDB driver) + Beanie ODM
"""

import os
from typing import Optional
from datetime import datetime
from beanie import Document
from fastapi_users.db import BeanieBaseUser
from pydantic import Field

# ── Plan limits ───────────────────────────────────────────────────────────────
PLAN_LIMITS = {
    "free":  {"queries": 50,      "docs": 5,       "users": 1},
    "pro":   {"queries": 999_999, "docs": 100,     "users": 1},
    "team":  {"queries": 999_999, "docs": 999_999, "users": 10},
}


class User(BeanieBaseUser, Document):
    """MongoDB user document — extends FastAPI-Users base."""

    # Profile
    full_name:   Optional[str] = None
    avatar_url:  Optional[str] = None

    # Plan & billing
    plan:                str           = "free"
    paypal_sub_id:       Optional[str] = None   # PayPal subscription ID
    plan_expires:        Optional[datetime] = None

    # Usage counters (reset monthly)
    queries_this_month: int      = 0
    usage_reset_at:     datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name            = "users"
        email_collation = None   # required by fastapi-users-db-beanie >= 2.0

    def within_limits(self, resource: str) -> bool:
        limit = PLAN_LIMITS.get(self.plan, PLAN_LIMITS["free"]).get(resource, 0)
        if resource == "queries":
            return self.queries_this_month < limit
        return True

    def plan_limit(self, resource: str) -> int:
        return PLAN_LIMITS.get(self.plan, PLAN_LIMITS["free"]).get(resource, 0)