"""
core/auth/setup.py — FastAPI-Users wiring
JWT auth + MongoDB via Motor + Beanie
"""

import os
from typing import Optional, AsyncGenerator

import motor.motor_asyncio
from beanie import init_beanie, PydanticObjectId
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend, BearerTransport, JWTStrategy,
)
from fastapi_users.db import BeanieUserDatabase

from core.auth.models import User

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI  = os.getenv("MONGODB_URI", "")
DB_NAME      = os.getenv("DB_NAME",    "knowledge_base")
JWT_SECRET   = os.getenv("JWT_SECRET", "CHANGE-THIS-SECRET-IN-PRODUCTION-MIN-32-CHARS")
JWT_LIFETIME = int(os.getenv("JWT_LIFETIME_SECONDS", str(60 * 60 * 24 * 7)))

# ── Motor async client ────────────────────────────────────────────────────────
_motor_client = None
_motor_db     = None


def get_motor_db():
    global _motor_client, _motor_db
    if _motor_client is None:
        _motor_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
        _motor_db     = _motor_client[DB_NAME]
    return _motor_db


async def init_db() -> None:
    db = get_motor_db()
    await init_beanie(database=db, document_models=[User])


async def close_db() -> None:
    global _motor_client
    if _motor_client:
        _motor_client.close()
        _motor_client = None


# ── User DB dependency ────────────────────────────────────────────────────────
async def get_user_db() -> AsyncGenerator[BeanieUserDatabase, None]:
    yield BeanieUserDatabase(User)


# ── User Manager ──────────────────────────────────────────────────────────────
class UserManager(BaseUserManager[User, PydanticObjectId]):
    reset_password_token_secret = JWT_SECRET
    verification_token_secret   = JWT_SECRET

    def parse_id(self, value: str) -> PydanticObjectId:
        """Convert JWT string user_id back to PydanticObjectId."""
        return PydanticObjectId(value)

    async def on_after_register(self, user: User,
                                request: Optional[Request] = None):
        import logging
        log = logging.getLogger(__name__)
        log.info(f"New user registered: {user.email}")
        # Send verification email
        try:
            from core.email_service import send_verification_email
            token = await self.create_verification_token(user, request)
            sent  = send_verification_email(
                email    = user.email,
                token = token,
                # name  = user.full_name or "",
            )
            if not sent:
                log.info(f"[DEV] Verify token for {user.email}: {token}")
        except Exception as exc:
            log.warning(f"Could not send verification email: {exc}")

    async def on_after_forgot_password(self, user: User, token: str,
                                       request: Optional[Request] = None):
        import logging
        log = logging.getLogger(__name__)
        try:
            from core.email_service import send_password_reset_email
            sent = send_password_reset_email(
                to    = user.email,
                token = token,
                name  = user.full_name or "",
            )
            if not sent:
                log.info(f"[DEV] Password reset token for {user.email}: {token}")
        except Exception as exc:
            log.warning(f"Could not send reset email: {exc}")
            log.info(f"[DEV] Password reset token for {user.email}: {token}")

    async def on_after_verify(self, user: User,
                              request: Optional[Request] = None):
        import logging
        log = logging.getLogger(__name__)
        log.info(f"Email verified: {user.email}")
        try:
            from core.email_service import send_welcome_email
            send_welcome_email(to=user.email, name=user.full_name or "")
        except Exception as exc:
            log.warning(f"Could not send welcome email: {exc}")

    async def on_after_request_verify(self, user: User, token: str,
                                      request: Optional[Request] = None):
        import logging
        log = logging.getLogger(__name__)
        try:
            from core.email_service import send_verification_email
            sent = await send_verification_email(
                email    = user.email,
                token = token,
                # name  = user.full_name or "",
            )
            if not sent:
                log.info(f"[DEV] Verify token for {user.email}: {token}")
        except Exception as exc:
            log.warning(f"Could not send verification email: {exc}")
            log.info(f"[DEV] Verify token for {user.email}: {token}")


async def get_user_manager(
    user_db: BeanieUserDatabase = Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


# ── JWT Strategy ──────────────────────────────────────────────────────────────
def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=JWT_LIFETIME)


bearer_transport = BearerTransport(tokenUrl="/auth/login")

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# ── FastAPIUsers instance ─────────────────────────────────────────────────────
fastapi_users = FastAPIUsers[User, PydanticObjectId](
    get_user_manager, [auth_backend]
)

# ── Dependency shortcuts ──────────────────────────────────────────────────────
current_active_user   = fastapi_users.current_user(active=True)
current_optional_user = fastapi_users.current_user(optional=True)
current_superuser     = fastapi_users.current_user(active=True, superuser=True)