# core/auth/__init__.py
from core.auth.setup import (
    fastapi_users,
    auth_backend,
    current_active_user,
    current_optional_user,
    current_superuser,
    init_db,
    close_db,
)
from core.auth.models  import User, PLAN_LIMITS
from core.auth.schemas import UserRead, UserCreate, UserUpdate

__all__ = [
    "fastapi_users", "auth_backend",
    "current_active_user", "current_optional_user", "current_superuser",
    "init_db", "close_db",
    "User", "PLAN_LIMITS",
    "UserRead", "UserCreate", "UserUpdate",
]