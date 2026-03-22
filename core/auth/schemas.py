"""
core/auth/schemas.py
"""

from typing import Optional
from beanie import PydanticObjectId
from fastapi_users import schemas
from pydantic import ConfigDict, field_serializer


class UserRead(schemas.BaseUser[PydanticObjectId]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    full_name:          Optional[str] = None
    plan:               str           = "free"
    queries_this_month: int           = 0

    @field_serializer("id")
    def serialize_id(self, v) -> str:
        return str(v)


class UserCreate(schemas.BaseUserCreate):
    full_name: Optional[str] = None


class UserUpdate(schemas.BaseUserUpdate):
    full_name: Optional[str] = None