"""User domain entity.

Represents the authenticated principal resolved from a JWT bearer token. Only
the minimal fields required by the auth-core layer are modelled here; the
upstream IdP (Logto) payload may contain many more claims which Pydantic
silently ignores thanks to ``extra="ignore"``.
"""

from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    """Authenticated user resolved from a JWT payload.

    Attributes:
        sub: Subject identifier (JWT ``sub`` claim) — the only required field.
        email: User email (optional — may be absent for service accounts).
        name: User full name (optional).
        username: Username (optional).
    """

    model_config = ConfigDict(extra="ignore")

    sub: str
    email: str | None = None
    name: str | None = None
    username: str | None = None
