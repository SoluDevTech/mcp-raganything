"""AuthContext domain entity.

The result of a successful authentication. Carries the resolved user
identifier, the authentication method that produced it and the raw credential
string (JWT token value or API key) for downstream RLS / audit wiring.
"""

from typing import Literal

from pydantic import BaseModel


class AuthContext(BaseModel):
    """Result of authenticating an incoming request.

    Attributes:
        user_id: Identifier of the authenticated principal (JWT ``sub`` or the
            user_id returned by the API key repository).
        method: Authentication method that produced this context.
        raw_credential: The raw credential value as received (JWT token without
            the ``Bearer `` prefix, or the API key plaintext). Used to set the
            RLS contextvar for audit / row-level security.
    """

    user_id: str
    method: Literal["jwt", "api_key"]
    raw_credential: str
