"""
Dataset Doctor: Authentication & Session Token Service.
Manages credentials, password hashing, and token validation.
"""
import hmac
import secrets
import time
from typing import Dict, Optional
from fastapi import Request

from app.config import (
    AUTH_USERNAME,
    AUTH_PASSWORD,
    AUTH_COOKIE_NAME,
    AUTH_TOKEN_EXPIRY_HOURS,
)
from app.security import hash_password, verify_password

# Pre-hash the temporary project password on startup
_HASHED_SYSTEM_PASSWORD = hash_password(AUTH_PASSWORD)

# Token store: token_str -> expiry_epoch_timestamp
_ACTIVE_AUTH_TOKENS: Dict[str, float] = {}


def authenticate_user(username: str, password: str) -> bool:
    """
    Validates username and password in constant time to prevent timing attacks.
    """
    if not username or not password:
        return False

    # Check username in constant time
    is_user_valid = hmac.compare_digest(username.strip(), AUTH_USERNAME)

    # Verify password against securely hashed password
    is_pass_valid = verify_password(password, _HASHED_SYSTEM_PASSWORD)

    return is_user_valid and is_pass_valid


def create_authenticated_session() -> str:
    """
    Generates a cryptographically strong session token and records its expiration.
    """
    cleanup_auth_tokens()
    token = secrets.token_urlsafe(32)
    expiry = time.time() + (AUTH_TOKEN_EXPIRY_HOURS * 3600)
    _ACTIVE_AUTH_TOKENS[token] = expiry
    return token


def is_token_valid(token: Optional[str]) -> bool:
    """Checks whether an auth token is active and not expired."""
    if not token or token not in _ACTIVE_AUTH_TOKENS:
        return False
    expiry = _ACTIVE_AUTH_TOKENS[token]
    if time.time() > expiry:
        del _ACTIVE_AUTH_TOKENS[token]
        return False
    return True


def revoke_token(token: Optional[str]):
    """Revokes an active authentication token."""
    if token and token in _ACTIVE_AUTH_TOKENS:
        del _ACTIVE_AUTH_TOKENS[token]


def get_auth_token_from_request(request: Request) -> Optional[str]:
    """Retrieves auth token from cookie or Authorization header."""
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return None


def is_authenticated(request: Request) -> bool:
    """Returns True if the current request has a valid, active authentication token."""
    token = get_auth_token_from_request(request)
    return is_token_valid(token)


def cleanup_auth_tokens():
    """Prunes expired authentication tokens."""
    now = time.time()
    expired = [t for t, exp in _ACTIVE_AUTH_TOKENS.items() if now > exp]
    for t in expired:
        del _ACTIVE_AUTH_TOKENS[t]
