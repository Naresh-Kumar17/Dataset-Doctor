"""
Dataset Doctor: Security Utilities.
Provides password hashing, input sanitization, path traversal protection,
and in-memory sliding-window rate limiting.
"""
from collections import defaultdict, deque
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Dict, Tuple

# -------------------------------------------------------------------------
# Password Hashing & Verification (PBKDF2-HMAC-SHA256 with random salt)
# -------------------------------------------------------------------------
HASH_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    """Hashes a password with a fresh random 16-byte salt using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS
    )
    return f"pbkdf2:sha256:{HASH_ITERATIONS}${salt.hex()}${key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verifies a plain password against a stored hashed string in constant time."""
    try:
        parts = hashed.split("$")
        if len(parts) != 3:
            return False
        meta, salt_hex, key_hex = parts
        algorithm, iterations_str = meta.split(":")[-2:]
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)

        actual_key = hashlib.pbkdf2_hmac(
            algorithm,
            password.encode("utf-8"),
            salt,
            iterations
        )
        return hmac.compare_digest(actual_key, expected_key)
    except Exception:
        return False


# -------------------------------------------------------------------------
# Input & Filename Sanitization (Anti-Path-Traversal & Anti-XSS)
# -------------------------------------------------------------------------
def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an uploaded filename to prevent directory traversal and injection.
    Strips directory separators, null bytes, and non-printable characters.
    """
    if not filename:
        return "dataset.csv"

    # Extract basename only (strips any ../ or ..\ paths)
    base = Path(filename).name

    # Remove null bytes and control chars
    base = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", base)

    # Allow alphanumeric, dashes, underscores, dots, and spaces
    clean = re.sub(r"[^\w\s\.-]", "_", base).strip()

    if not clean:
        clean = "dataset.csv"

    # Limit length
    if len(clean) > 120:
        stem = Path(clean).stem[:110]
        suffix = Path(clean).suffix or ".csv"
        clean = f"{stem}{suffix}"

    return clean


def sanitize_text_input(val: str, max_length: int = 200) -> str:
    """Sanitizes plain form text inputs to prevent XSS and control character injection."""
    if not val:
        return ""
    # Strip null bytes and control characters
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", val)
    return cleaned.strip()[:max_length]


# -------------------------------------------------------------------------
# Sliding-Window Rate Limiter (Thread-safe, In-Memory)
# -------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    """
    Lightweight, thread-safe in-memory sliding-window rate limiter per client IP.
    """
    def __init__(self):
        self._lock = threading.Lock()
        # key -> deque of epoch timestamps
        self._requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> Tuple[bool, int]:
        """
        Returns (is_allowed, seconds_until_reset).
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            q = self._requests[key]
            # Prune old timestamps
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= max_requests:
                earliest = q[0]
                retry_after = max(1, int(earliest + window_seconds - now))
                return False, retry_after

            q.append(now)
            return True, 0

    def cleanup(self, window_seconds: int = 300):
        """Purges stale keys with no recent activity."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            stale_keys = [
                k for k, q in self._requests.items()
                if not q or q[-1] <= cutoff
            ]
            for k in stale_keys:
                del self._requests[k]


rate_limiter = SlidingWindowRateLimiter()
