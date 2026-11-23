"""
Dataset Doctor: Authentication Routes.
Handles user login, credential verification, rate limiting, and logout.
"""
from pathlib import Path
from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.auth import (
    authenticate_user,
    create_authenticated_session,
    is_authenticated,
    revoke_token,
    get_auth_token_from_request,
)
from app.config import (
    AUTH_COOKIE_NAME,
    AUTH_TOKEN_EXPIRY_HOURS,
    RATE_LIMIT_LOGIN,
)
from app.security import rate_limiter, sanitize_text_input, get_client_ip

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def is_safe_redirect_url(url: str) -> bool:
    """Verifies that redirect URL is strictly local and cannot trigger protocol-relative open redirects."""
    if not url or not url.startswith("/"):
        return False
    if url.startswith("//") or url.startswith("/\\"):
        return False
    import re
    if re.search(r"[\r\n\t]", url):
        return False
    return True


@router.get("/login", response_class=HTMLResponse)
def get_login_page(request: Request, next: str = "/"):
    clean_next = next if is_safe_redirect_url(next) else "/"
    if is_authenticated(request):
        return RedirectResponse(url=clean_next, status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None,
            "next_url": clean_next,
            "active_filename": None,
            "is_logged_in": False,
        }
    )


@router.post("/login", response_class=HTMLResponse)
async def post_login(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
    next_url: str = Form("/"),
):
    ip = get_client_ip(request)
    allowed, retry_after = rate_limiter.is_allowed(f"login:{ip}", RATE_LIMIT_LOGIN, 60)
    if not allowed:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": f"Too many login attempts. Please wait {retry_after} seconds before retrying.",
                "next_url": next_url,
                "active_filename": None,
                "is_logged_in": False,
            },
            status_code=429,
        )

    clean_user = sanitize_text_input(username, 100)
    is_valid = authenticate_user(clean_user, password)

    if not is_valid:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": "Invalid username or password. Access is restricted to authorized project members.",
                "next_url": next_url,
                "active_filename": None,
                "is_logged_in": False,
            },
            status_code=401,
        )

    # Valid login -> create token & set cookie
    token = create_authenticated_session()
    safe_next = next_url if is_safe_redirect_url(next_url) else "/"
    
    response = RedirectResponse(url=safe_next, status_code=303)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=AUTH_TOKEN_EXPIRY_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.post("/login/guest")
def post_guest_login(request: Request, next_url: str = Form("/")):
    """Instant 1-click guest authentication for recruiters, demo visitors, and public showcases."""
    token = create_authenticated_session()
    safe_next = next_url if is_safe_redirect_url(next_url) else "/"
    response = RedirectResponse(url=safe_next, status_code=303)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=AUTH_TOKEN_EXPIRY_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/demo")
def get_demo_access(request: Request):
    """Direct 1-click URL link for LinkedIn viewers & demo visitors."""
    token = create_authenticated_session()
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=AUTH_TOKEN_EXPIRY_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/logout")
def get_logout(request: Request):
    token = get_auth_token_from_request(request)
    revoke_token(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=AUTH_COOKIE_NAME)
    return response
