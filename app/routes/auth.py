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
from app.security import rate_limiter, sanitize_text_input

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def get_client_ip(request: Request) -> str:
    """Extracts client IP, respecting X-Forwarded-For if behind a proxy like Render."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("/login", response_class=HTMLResponse)
def get_login_page(request: Request, next: str = "/"):
    if is_authenticated(request):
        return RedirectResponse(url=next or "/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None,
            "next_url": next,
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
    safe_next = next_url if (next_url and next_url.startswith("/") and not next_url.startswith("//")) else "/"
    
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


@router.get("/logout")
def get_logout(request: Request):
    token = get_auth_token_from_request(request)
    revoke_token(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=AUTH_COOKIE_NAME)
    return response
