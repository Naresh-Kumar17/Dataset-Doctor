"""
Dataset Doctor: Main FastAPI Application.
Wires routes, static asset serving, session management, and comprehensive security hardening:
- Authentication & Route Guarding
- Security Headers & CORS
- IP-based Sliding-Window Rate Limiting
- Debug Mode Suppression & Safe Error Shielding
- Input Sanitization & Path Traversal Protection
"""
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import (
    DEBUG,
    ALLOWED_ORIGINS,
    RATE_LIMIT_GLOBAL,
)
from app.auth import is_authenticated
from app.security import rate_limiter
from app.routes.auth import router as auth_router
from app.routes.upload import router as upload_router
from app.routes.bench import router as bench_router
from app.routes.issues import router as issues_router
from app.routes.decide import router as decide_router

# Setup safe server-side logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_doctor")

# Instantiate FastAPI with debug mode disabled to hide stack traces in production
app = FastAPI(
    title="Dataset Doctor",
    debug=False,
    docs_url="/docs" if DEBUG else None,
    redoc_url=None,
    openapi_url="/openapi.json" if DEBUG else None,
)

# -------------------------------------------------------------------------
# CORS Configuration
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

# -------------------------------------------------------------------------
# Security & Authentication Middleware
# -------------------------------------------------------------------------
PUBLIC_EXACT_PATHS = {"/login", "/logout", "/health"}
PUBLIC_PREFIXES = ("/static",)


class SecurityAndAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        client_ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        # 1. Global IP Rate Limiting Check
        is_allowed, retry_after = rate_limiter.is_allowed(f"global:{client_ip}", RATE_LIMIT_GLOBAL, 60)
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Too many requests. Please wait.", "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)},
            )

        # 2. Authentication Check
        authenticated = is_authenticated(request)
        request.state.is_authenticated = authenticated

        is_public = path in PUBLIC_EXACT_PATHS or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)

        if not is_public and not authenticated:
            # Check if this is an API/JSON request or browser navigation
            accept = request.headers.get("accept", "")
            is_browser = "text/html" in accept

            if is_browser:
                next_param = f"?next={path}" if path not in ("/", "") else ""
                return RedirectResponse(url=f"/login{next_param}", status_code=303)
            else:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required. Please log in.", "login_url": "/login"},
                )

        # 3. Process Request
        response = await call_next(request)

        # 4. Attach Modern HTTP Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        return response


app.add_middleware(SecurityAndAuthMiddleware)

# -------------------------------------------------------------------------
# Global Exception Handler (Hides stack traces and internals)
# -------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception at %s: %s", request.url.path, exc, exc_info=True)
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        html_content = (
            "<!DOCTYPE html><html><head><title>System Error — Dataset Doctor</title>"
            "<link rel='stylesheet' href='/static/style.css'></head>"
            "<body style='padding: 40px;'><div class='card' style='max-width: 600px; margin: 40px auto;'>"
            "<div class='chip high' style='margin-bottom: 12px;'>SERVER ERROR</div>"
            "<h2>An unexpected error occurred</h2>"
            "<p>The request could not be completed safely. Our diagnostics have logged the issue.</p>"
            "<a href='/' class='btn' style='margin-top: 16px;'>Return Home</a>"
            "</div></body></html>"
        )
        return HTMLResponse(content=html_content, status_code=500)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred. Please try again later."},
    )

# -------------------------------------------------------------------------
# Static File Serving
# -------------------------------------------------------------------------
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# -------------------------------------------------------------------------
# Register Routers
# -------------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(bench_router)
app.include_router(issues_router)
app.include_router(decide_router)

# -------------------------------------------------------------------------
# Health Check Endpoint (Public for Liveness & Monitoring)
# -------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "Dataset Doctor is running"}
