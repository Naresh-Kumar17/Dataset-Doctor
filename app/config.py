"""
Dataset Doctor: Central Configuration Module.
Loads environment variables from .env if available, with secure defaults.
"""
import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Application & Debug Mode
DEBUG = os.getenv("DEBUG", "False").strip().lower() in ("true", "1", "yes")
ENV = os.getenv("APP_ENV", "production" if not DEBUG else "development")

# Cryptographic Secret Key for cookies and tokens
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    # Generate persistent or fallback session key
    SECRET_KEY = secrets.token_hex(32)

# Project Authentication Credentials
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")
if not AUTH_PASSWORD:
    if ENV == "production":
        AUTH_PASSWORD = secrets.token_urlsafe(16)
    else:
        AUTH_PASSWORD = "dev-dataset-doctor-pass"
AUTH_COOKIE_NAME = "dataset_doctor_auth"
AUTH_TOKEN_EXPIRY_HOURS = int(os.getenv("AUTH_TOKEN_EXPIRY_HOURS", "24"))

# CORS Configuration
default_origins = "http://localhost:8000,http://127.0.0.1:8000,https://dataset-doctor-8c7b.onrender.com"
raw_origins = os.getenv("ALLOWED_ORIGINS", default_origins)
ALLOWED_ORIGINS = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]

# Upload & File Limits
DEFAULT_MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", str(DEFAULT_MAX_FILE_SIZE_MB)))

# Rate Limiting (requests per minute per IP)
RATE_LIMIT_LOGIN = int(os.getenv("RATE_LIMIT_LOGIN", "5"))
RATE_LIMIT_UPLOAD = int(os.getenv("RATE_LIMIT_UPLOAD", "15"))
RATE_LIMIT_GLOBAL = int(os.getenv("RATE_LIMIT_GLOBAL", "120"))
