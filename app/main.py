"""
Dataset Doctor: Main FastAPI Application.
Wires routes, static asset serving, templates, and session management.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.upload import router as upload_router
from app.routes.bench import router as bench_router
from app.routes.issues import router as issues_router

app = FastAPI(title="Dataset Doctor")

# Static files mounting
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(upload_router)
app.include_router(bench_router)
app.include_router(issues_router)


@app.get("/health")
def health_check():
    return {"status": "Dataset Doctor is running"}

