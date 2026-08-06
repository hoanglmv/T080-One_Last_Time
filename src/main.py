from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print(f"Starting {settings.app_name} in {settings.app_env} mode")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Alternative Credit Scoring POC",
    description="Home Credit research scoring, local explanations and optional guarded LLM narration",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Alternative Credit Scoring POC API is running",
        "docs": "/docs",
        "health": "/health",
        "credit_demo": "/api/v1/credit/demo",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/ready")
async def readiness():
    from pathlib import Path

    model_ready = Path(settings.credit_model_path).is_file()
    return {
        "status": "ready" if model_ready else "degraded",
        "api": True,
        "credit_model": model_ready,
    }
