"""
AgroNexus — FastAPI entry point.
Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="AgroNexus",
    description=(
        "An OpenEnv-compliant reinforcement learning environment "
        "where an AI agent manages a simulated farm across a 90-day "
        "growing season. Supports 3 tasks: easy, medium, hard."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

from fastapi.staticfiles import StaticFiles
import os

if not os.path.exists("ui"):
    os.makedirs("ui")

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/")
def root():
    """Health check — returns environment info."""
    return {
        "name":        "agronexus",
        "version":     "1.0.0",
        "status":      "running",
        "tasks":       ["task_1_easy", "task_2_medium", "task_3_hard"],
        "docs":        "/docs",
        "openenv":     "openenv.yaml",
    }
