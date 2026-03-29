"""
AgroNexus — Precision Agriculture OpenEnv
FastAPI entry point with UI serving
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="AgroNexus — Precision Agriculture OpenEnv",
    description="OpenEnv-compliant RL environment for precision agriculture.",
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

# API routes MUST come before static file mount
app.include_router(router)

# Serve UI pages as named routes
@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse("ui/index.html")

@app.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse("ui/demo.html")

@app.get("/control", include_in_schema=False)
def control_page():
    return FileResponse("ui/control.html")

# Root serves the main UI dashboard
@app.get("/", include_in_schema=False)
def root():
    if os.path.exists("ui/index.html"):
        return FileResponse("ui/index.html")
    return JSONResponse({
        "name":    "agronexus",
        "version": "1.0.0",
        "status":  "running",
        "tasks":   ["task_1_easy", "task_2_medium", "task_3_hard"],
        "docs":    "/docs",
        "openenv": "openenv.yaml",
    })

# Mount static files LAST — after all routes
if os.path.exists("ui"):
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
