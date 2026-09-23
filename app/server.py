"""
IPMA Web Server — FastAPI Backend
==================================
Serves the Industrial Plant Monitoring Agent (IPMA) Web UI, providing
REST endpoints for real-time fleet telemetry, engineering diagnostics,
and interactive agent chat sessions powered by Gemini.
"""

import os
import sys
import time
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.core import (
    IndustrialAgent,
    list_equipment,
    get_equipment_snapshot,
    analyze_fouling,
    analyze_bearing,
)

logger = logging.getLogger("ipma.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Path to static frontend assets
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="IPMA — Industrial Plant Monitoring Agent",
    description="Real-time predictive maintenance and diagnostics platform",
    version="1.0.0",
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Agent Lifecycle
# ---------------------------------------------------------------------------
_agent_instance: Optional[IndustrialAgent] = None


def get_agent() -> IndustrialAgent:
    """Lazily initialize or return the singleton IndustrialAgent."""
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = IndustrialAgent(verbose=False)
            logger.info("IndustrialAgent initialized successfully.")
        except Exception as e:
            logger.warning(f"IndustrialAgent initialization deferred/failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Agent initialization error: {str(e)}. Check your GOOGLE_API_KEY in .env.",
            )
    return _agent_instance


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Diagnostic query or instruction")


class ChatResponse(BaseModel):
    response: str
    timestamp: float
    model: str


class ResetResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint returning system status and model info."""
    api_key_set = bool(os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_API_KEY") != "your_google_api_key_here")
    model_name = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    return {
        "status": "healthy",
        "service": "IPMA Industrial Monitoring API",
        "model": model_name,
        "api_key_configured": api_key_set,
        "timestamp": time.time(),
    }


@app.get("/api/fleet")
async def get_fleet_status():
    """
    Returns real-time equipment catalog and current health snapshots
    directly without requiring an LLM call.
    """
    try:
        catalog = list_equipment()
        snapshot = get_equipment_snapshot()
        return {
            "catalog": catalog,
            "snapshot": snapshot,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error fetching fleet status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/equipment/hx/{exchanger_id}")
async def get_hx_detail(exchanger_id: str):
    """Retrieve fouling analysis and operating status for a specific heat exchanger."""
    exchanger_id = exchanger_id.upper()
    if exchanger_id not in ["E01", "E02", "E03", "E04", "E05"]:
        raise HTTPException(status_code=404, detail=f"Exchanger {exchanger_id} not found.")
    try:
        result = analyze_fouling(exchanger_id=exchanger_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/equipment/bearing/{test_id}/{bearing_id}")
async def get_bearing_detail(test_id: int, bearing_id: str):
    """Retrieve vibration analysis and ISO status for a specific bearing."""
    if test_id not in [1, 2, 3]:
        raise HTTPException(status_code=404, detail=f"Bearing Test {test_id} not found.")
    try:
        result = analyze_bearing(test_id=test_id, bearing_id=bearing_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(req: ChatRequest):
    """
    Send natural language question to IPMA agent with automatic tool calling.
    """
    agent = get_agent()
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        response_text = agent.ask(user_msg)
        return ChatResponse(
            response=response_text,
            timestamp=time.time(),
            model=agent.model_name,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.post("/api/reset", response_model=ResetResponse)
async def reset_session():
    """Reset the agent's chat history for a fresh diagnostic session."""
    agent = get_agent()
    try:
        agent.reset()
        return ResetResponse(
            status="reset",
            message="Diagnostic session reset. Agent memory is cleared.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Static Assets and SPA Frontend Route
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the primary HTML application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "Frontend index.html not found. Please compile/create static assets."},
        )
    return FileResponse(str(index_file))


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start uvicorn server programmatically."""
    import uvicorn
    print(f"\n=======================================================")
    print(f"  IPMA — Industrial Plant Monitoring Agent Dashboard   ")
    print(f"  Local URL: http://{host}:{port}                      ")
    print(f"=======================================================\n")
    uvicorn.run("app.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
