"""
Medical Prescription Validation Environment - FastAPI Server
================================================================================

This creates the web server that hosts the environment.

Endpoints created automatically by OpenEnv:
- /ws          - WebSocket endpoint (primary)
- /reset       - HTTP POST to reset environment
- /step        - HTTP POST to take action
- /state       - HTTP GET current state
- /health      - HTTP GET health check
- /docs        - Interactive API documentation

Author: Suhel Mulla
"""

import sys
import os

# Ensure project root is on the path so models.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from server.environment import PrescriptionValidationEnvironment
from models import PrescriptionAction, PrescriptionObservation

# Create the FastAPI application
# - env: Factory function that returns a new Environment instance
# - action_cls: The Pydantic Action model
# - observation_cls: The Pydantic Observation model
app = create_fastapi_app(
    env=PrescriptionValidationEnvironment,
    action_cls=PrescriptionAction,
    observation_cls=PrescriptionObservation,
)

from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Interface Building...</h1>"
