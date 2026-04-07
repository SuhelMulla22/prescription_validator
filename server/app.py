"""FastAPI server for the prescription validation environment."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app
from server.environment import PrescriptionValidationEnvironment
from models import PrescriptionAction, PrescriptionObservation

app = create_fastapi_app(
    env=PrescriptionValidationEnvironment,
    action_cls=PrescriptionAction,
    observation_cls=PrescriptionObservation,
)


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Interface Building...</h1>"

def main():
    import uvicorn
    # Configured for Hugging Face Spaces port, defaulting to 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
