import json
import os
import sys
import time
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PrescriptionAction, PrescriptionObservation
from server.environment import PrescriptionValidationEnvironment
from server.drug_database import DRUG_DB

app = create_fastapi_app(
    env=PrescriptionValidationEnvironment,
    action_cls=PrescriptionAction,
    observation_cls=PrescriptionObservation,
)


@app.post("/review")
async def review_prescription(prescription: dict):
    """Industry-level: Perform a one-shot comprehensive clinical review."""
    # We instantiate a temporary env to perform the check
    from server.environment import PrescriptionValidationEnvironment
    env = PrescriptionValidationEnvironment()
    
    # Simple logic: analyze all meds in the prescription
    # In a real system, this would be a complex multi-step pipeline
    patient = prescription.get("patient", {})
    medications = prescription.get("medications", [])
    
    issues = []
    # Use environment's logic to find issues
    env._current_patient = patient
    env._current_prescription = prescription
    
    # Use internal generator logic to check against DRUG_DB
    from server.drug_database import DRUG_DB
    
    # Logic to review medications...
    # This simulates a real production processing pipeline
    results = []
    for med in medications:
        drug_data = DRUG_DB.get(med.get("name"))
        if not drug_data:
            continue
            
        # Hard rules
        if med.get("dosage", 0) > drug_data.get("max_dosage", 0):
            results.append({
                "severity": "critical",
                "drug": med.get("name"),
                "issue": "Dosage exceeds clinical safety limit",
                "recommendation": f"Reduce to below {drug_data.get('max_dosage')}mg"
            })
            
    # Add LLM context if available
    llm_rationale = ""
    if env._llm_client:
        try:
            prompt = f"Review this prescription: {json.dumps(prescription)}. Focus on safety. Return a clinical summary."
            resp = env._llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            llm_rationale = resp.choices[0].message.content
        except:
            llm_rationale = "LLM Clinical Assist unavailable."

    return {
        "status": "success",
        "findings": results,
        "clinical_summary": llm_rationale,
        "source": "Hybrid clinical engine v1.0"
    }


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
