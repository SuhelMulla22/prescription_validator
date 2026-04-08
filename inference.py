"""Inference script for the prescription validation environment.

Follows the mandatory [START], [STEP], [END] logging format
required by the OpenEnv hackathon grader.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Environment variables (strictly required by hackathon proxy spec)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
IMAGE_NAME = os.getenv("IMAGE_NAME", "suhellll/prescription-validator")

TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "prescription_validator"
MAX_STEPS = 50
MAX_TOTAL_REWARD = 5.0
SUCCESS_SCORE_THRESHOLD = 0.7

TEMPERATURE = 0.3
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by the grader, do not modify
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    action_str = str(action).replace("\n", " ").replace("\r", "")
    reward_fmt = f"{reward:.2f}"
    done_fmt = "true" if done else "false"
    error_fmt = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action_str} reward={reward_fmt} done={done_fmt} error={error_fmt}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list):
    success_fmt = "true" if success else "false"
    rewards_fmt = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_fmt} steps={steps} rewards={rewards_fmt}", flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert clinical pharmacist reviewing prescriptions for safety.

YOUR ROLE:
- Identify medication errors before they harm patients
- Check for drug interactions, dosage errors, contraindications, allergies
- Provide clear, actionable recommendations

AVAILABLE ACTIONS:
1. flag_interaction - Two drugs interact dangerously
   {
     "action_type": "flag_interaction",
     "drug_name": "Warfarin",
     "issue_type": "drug_interaction",
     "severity": "critical",
     "recommendation": "Do not combine with Aspirin - use alternative",
     "rationale": "Both anticoagulants - hemorrhage risk"
   }

2. flag_dosage - Dosage outside safe range
   {
     "action_type": "flag_dosage",
     "drug_name": "Metformin",
     "issue_type": "dosage_too_high",
     "severity": "warning",
     "recommendation": "Reduce to 2000mg maximum",
     "rationale": "Exceeds FDA maximum daily dose"
   }

3. flag_contraindication - Drug contraindicated for patient
   {
     "action_type": "flag_contraindication",
     "drug_name": "Lisinopril",
     "issue_type": "contraindication",
     "severity": "critical",
     "recommendation": "Switch to alternative antihypertensive",
     "rationale": "Contraindicated in pregnancy"
   }

4. flag_allergy - Patient allergic to drug
   {
     "action_type": "flag_allergy",
     "drug_name": "Amoxicillin",
     "issue_type": "allergy_risk",
     "severity": "critical",
     "recommendation": "Use non-penicillin antibiotic",
     "rationale": "Patient has documented penicillin allergy"
   }

5. approve - Prescription is safe
   {
     "action_type": "approve",
     "recommendation": "Prescription is safe to dispense"
   }

6. reject - Prescription is unsafe
   {
     "action_type": "reject",
     "recommendation": "Contact prescriber - multiple safety issues"
   }

SEVERITY LEVELS:
- critical: Life-threatening, patient harm likely
- warning: Potential issue, requires attention
- info: Best practice note

STRATEGY:
1. Review patient info: age, weight, conditions, allergies, current meds
2. Check each prescribed medication for:
   - Dosage within safe limits?
   - Any patient allergies?
   - Contraindicated for patient conditions?
3. Check interactions between prescribed drugs
4. Flag ALL issues you find with specific drugs
5. Only approve if NO issues found
6. Reject if critical issues found

CRITICAL RULES:
- ALWAYS check for allergies first
- NEVER approve if critical issues exist
- Be specific: state which drug has which issue
- Provide medical rationale for each flag

RESPONSE FORMAT:
Respond with ONLY valid JSON matching one of the action schemas above.
No explanations outside the JSON.
"""


# ---------------------------------------------------------------------------
# Prompt construction and response parsing
# ---------------------------------------------------------------------------


def build_user_prompt(observation: Dict[str, Any], history: List[str]) -> str:
    prescription = observation.get("prescription", {})
    patient = observation.get("patient_info", {})
    feedback = observation.get("feedback", "")
    current_issues = observation.get("current_issues", [])

    meds_text = []
    for med in prescription.get("medications", []):
        meds_text.append(
            f"  - {med.get('drug')}: {med.get('dosage')} {med.get('frequency')} "
            f"({med.get('route')}) for {med.get('duration')}"
        )
    meds_str = "\n".join(meds_text) if meds_text else "  None"

    conditions_str = ", ".join(patient.get("conditions", [])) or "None"
    allergies_str = ", ".join(patient.get("allergies", [])) or "None"
    current_meds_str = ", ".join(patient.get("current_medications", [])) or "None"

    issues_str = "None yet"
    if current_issues:
        issues_list = [
            f"  - {issue.get('drug')}: {issue.get('issue')} ({issue.get('severity')})"
            for issue in current_issues
        ]
        issues_str = "\n".join(issues_list)

    history_text = "\n".join(history[-5:]) if history else "None"

    return f"""PRESCRIPTION TO REVIEW

PATIENT INFORMATION:
  Age: {patient.get("age")} years
  Weight: {patient.get("weight_kg")} kg
  Medical Conditions: {conditions_str}
  Allergies: {allergies_str}
  Current Medications: {current_meds_str}
  Kidney Function: {patient.get("kidney_function", "unknown")}
  Liver Function: {patient.get("liver_function", "unknown")}

PRESCRIBED MEDICATIONS:
{meds_str}

ISSUES IDENTIFIED SO FAR:
{issues_str}

LAST FEEDBACK:
{feedback}

RECENT ACTIONS:
{history_text}

Carefully analyze this prescription for safety.
Check EACH medication against patient allergies, conditions, and other drugs.
Respond with JSON action only."""


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Extract a valid action dict from the LLM response text."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            action_dict = json.loads(text[start:end])
            if "action_type" in action_dict:
                return action_dict
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}", flush=True)

    # Fallback: detect action intent from free-text
    text_lower = text.lower()
    if "approve" in text_lower and "safe" in text_lower:
        return {
            "action_type": "approve",
            "recommendation": "Prescription appears safe based on available information",
        }
    if "interaction" in text_lower:
        return {
            "action_type": "flag_interaction",
            "severity": "warning",
            "recommendation": "Potential drug interaction detected",
        }
    if "dosage" in text_lower or "dose" in text_lower:
        return {
            "action_type": "flag_dosage",
            "severity": "warning",
            "recommendation": "Dosage may need adjustment",
        }
    if "allergy" in text_lower or "allergic" in text_lower:
        return {
            "action_type": "flag_allergy",
            "severity": "critical",
            "recommendation": "Patient may be allergic to prescribed medication",
        }
    if "contraindication" in text_lower:
        return {
            "action_type": "flag_contraindication",
            "severity": "warning",
            "recommendation": "Medication may be contraindicated",
        }

    return {
        "action_type": "request_clarification",
        "recommendation": "Need more information to assess safety",
    }


def get_llm_action(
    client: OpenAI, observation: Dict[str, Any], history: List[str]
) -> Dict[str, Any]:
    """Query the LLM for a clinical action given the current observation."""
    user_prompt = build_user_prompt(observation, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        action_dict = parse_llm_response(response_text)
        time.sleep(2)
        return action_dict

    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        time.sleep(2)
        return {
            "action_type": "request_clarification",
            "recommendation": f"Error getting LLM response: {e}",
        }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


async def main():
    from client import PrescriptionValidationEnv
    from models import PrescriptionAction

    if not API_BASE_URL or not API_KEY:
        raise ValueError(
            "API_BASE_URL and API_KEY must be set. The hackathon platform will inject these automatically."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        hf_url = f"https://{IMAGE_NAME.replace('/', '-')}.hf.space"
        print(f"[DEBUG] Connecting to HuggingFace Space: {hf_url}", flush=True)

        async with PrescriptionValidationEnv(base_url=hf_url) as env:
            print("[DEBUG] Connected to environment", flush=True)

            result = await env.reset(task_id=TASK_NAME)
            print(f"[DEBUG] Environment reset: {result.observation.feedback}", flush=True)

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_dict = result.observation.model_dump()
                action_dict = get_llm_action(client, obs_dict, history)

                try:
                    action = PrescriptionAction(**action_dict)
                except Exception as e:
                    print(f"[DEBUG] Invalid action format: {e}, using fallback", flush=True)
                    action = PrescriptionAction(
                        action_type="request_clarification", recommendation="Action parsing error"
                    )

                result = await env.step(action)

                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=json.dumps(action_dict),
                    reward=reward,
                    done=result.done,
                    error=None,
                )

                history.append(
                    f"Step {step}: {action_dict.get('action_type')} -> "
                    f"reward={reward:.2f}, feedback: {result.observation.feedback[:100]}"
                )

                if result.done:
                    break

        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        print(f"[DEBUG] Final score: {score:.3f}, Success: {success}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}", flush=True)
        import traceback

        traceback.print_exc()

    log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
