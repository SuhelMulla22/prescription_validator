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
IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME") or os.getenv(
    "IMAGE_NAME", "suhellll/prescription-validator"
)

TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "prescription_validator"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.7

TEMPERATURE = 0.2
MAX_TOKENS = 800


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


def log_end(success: bool, steps: int, score: float, rewards: list):
    success_fmt = "true" if success else "false"
    rewards_fmt = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_fmt} steps={steps} score={score:.2f} rewards={rewards_fmt}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt — comprehensive clinical pharmacist instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert clinical pharmacist AI reviewing prescriptions for safety.

YOUR GOAL: Find ALL medication safety issues in the prescription. Review EACH drug systematically.

PROCESS (do one action per response):
1. Check each drug for ALLERGY issues (compare drug to patient's allergy list)
2. Check each drug for DOSAGE issues (is dose within safe range?)
3. Check each drug for CONTRAINDICATION issues (any conflict with patient conditions?)
4. Check PAIRS of drugs for INTERACTION issues
5. After flagging ALL issues, REJECT the prescription
6. If you found NO issues after checking everything, APPROVE

AVAILABLE ACTIONS (respond with ONE JSON object per turn):

1. flag_allergy — Patient allergic to a drug
   {"action_type": "flag_allergy", "drug_name": "DrugName", "issue_type": "allergy_risk",
    "severity": "critical", "recommendation": "specific alternative", "rationale": "reason"}

2. flag_dosage — Dose outside safe range
   {"action_type": "flag_dosage", "drug_name": "DrugName", "issue_type": "dosage_too_high",
    "severity": "critical", "recommendation": "reduce to Xmg", "rationale": "exceeds max"}

3. flag_interaction — Two drugs interact dangerously
   {"action_type": "flag_interaction", "drug_name": "Drug1", "issue_type": "drug_interaction",
    "severity": "critical", "recommendation": "avoid combination", "rationale": "reason"}

4. flag_contraindication — Drug contraindicated for patient condition
   {"action_type": "flag_contraindication", "drug_name": "DrugName", "issue_type": "contraindication",
    "severity": "critical", "recommendation": "switch to X", "rationale": "reason"}

5. approve — Prescription is safe (ONLY if NO issues exist)
   {"action_type": "approve", "recommendation": "Prescription safe to dispense"}

6. reject — Prescription has issues (do this AFTER flagging all issues)
   {"action_type": "reject", "recommendation": "summary of all issues found"}

SEVERITY: "critical" for life-threatening, "warning" for needs-attention, "info" for best-practice

IMPORTANT RULES:
- Respond with ONLY a single JSON object, no other text
- Do ONE action per turn — flag one issue, then wait for feedback
- Check allergies FIRST (penicillin allergy means NO amoxicillin!)
- Common critical interactions: Warfarin+NSAIDs, Warfarin+Aspirin, Opioid+Benzodiazepine, Lithium+NSAIDs
- When the feedback says "All issues found", respond with reject
- NEVER approve if you identified any issues
- Include drug_name for all flag actions
"""


# ---------------------------------------------------------------------------
# Prompt construction and response parsing
# ---------------------------------------------------------------------------


def build_user_prompt(observation: Dict[str, Any], history: List[str], step_num: int) -> str:
    """Build a detailed, structured prompt from the observation."""
    prescription = observation.get("prescription", {})
    patient = observation.get("patient_info", {})
    feedback = observation.get("feedback", "")
    current_issues = observation.get("current_issues", [])
    validation_results = observation.get("validation_results", [])

    # Build medication details
    meds_text = []
    for med in prescription.get("medications", []):
        meds_text.append(
            f"  - {med.get('drug')}: {med.get('dosage')} {med.get('frequency')} "
            f"via {med.get('route')} for {med.get('duration')}"
        )
    meds_str = "\n".join(meds_text) if meds_text else "  None"

    conditions_str = ", ".join(patient.get("conditions", [])) or "None"
    allergies_str = ", ".join(patient.get("allergies", [])) or "None"
    current_meds_str = ", ".join(patient.get("current_medications", [])) or "None"

    # Already-identified issues
    issues_str = "None yet"
    if current_issues:
        issues_list = [
            f"  - {issue.get('drug')}: {issue.get('issue')} [{issue.get('severity')}]"
            for issue in current_issues
        ]
        issues_str = "\n".join(issues_list)

    # Automated validation hints (from the environment's built-in checks)
    hints_str = ""
    if validation_results:
        failed = [r for r in validation_results if r.get("status") == "fail"]
        if failed:
            hints_list = [f"  ⚠ {r.get('drug')}: {r.get('message')}" for r in failed]
            hints_str = "\nAUTOMATED CHECK RESULTS (FAILURES):\n" + "\n".join(hints_list)

    history_text = "\n".join(history[-8:]) if history else "None"

    return f"""STEP {step_num} — PRESCRIPTION REVIEW

PATIENT:
  Age: {patient.get("age")} years | Weight: {patient.get("weight_kg")} kg
  Conditions: {conditions_str}
  Allergies: {allergies_str}
  Current Medications: {current_meds_str}
  Kidney: {patient.get("kidney_function", "unknown")} | Liver: {patient.get("liver_function", "unknown")}

PRESCRIBED MEDICATIONS:
{meds_str}

ISSUES FOUND SO FAR:
{issues_str}
{hints_str}

ENVIRONMENT FEEDBACK:
{feedback}

PREVIOUS ACTIONS:
{history_text}

Analyze carefully. Respond with a SINGLE JSON action object.
If all issues have been found (see feedback above), respond with reject.
If this is a safe prescription with no issues, respond with approve."""


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Extract a valid action dict from the LLM response text."""
    # Try direct JSON parse first
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line (code fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        action_dict = json.loads(text)
        if isinstance(action_dict, dict) and "action_type" in action_dict:
            return action_dict
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from within text
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            candidate = text[start:end]
            action_dict = json.loads(candidate)
            if "action_type" in action_dict:
                return action_dict
    except (json.JSONDecodeError, Exception):
        pass

    # Last resort: detect intent from free-text
    text_lower = text.lower()

    if "all issues found" in text_lower or "reject" in text_lower:
        return {
            "action_type": "reject",
            "recommendation": "Multiple safety issues identified - prescription rejected",
        }
    if "approve" in text_lower and ("safe" in text_lower or "no issue" in text_lower):
        return {
            "action_type": "approve",
            "recommendation": "Prescription appears safe to dispense",
        }
    if "allergy" in text_lower or "allergic" in text_lower:
        return {
            "action_type": "flag_allergy",
            "issue_type": "allergy_risk",
            "severity": "critical",
            "recommendation": "Patient may have allergy to prescribed medication",
        }
    if "interaction" in text_lower:
        return {
            "action_type": "flag_interaction",
            "issue_type": "drug_interaction",
            "severity": "warning",
            "recommendation": "Potential drug interaction detected",
        }
    if "dosage" in text_lower or "dose" in text_lower:
        return {
            "action_type": "flag_dosage",
            "issue_type": "dosage_too_high",
            "severity": "warning",
            "recommendation": "Dosage may need adjustment",
        }
    if "contraindication" in text_lower or "contraindicated" in text_lower:
        return {
            "action_type": "flag_contraindication",
            "issue_type": "contraindication",
            "severity": "warning",
            "recommendation": "Medication may be contraindicated",
        }

    # Absolute fallback
    return {
        "action_type": "reject",
        "recommendation": "Unable to fully verify safety — rejecting as precaution",
    }


def get_llm_action(
    client: OpenAI,
    observation: Dict[str, Any],
    history: List[str],
    step_num: int,
    conversation: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Query the LLM for a clinical action given the current observation.

    Uses conversation history so the LLM remembers previous actions.
    """
    user_prompt = build_user_prompt(observation, history, step_num)

    # Build messages with conversation context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent conversation history (last 6 turns to stay in context window)
    for msg in conversation[-12:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM raw response: {response_text[:300]}", flush=True)

        action_dict = parse_llm_response(response_text)

        # Store assistant response in conversation
        conversation.append({"role": "user", "content": user_prompt})
        conversation.append({"role": "assistant", "content": response_text})

        # Rate limiting
        time.sleep(1)
        return action_dict

    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        # On LLM failure, return a safe fallback
        return {
            "action_type": "reject",
            "recommendation": f"LLM error ({type(e).__name__}) — rejecting as safety precaution",
        }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


async def run_task(client: OpenAI, task_id: str):
    """Run a single task through to completion or max steps."""
    history: List[str] = []
    rewards: List[float] = []
    conversation: List[Dict[str, str]] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Task-specific max reward for normalization
    task_max_rewards = {"easy": 1.5, "medium": 3.0, "hard": 5.0}
    max_task_reward = task_max_rewards.get(task_id, 3.0)

    try:
        from client import PrescriptionValidationEnv
        from models import PrescriptionAction

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        local_env_url = os.getenv("LOCAL_ENV_URL")  # e.g. http://localhost:7860
        print(f"[DEBUG] Initializing environment for task: {task_id}", flush=True)

        if local_env_url:
            # Direct connection to a running local/remote server
            print(f"[DEBUG] Using LOCAL_ENV_URL: {local_env_url}", flush=True)
            env = PrescriptionValidationEnv(base_url=local_env_url)
        else:
            try:
                env = await PrescriptionValidationEnv.from_docker_image(IMAGE_NAME)
            except Exception as e:
                print(
                    f"[DEBUG] Docker init failed: {e}. Falling back to HuggingFace space...",
                    flush=True,
                )
                hf_url = f"https://{IMAGE_NAME.replace('/', '-')}.hf.space"
                env = PrescriptionValidationEnv(base_url=hf_url)

        print(f"[DEBUG] Connected to environment for {task_id}", flush=True)

        result = await env.reset(task_id=task_id)
        print(f"[DEBUG] {task_id} reset OK. Feedback: {result.observation.feedback}", flush=True)

        # Print prescription details for debugging
        obs = result.observation
        print(f"[DEBUG] Patient allergies: {obs.patient_info.get('allergies', [])}", flush=True)
        print(f"[DEBUG] Patient conditions: {obs.patient_info.get('conditions', [])}", flush=True)
        meds = obs.prescription.get("medications", [])
        for m in meds:
            print(f"[DEBUG] Med: {m.get('drug')} {m.get('dosage')}", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = result.observation.model_dump()
            action_dict = get_llm_action(client, obs_dict, history, step, conversation)

            print(
                f"[DEBUG] Step {step}: LLM chose action_type={action_dict.get('action_type')}, "
                f"drug={action_dict.get('drug_name', 'N/A')}",
                flush=True,
            )

            try:
                action = PrescriptionAction(**action_dict)
            except Exception as e:
                print(f"[DEBUG] Invalid action format: {e}, using reject fallback", flush=True)
                action = PrescriptionAction(
                    action_type="reject",
                    recommendation="Action format error — rejecting as precaution",
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
                f"Step {step}: {action_dict.get('action_type')} "
                f"drug={action_dict.get('drug_name', 'N/A')} → "
                f"reward={reward:+.2f} | {result.observation.feedback[:120]}"
            )

            print(
                f"[DEBUG] Step {step} result: reward={reward:.2f}, done={result.done}, "
                f"feedback={result.observation.feedback[:150]}",
                flush=True,
            )

            if result.done:
                break

        total_reward = sum(rewards)
        score = total_reward / max_task_reward if max_task_reward > 0 else 0.0

        # Clamp score to strictly (0, 1) as required by hackathon validator
        score = min(max(score, 0.01), 0.99)

        success = score >= SUCCESS_SCORE_THRESHOLD
        print(
            f"[DEBUG] {task_id} complete: total_reward={total_reward:.2f}, "
            f"score={score:.3f}, success={success}",
            flush=True,
        )

    except Exception as e:
        print(f"[DEBUG] Error during {task_id} inference: {e}", flush=True)
        import traceback
        traceback.print_exc()
        score = 0.01
    finally:
        try:
            if "env" in locals():
                await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main():
    if not API_BASE_URL or not API_KEY:
        print(
            "[DEBUG] WARNING: API_BASE_URL or API_KEY not set. Using dummy values for test.",
            flush=True,
        )
        proxy_url = API_BASE_URL or "https://api.openai.com/v1"
        proxy_key = API_KEY or "dummy_key"
    else:
        proxy_url = API_BASE_URL
        proxy_key = API_KEY

    print(f"[DEBUG] Using API_BASE_URL: {proxy_url}", flush=True)
    print(f"[DEBUG] Using MODEL_NAME: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Using IMAGE_NAME: {IMAGE_NAME}", flush=True)

    client = OpenAI(base_url=proxy_url, api_key=proxy_key)

    # Quick LLM health check
    print("[DEBUG] === TESTING LLM CLIENT ===", flush=True)
    try:
        test_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Respond with just the word OK"}],
            max_tokens=5,
        )
        test_text = test_resp.choices[0].message.content
        print(f"[DEBUG] ✅ LLM TEST PASSED: '{test_text}'", flush=True)
    except Exception as e:
        print(f"[DEBUG] ❌ LLM TEST FAILED: {e}", flush=True)
        print("[DEBUG] Continuing anyway — tasks will fail gracefully", flush=True)

    # Run easy, medium, and hard tasks
    tasks = ["easy", "medium", "hard"]
    for task_id in tasks:
        print(f"\n[DEBUG] ========== STARTING TASK: {task_id} ==========", flush=True)
        try:
            await run_task(client, task_id)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Emit minimal valid logs so grader doesn't break
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.01, rewards=[])


if __name__ == "__main__":
    asyncio.run(main())
