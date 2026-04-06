"""
Medical Prescription Validation - Inference Script
================================================================================

This script is MANDATORY for hackathon evaluation.
It MUST follow the exact logging format: [START], [STEP], [END]

DO NOT modify the logging functions!
DO NOT change the environment variable names!

Author: Suhel Mulla
"""

import os
import sys
import asyncio
import json

from typing import List, Dict, Any
from openai import OpenAI

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# MANDATORY ENVIRONMENT VARIABLES
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
API_KEY = os.getenv("HF_TOKEN")  # USE HF TOKEN, NOT OPENAI KEY!
IMAGE_NAME = os.getenv("IMAGE_NAME", "suhellll/prescription-validator")

# Task configuration
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "prescription_validator"
MAX_STEPS = 50
MAX_TOTAL_REWARD = 5.0
SUCCESS_SCORE_THRESHOLD = 0.7

# LLM settings
TEMPERATURE = 0.3  # Lower = more deterministic (better for medical decisions)
MAX_TOKENS = 512


# ============================================================================
# MANDATORY LOGGING FUNCTIONS (DO NOT MODIFY!)
# ============================================================================

def log_start(task: str, env: str, model: str):
    """Log episode start - EXACT FORMAT REQUIRED"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    """Log each step - EXACT FORMAT REQUIRED"""
    # Escape special characters in action string
    action_str = str(action).replace('\n', ' ').replace('\r', '')
    print(f"[STEP] step={step} action={action_str} reward={reward} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    """Log episode end - EXACT FORMAT REQUIRED"""
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)


# ============================================================================
# SYSTEM PROMPT - Medical Expert Instructions
# ============================================================================

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_user_prompt(observation: Dict[str, Any], history: List[str]) -> str:
    """
    Build detailed prompt from observation.
    
    Args:
        observation: Current environment observation
        history: Previous actions and results
    
    Returns:
        Formatted prompt string
    """
    prescription = observation.get("prescription", {})
    patient = observation.get("patient_info", {})
    feedback = observation.get("feedback", "")
    validation_results = observation.get("validation_results", [])
    current_issues = observation.get("current_issues", [])
    
    # Format medications
    meds_text = []
    for med in prescription.get("medications", []):
        meds_text.append(
            f"  • {med.get('drug')}: {med.get('dosage')} {med.get('frequency')} "
            f"({med.get('route')}) for {med.get('duration')}"
        )
    meds_str = "\n".join(meds_text) if meds_text else "  None"
    
    # Format patient conditions
    conditions_str = ", ".join(patient.get("conditions", [])) or "None"
    allergies_str = ", ".join(patient.get("allergies", [])) or "None"
    current_meds_str = ", ".join(patient.get("current_medications", [])) or "None"
    
    # Format issues already found
    issues_str = "None yet"
    if current_issues:
        issues_list = [
            f"  • {issue.get('drug')}: {issue.get('issue')} ({issue.get('severity')})"
            for issue in current_issues
        ]
        issues_str = "\n".join(issues_list)
    
    # Recent history
    history_text = "\n".join(history[-5:]) if history else "None"
    
    prompt = f"""
═══════════════════════════════════════════════════════════════
PRESCRIPTION TO REVIEW
═══════════════════════════════════════════════════════════════

PATIENT INFORMATION:
  Age: {patient.get('age')} years
  Weight: {patient.get('weight_kg')} kg
  Medical Conditions: {conditions_str}
  Allergies: {allergies_str}
  Current Medications: {current_meds_str}
  Kidney Function: {patient.get('kidney_function', 'unknown')}
  Liver Function: {patient.get('liver_function', 'unknown')}

PRESCRIBED MEDICATIONS:
{meds_str}

ISSUES IDENTIFIED SO FAR:
{issues_str}

LAST FEEDBACK:
{feedback}

RECENT ACTIONS:
{history_text}

═══════════════════════════════════════════════════════════════

Carefully analyze this prescription for safety.
Check EACH medication against patient allergies, conditions, and other drugs.
Respond with JSON action only.
"""
    
    return prompt.strip()


def parse_llm_response(text: str) -> Dict[str, Any]:
    """
    Parse LLM response into action dictionary.
    
    Args:
        text: Raw LLM output
    
    Returns:
        Action dictionary
    """
    # Try to extract JSON from response
    try:
        # Look for JSON object in response
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            action_dict = json.loads(json_str)
            
            # Validate required field
            if "action_type" in action_dict:
                return action_dict
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}", flush=True)
    
    # Fallback: try to detect action type from text
    text_lower = text.lower()
    
    if "approve" in text_lower and "safe" in text_lower:
        return {
            "action_type": "approve",
            "recommendation": "Prescription appears safe based on available information"
        }
    elif "interaction" in text_lower:
        return {
            "action_type": "flag_interaction",
            "severity": "warning",
            "recommendation": "Potential drug interaction detected"
        }
    elif "dosage" in text_lower or "dose" in text_lower:
        return {
            "action_type": "flag_dosage",
            "severity": "warning",
            "recommendation": "Dosage may need adjustment"
        }
    elif "allergy" in text_lower or "allergic" in text_lower:
        return {
            "action_type": "flag_allergy",
            "severity": "critical",
            "recommendation": "Patient may be allergic to prescribed medication"
        }
    elif "contraindication" in text_lower:
        return {
            "action_type": "flag_contraindication",
            "severity": "warning",
            "recommendation": "Medication may be contraindicated"
        }
    
    # Final fallback: request clarification
    return {
        "action_type": "request_clarification",
        "recommendation": "Need more information to assess safety"
    }


def get_llm_action(
    client: OpenAI,
    observation: Dict[str, Any],
    history: List[str]
) -> Dict[str, Any]:
    """
    Get action from LLM.
    
    Args:
        client: OpenAI client
        observation: Current environment observation
        history: Action history
    
    Returns:
        Action dictionary
    """
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
        
        import time
        time.sleep(2)  # Prevent burning rate limit
        return action_dict
    
    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        action_dict = {
            "action_type": "request_clarification",
            "recommendation": f"Error getting LLM response: {e}"
        }
        import time
        time.sleep(2)  # Prevent burning rate limit on error
        return action_dict


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

async def main():
    """Run inference on the prescription validation environment"""
    
    # Import environment client
    from client import PrescriptionValidationEnv
    from models import PrescriptionAction
    
    # Initialize OpenAI client
    if not API_KEY:
        print("[ERROR] OPENROUTER_API_KEY environment variable not set!", flush=True)
        sys.exit(1)
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Tracking
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Connect to environment
        hf_url = f"https://{IMAGE_NAME.replace('/', '-')}.hf.space"
        print(f"[DEBUG] Connecting to HuggingFace Space: {hf_url}", flush=True)
        
        async with PrescriptionValidationEnv(base_url=hf_url) as env:
            print("[DEBUG] Connected to environment", flush=True)
            
            # Reset environment with specified task
            result = await env.reset(task_id=TASK_NAME)
            print(f"[DEBUG] Environment reset: {result.observation.feedback}", flush=True)
            
            # Run episode
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    print(f"[DEBUG] Episode finished at step {step}", flush=True)
                    break
                
                # Get observation as dict
                obs_dict = result.observation.model_dump()
                
                # Get action from LLM
                action_dict = get_llm_action(client, obs_dict, history)
                
                # Create typed action
                try:
                    action = PrescriptionAction(**action_dict)
                except Exception as e:
                    print(f"[DEBUG] Invalid action format: {e}, using fallback", flush=True)
                    action = PrescriptionAction(
                        action_type="request_clarification",
                        recommendation="Action parsing error"
                    )
                
                # Take step in environment
                result = await env.step(action)
                
                # Track reward
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                
                # Log step
                log_step(
                    step=step,
                    action=json.dumps(action_dict),
                    reward=reward,
                    done=result.done,
                    error=None
                )
                
                # Update history
                history.append(
                    f"Step {step}: {action_dict.get('action_type')} → "
                    f"reward={reward:.2f}, feedback: {result.observation.feedback[:100]}"
                )
                
                if result.done:
                    break
        
        # Calculate final score (0.0 to 1.0)
        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        
        success = score >= SUCCESS_SCORE_THRESHOLD
        
        print(f"[DEBUG] Final score: {score:.3f}, Success: {success}", flush=True)
    
    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())