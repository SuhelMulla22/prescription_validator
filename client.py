"""
Medical Prescription Validation Environment - Client
================================================================================

This is what users import to interact with the environment.

Usage:
    from prescription_validator import PrescriptionValidationEnv, PrescriptionAction
    
    # Connect to remote environment
    with PrescriptionValidationEnv(base_url="https://...").sync() as env:
        result = env.reset(task_id="easy")
        
        action = PrescriptionAction(
            action_type="flag_interaction",
            drug_name="Warfarin",
            severity="critical",
            recommendation="Switch to alternative"
        )
        
        result = env.step(action)
        print(result.observation.feedback)

Author: Suhel Mulla
"""

from typing import Optional
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import (
    PrescriptionAction,
    PrescriptionObservation,
    PrescriptionState
)


class PrescriptionValidationEnv(EnvClient[PrescriptionAction, PrescriptionObservation, PrescriptionState]):
    """
    Client for the Medical Prescription Validation Environment.
    
    This handles all communication between your code and the environment server.
    You never need to touch this class - it just works!
    """
    
    def _step_payload(self, action: PrescriptionAction) -> dict:
        """
        Convert action object to JSON for network transfer.
        
        Args:
            action: The action to send
        
        Returns:
            Dictionary representation of the action
        """
        return {
            "action_type": action.action_type,
            "drug_name": action.drug_name,
            "issue_type": action.issue_type,
            "severity": action.severity,
            "recommendation": action.recommendation,
            "rationale": action.rationale,
        }
    
    def _parse_result(self, payload: dict) -> StepResult:
        """
        Convert JSON response from server into typed objects.
        
        Args:
            payload: Raw JSON from server
        
        Returns:
            Typed StepResult with observation
        """
        obs_data = payload.get("observation", {})
        
        observation = PrescriptionObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            prescription=obs_data.get("prescription", {}),
            patient_info=obs_data.get("patient_info", {}),
            validation_results=obs_data.get("validation_results", []),
            current_issues=obs_data.get("current_issues", []),
            feedback=obs_data.get("feedback", ""),
            task_id=obs_data.get("task_id", "easy"),
            step_count=obs_data.get("step_count", 0),
            available_actions=obs_data.get("available_actions", [])
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )
    
    def _parse_state(self, payload: dict) -> PrescriptionState:
        """
        Convert JSON state from server into typed object.
        
        Args:
            payload: Raw JSON from server
        
        Returns:
            Typed PrescriptionState
        """
        return PrescriptionState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            difficulty=payload.get("difficulty", "easy"),
            total_issues=payload.get("total_issues", 0),
            issues_found=payload.get("issues_found", 0),
            false_positives=payload.get("false_positives", 0),
            false_negatives=payload.get("false_negatives", 0),
            critical_issues_found=payload.get("critical_issues_found", 0),
            total_critical_issues=payload.get("total_critical_issues", 0),
            prescription_status=payload.get("prescription_status", "pending_review"),
            safety_score=payload.get("safety_score", 0.0)
        )


# ============================================================================
# EXPORT
# ============================================================================

__all__ = ["PrescriptionValidationEnv"]