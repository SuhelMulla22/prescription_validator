"""
Medical Prescription Validation Environment
============================================

A real-world reinforcement learning environment for training AI agents
to prevent medication errors in clinical settings.

Usage:
    from prescription_validator import PrescriptionValidationEnv, PrescriptionAction

    async with PrescriptionValidationEnv(base_url="https://...") as env:
        result = await env.reset(task_id="easy")
        action = PrescriptionAction(
            action_type="approve",
            recommendation="Prescription is safe to dispense"
        )
        result = await env.step(action)
"""

from .client import PrescriptionValidationEnv
from .models import (
    PrescriptionAction,
    PrescriptionObservation,
    PrescriptionState,
)

__all__ = [
    "PrescriptionAction",
    "PrescriptionObservation",
    "PrescriptionState",
    "PrescriptionValidationEnv",
]
