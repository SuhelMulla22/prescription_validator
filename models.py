"""Type definitions for the prescription validation environment."""

from typing import List, Dict, Optional, Any, Literal
from pydantic import Field, field_validator
from openenv.core.env_server import Action, Observation, State


class PrescriptionAction(Action):
    """An action taken by the agent when validating a prescription."""

    action_type: Literal[
        "approve",
        "flag_interaction",
        "flag_dosage",
        "flag_contraindication",
        "flag_allergy",
        "request_clarification",
        "reject"
    ]

    drug_name: Optional[str] = Field(
        default=None,
        description="Name of the drug being flagged"
    )

    issue_type: Optional[Literal[
        "drug_interaction",
        "dosage_too_high",
        "dosage_too_low",
        "contraindication",
        "allergy_risk",
        "duplicate_therapy",
        "missing_information"
    ]] = None

    severity: Optional[Literal["critical", "warning", "info"]] = Field(
        default=None,
        description="Severity: critical (patient harm likely), warning (potential issue), info (best practice)"
    )

    recommendation: str = Field(
        description="What should be done, e.g. 'Reduce dosage to 50mg' or 'Switch to alternative drug'"
    )

    rationale: Optional[str] = Field(
        default=None,
        description="Medical reasoning for this action"
    )

    @field_validator('drug_name')
    @classmethod
    def validate_drug_name(cls, v):
        if v:
            return v.strip().title()
        return v


class PrescriptionObservation(Observation):
    """What the agent observes after taking an action.

    Inherits `done` and `reward` from the Observation base class.
    """

    prescription: Dict[str, Any] = Field(
        default_factory=dict,
        description="The prescription to validate (patient_id, prescriber, medications list)"
    )

    patient_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patient demographics, conditions, allergies, organ function"
    )

    validation_results: List[Dict] = Field(
        default_factory=list,
        description="Results of automated safety checks performed so far"
    )

    current_issues: List[Dict] = Field(
        default_factory=list,
        description="Issues the agent has identified so far"
    )

    feedback: str = Field(
        default="",
        description="Human-readable feedback on the last action taken"
    )

    task_id: str = Field(
        default="easy",
        description="Current task difficulty: easy, medium, or hard"
    )

    step_count: int = Field(
        default=0,
        description="Number of actions taken in this episode"
    )

    available_actions: List[str] = Field(
        default_factory=lambda: [
            "approve",
            "flag_interaction",
            "flag_dosage",
            "flag_contraindication",
            "flag_allergy",
            "reject"
        ],
        description="Actions available to the agent"
    )


class PrescriptionState(State):
    """Episode-level state tracking.

    Inherits `episode_id` and `step_count` from the State base class.
    """

    task_id: str = Field(default="easy")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    total_issues: int = Field(default=0)
    issues_found: int = Field(default=0)
    false_positives: int = Field(default=0)
    false_negatives: int = Field(default=0)
    critical_issues_found: int = Field(default=0)
    total_critical_issues: int = Field(default=0)

    prescription_status: Literal[
        "pending_review",
        "approved",
        "rejected",
        "needs_clarification"
    ] = Field(default="pending_review")

    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DrugInfo:
    """Internal drug reference entry. Not exposed to the agent."""

    def __init__(
        self,
        name: str,
        max_daily_dose_mg: float,
        min_daily_dose_mg: float,
        contraindications: List[str],
        interactions: List[str],
        common_allergies: List[str],
        requires_kidney_adjustment: bool = False,
        requires_liver_adjustment: bool = False
    ):
        self.name = name
        self.max_daily_dose_mg = max_daily_dose_mg
        self.min_daily_dose_mg = min_daily_dose_mg
        self.contraindications = contraindications
        self.interactions = interactions
        self.common_allergies = common_allergies
        self.requires_kidney_adjustment = requires_kidney_adjustment
        self.requires_liver_adjustment = requires_liver_adjustment