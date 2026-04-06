"""
Medical Prescription Validation Environment - Type Definitions
================================================================================

This module defines the data contracts for the prescription validation environment.

Real-world application: Prevent medication errors in hospitals and pharmacies
Target users: Healthcare AI systems, electronic health records (EHR), pharmacy systems

Author: Suhel Mulla
Date: 2026
License: MIT
"""

from typing import List, Dict, Optional, Any, Literal
from pydantic import Field, field_validator
from openenv.core.env_server import Action, Observation, State


# ============================================================================
# ACTION DEFINITIONS - What the AI agent can do
# ============================================================================

class PrescriptionAction(Action):
    """
    Represents an action the AI agent can take when validating a prescription.
    
    The agent acts like a clinical pharmacist reviewing prescriptions for safety.
    
    Attributes:
        action_type: The type of action to perform
        drug_name: Which medication to check (if applicable)
        issue_type: Type of issue identified (if validating)
        severity: How severe is the issue (critical/warning/info)
        recommendation: What should be done about it
        rationale: Medical reasoning for the action
    
    Examples:
        # Approve a safe prescription
        PrescriptionAction(
            action_type="approve",
            recommendation="Prescription is safe to dispense"
        )
        
        # Flag a drug interaction
        PrescriptionAction(
            action_type="flag_interaction",
            drug_name="Warfarin",
            issue_type="drug_interaction",
            severity="critical",
            recommendation="Do not combine with Aspirin - bleeding risk",
            rationale="Both are anticoagulants; combined use increases hemorrhage risk"
        )
    """
    
    action_type: Literal[
        "approve",              # Prescription is safe
        "flag_interaction",     # Drug-drug interaction found
        "flag_dosage",          # Dosage out of safe range
        "flag_contraindication",# Drug contraindicated for patient
        "flag_allergy",         # Patient allergic to drug
        "request_clarification",# Need more information
        "reject"                # Prescription is unsafe
    ]
    
    drug_name: Optional[str] = Field(
        default=None,
        description="Name of the drug being flagged (if applicable)"
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
        description="How severe is the issue? critical=patient harm likely, warning=potential issue, info=best practice note"
    )
    
    recommendation: str = Field(
        description="What should be done? E.g., 'Reduce dosage to 50mg' or 'Switch to alternative drug'"
    )
    
    rationale: Optional[str] = Field(
        default=None,
        description="Medical reasoning for this action (helps with training)"
    )
    
    @field_validator('drug_name')
    @classmethod
    def validate_drug_name(cls, v):
        """Ensure drug names are capitalized consistently"""
        if v:
            return v.strip().title()
        return v


# ============================================================================
# OBSERVATION DEFINITIONS - What the AI agent sees
# ============================================================================

class PrescriptionObservation(Observation):
    """
    What the AI agent observes after taking an action.
    
    Inherited from Observation base class:
        - done (bool): Is the episode finished?
        - reward (Optional[float]): Score for the last action
    
    Additional fields specific to prescription validation:
        - prescription: The prescription being reviewed
        - patient_info: Relevant patient medical history
        - validation_results: What checks have been performed so far
        - current_issues: Issues identified in this episode
        - feedback: Human-readable feedback on last action
        - task_id: Which task is being attempted
        - step_count: How many actions taken so far
    """
    
    # Inherited: done: bool, reward: Optional[float]
    
    prescription: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
        The prescription to validate. Structure:
        {
            'patient_id': '12345',
            'prescriber': 'Dr. Smith',
            'medications': [
                {
                    'drug': 'Aspirin',
                    'dosage': '81mg',
                    'frequency': 'once daily',
                    'route': 'oral',
                    'duration': '30 days'
                },
                ...
            ]
        }
        """
    )
    
    patient_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Relevant patient information. Structure:
        {
            'age': 65,
            'weight_kg': 70,
            'conditions': ['Hypertension', 'Diabetes'],
            'allergies': ['Penicillin'],
            'current_medications': ['Metformin', 'Lisinopril'],
            'kidney_function': 'normal',  # or 'impaired'
            'liver_function': 'normal'     # or 'impaired'
        }
        """
    )
    
    validation_results: List[Dict] = Field(
        default_factory=list,
        description="Results of automated checks performed so far"
    )
    
    current_issues: List[Dict] = Field(
        default_factory=list,
        description="""
        Issues the agent has identified. Structure:
        [
            {
                'drug': 'Warfarin',
                'issue': 'drug_interaction',
                'severity': 'critical',
                'with_drug': 'Aspirin',
                'description': 'Increased bleeding risk'
            },
            ...
        ]
        """
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


# ============================================================================
# STATE DEFINITIONS - Episode metadata
# ============================================================================

class PrescriptionState(State):
    """
    Episode-level state tracking.
    
    Inherited from State base class:
        - episode_id (Optional[str]): Unique identifier for this episode
        - step_count (int): Number of steps taken
    
    Additional tracking for prescription validation:
        - task_id: Which task is being attempted
        - difficulty: easy, medium, or hard
        - total_issues: How many issues exist in the prescription
        - issues_found: How many the agent correctly identified
        - false_positives: Incorrect flags raised
        - false_negatives: Missed issues
        - critical_issues_found: Most important issues caught
        - prescription_status: Current status of the review
    """
    
    # Inherited: episode_id: Optional[str], step_count: int
    
    task_id: str = Field(
        default="easy",
        description="Task identifier"
    )
    
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Task difficulty level"
    )
    
    total_issues: int = Field(
        default=0,
        description="Ground truth: how many issues actually exist"
    )
    
    issues_found: int = Field(
        default=0,
        description="How many issues the agent correctly identified"
    )
    
    false_positives: int = Field(
        default=0,
        description="Incorrect issues flagged by agent"
    )
    
    false_negatives: int = Field(
        default=0,
        description="Real issues the agent missed"
    )
    
    critical_issues_found: int = Field(
        default=0,
        description="How many critical (life-threatening) issues were caught"
    )
    
    total_critical_issues: int = Field(
        default=0,
        description="How many critical issues exist"
    )
    
    prescription_status: Literal[
        "pending_review",
        "approved",
        "rejected",
        "needs_clarification"
    ] = Field(
        default="pending_review",
        description="Current status of the prescription"
    )
    
    safety_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall safety score (0.0 = dangerous, 1.0 = perfectly safe)"
    )


# ============================================================================
# HELPER TYPE DEFINITIONS
# ============================================================================

class DrugInfo:
    """
    Internal drug database information.
    Not exposed to the agent - used server-side for validation.
    """
    
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