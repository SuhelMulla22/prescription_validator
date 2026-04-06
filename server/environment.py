"""
Medical Prescription Validation Environment - Core Logic
================================================================================

This is the heart of the environment where all validation logic lives.

The environment simulates a clinical pharmacist reviewing prescriptions for:
- Drug-drug interactions
- Dosage errors
- Contraindications
- Allergy risks
- Duplicate therapies

Author: Suhel Mulla
Date: 2026
License: MIT
"""

import random
import uuid
from typing import Dict, List, Any, Optional, Tuple
from openenv.core.env_server import Environment

# Import our custom types
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    PrescriptionAction,
    PrescriptionObservation,
    PrescriptionState
)
from server.drug_database import DRUG_DB, PrescriptionGenerator


# ============================================================================
# MAIN ENVIRONMENT CLASS
# ============================================================================

class PrescriptionValidationEnvironment(Environment):
    """
    Medical Prescription Validation Environment
    
    Real-world application: Train AI to catch medication errors before they harm patients.
    
    Task Structure:
    - Easy: Single drug, one obvious issue (or safe prescription)
    - Medium: 2-3 drugs, multiple potential issues
    - Hard: 4+ drugs, complex interactions, patient comorbidities
    
    Episode Flow:
    1. Agent receives prescription + patient info
    2. Agent analyzes for safety issues
    3. Agent flags issues or approves prescription
    4. Environment calculates reward based on accuracy
    5. Episode ends when agent approves/rejects or reaches step limit
    
    Reward Structure:
    - Correctly identify critical issue: +1.0
    - Correctly identify warning: +0.5
    - Miss critical issue: -1.0
    - False positive: -0.2
    - Correct approval of safe prescription: +1.0
    - Incorrect rejection of safe prescription: -0.5
    """
    
    # Enable concurrent sessions (multiple users can connect)
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    # Maximum steps per episode
    MAX_STEPS = 20
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(self):
        """Initialize the environment"""
        self._state = PrescriptionState()
        self._prescription_gen = PrescriptionGenerator(DRUG_DB)
        
        # Current episode data
        self._current_prescription: Dict = {}
        self._current_patient: Dict = {}
        self._ground_truth_issues: List[Dict] = []
        self._identified_issues: List[Dict] = []
        self._task_id = "easy"
        
        # Tracking
        self._issues_correctly_found = set()
        self._false_positives = []
        self._episode_complete = False
    
    # ========================================================================
    # CORE OPENENV METHODS (REQUIRED)
    # ========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs
    ) -> PrescriptionObservation:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Unique identifier for this episode
            task_id: Which task to run ("easy", "medium", "hard")
            **kwargs: Additional arguments
        
        Returns:
            Initial observation
        """
        if seed is not None:
            random.seed(seed)
        
        # Store task
        self._task_id = task_id
        
        # Generate prescription case based on task difficulty
        self._generate_case(task_id)
        
        # Reset tracking
        self._identified_issues = []
        self._issues_correctly_found = set()
        self._false_positives = []
        self._episode_complete = False
        
        # Count issues by severity
        critical_count = sum(
            1 for issue in self._ground_truth_issues 
            if issue.get("severity") == "critical"
        )
        
        # Initialize state
        self._state = PrescriptionState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=task_id,
            total_issues=len(self._ground_truth_issues),
            issues_found=0,
            false_positives=0,
            false_negatives=0,
            critical_issues_found=0,
            total_critical_issues=critical_count,
            prescription_status="pending_review",
            safety_score=0.0
        )
        
        # Return initial observation
        return PrescriptionObservation(
            done=False,
            reward=None,
            prescription=self._current_prescription,
            patient_info=self._current_patient,
            validation_results=[],
            current_issues=[],
            feedback=f"New {task_id} task: Please review this prescription for safety.",
            task_id=task_id,
            step_count=0,
            available_actions=[
                "approve",
                "flag_interaction",
                "flag_dosage",
                "flag_contraindication",
                "flag_allergy",
                "reject"
            ]
        )
    
    def step(
        self,
        action: PrescriptionAction,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> PrescriptionObservation:
        """
        Process an action from the agent.
        
        Args:
            action: The action taken by the agent
            timeout_s: Maximum time allowed for this step
            **kwargs: Additional arguments
        
        Returns:
            Observation after taking the action
        """
        # Increment step counter
        self._state.step_count += 1
        
        # Process the action
        reward, feedback, done = self._process_action(action)
        
        # Update state
        if done:
            self._episode_complete = True
            self._state.prescription_status = (
                "approved" if action.action_type == "approve" else "rejected"
            )
            # Calculate final safety score
            self._state.safety_score = self._calculate_safety_score()
        
        # Build observation
        observation = PrescriptionObservation(
            done=done,
            reward=reward,
            prescription=self._current_prescription,
            patient_info=self._current_patient,
            validation_results=self._build_validation_results(),
            current_issues=self._identified_issues,
            feedback=feedback,
            task_id=self._task_id,
            step_count=self._state.step_count,
            available_actions=self._get_available_actions()
        )
        
        return observation
    
    @property
    def state(self) -> PrescriptionState:
        """
        Get current episode state.
        
        Returns:
            Current state object
        """
        return self._state
    
    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================
    
    def _generate_case(self, task_id: str):
        """
        Generate a prescription case based on task difficulty.
        
        Args:
            task_id: "easy", "medium", or "hard"
        """
        if task_id == "easy":
            # Easy: Simple cases with 0-1 issues
            case_type = random.choice([
                "safe",
                "interaction",
                "dosage",
                "allergy"
            ])
            
            if case_type == "safe":
                prescription, patient = self._prescription_gen.generate_safe_prescription()
                issues = []
            elif case_type == "interaction":
                prescription, patient, issues = self._prescription_gen.generate_interaction_case()
            elif case_type == "dosage":
                prescription, patient, issues = self._prescription_gen.generate_dosage_error()
            else:  # allergy
                prescription, patient, issues = self._prescription_gen.generate_contraindication_case()
        
        elif task_id == "medium":
            # Medium: 2-3 drugs, 1-2 issues
            case_type = random.choice([
                "interaction",
                "dosage",
                "contraindication"
            ])
            
            if case_type == "interaction":
                prescription, patient, issues = self._prescription_gen.generate_interaction_case()
            elif case_type == "dosage":
                prescription, patient, issues = self._prescription_gen.generate_dosage_error()
            else:
                prescription, patient, issues = self._prescription_gen.generate_contraindication_case()
        
        else:  # hard
            # Hard: Complex case with multiple issues
            prescription, patient, issues = self._prescription_gen.generate_complex_case()
        
        self._current_prescription = prescription
        self._current_patient = patient
        self._ground_truth_issues = issues
    
    def _process_action(
        self,
        action: PrescriptionAction
    ) -> Tuple[float, str, bool]:
        """
        Process an agent action and calculate reward.
        
        Args:
            action: The action taken
        
        Returns:
            Tuple of (reward, feedback_message, is_done)
        """
        reward = 0.0
        feedback = ""
        done = False
        
        # ====== APPROVAL ACTION ======
        if action.action_type == "approve":
            done = True
            if len(self._ground_truth_issues) == 0:
                # Correctly approved safe prescription
                reward = 1.0
                feedback = "✅ Correct! This prescription is safe to dispense."
            else:
                # Approved prescription with issues - DANGEROUS!
                missed_critical = sum(
                    1 for issue in self._ground_truth_issues
                    if issue.get("severity") == "critical" 
                    and issue not in self._identified_issues
                )
                reward = -1.0 * missed_critical
                self._state.false_negatives = len(self._ground_truth_issues) - len(self._issues_correctly_found)
                feedback = f"❌ UNSAFE! You approved a prescription with {len(self._ground_truth_issues)} issue(s). "
                feedback += f"Missed critical issues: {missed_critical}. "
                feedback += "Issues: " + "; ".join([
                    f"{issue['type']}: {issue['description']}"
                    for issue in self._ground_truth_issues
                ])
        
        # ====== REJECTION ACTION ======
        elif action.action_type == "reject":
            done = True
            if len(self._ground_truth_issues) > 0:
                # Correctly rejected unsafe prescription
                critical_found = sum(
                    1 for issue in self._ground_truth_issues
                    if issue.get("severity") == "critical"
                )
                reward = 0.5 + (0.5 * (len(self._issues_correctly_found) / max(1, len(self._ground_truth_issues))))
                feedback = f"✅ Correct rejection. Found {len(self._issues_correctly_found)}/{len(self._ground_truth_issues)} issues."
            else:
                # Incorrectly rejected safe prescription
                reward = -0.5
                self._state.false_positives += 1
                feedback = "❌ This prescription was actually safe. Unnecessary rejection delays patient care."
        
        # ====== FLAG INTERACTION ======
        elif action.action_type == "flag_interaction":
            reward, feedback = self._check_flagged_issue(
                action,
                expected_type="drug_interaction"
            )
        
        # ====== FLAG DOSAGE ======
        elif action.action_type == "flag_dosage":
            reward, feedback = self._check_flagged_issue(
                action,
                expected_type=["dosage_too_high", "dosage_too_low"]
            )
        
        # ====== FLAG CONTRAINDICATION ======
        elif action.action_type == "flag_contraindication":
            reward, feedback = self._check_flagged_issue(
                action,
                expected_type="contraindication"
            )
        
        # ====== FLAG ALLERGY ======
        elif action.action_type == "flag_allergy":
            reward, feedback = self._check_flagged_issue(
                action,
                expected_type="allergy_risk"
            )
        
        # ====== REQUEST CLARIFICATION ======
        elif action.action_type == "request_clarification":
            # Neutral action - doesn't help or hurt
            reward = 0.0
            feedback = "Clarification requested. In a real system, this would contact the prescriber."
        
        # Check if max steps reached
        if self._state.step_count >= self.MAX_STEPS:
            done = True
            if not self._episode_complete:
                reward -= 0.3
                feedback += " [Episode ended: maximum steps reached]"
        
        return reward, feedback, done
    
    def _check_flagged_issue(
        self,
        action: PrescriptionAction,
        expected_type: Any
    ) -> Tuple[float, str]:
        """
        Check if a flagged issue matches ground truth.
        
        Args:
            action: The flag action
            expected_type: Expected issue type(s)
        
        Returns:
            Tuple of (reward, feedback)
        """
        # Normalize expected_type to list
        if isinstance(expected_type, str):
            expected_types = [expected_type]
        else:
            expected_types = expected_type
        
        # Find matching ground truth issue
        matching_issue = None
        for issue in self._ground_truth_issues:
            if issue["type"] in expected_types:
                # Check if drug matches (if specified)
                if action.drug_name:
                    drug_match = (
                        issue.get("drug", "").lower() == action.drug_name.lower() or
                        issue.get("drug1", "").lower() == action.drug_name.lower() or
                        issue.get("drug2", "").lower() == action.drug_name.lower()
                    )
                    if drug_match:
                        matching_issue = issue
                        break
                else:
                    matching_issue = issue
                    break
        
        if matching_issue:
            # Correctly identified an issue!
            issue_id = str(matching_issue)
            if issue_id not in self._issues_correctly_found:
                self._issues_correctly_found.add(issue_id)
                self._state.issues_found += 1
                
                # Add to identified issues
                self._identified_issues.append({
                    "drug": action.drug_name or matching_issue.get("drug", ""),
                    "issue": matching_issue["type"],
                    "severity": matching_issue.get("severity", "warning"),
                    "description": matching_issue["description"]
                })
                
                # Reward based on severity
                if matching_issue.get("severity") == "critical":
                    reward = 1.0
                    self._state.critical_issues_found += 1
                    feedback = f"🎯 CRITICAL ISSUE FOUND: {matching_issue['description']}"
                else:
                    reward = 0.5
                    feedback = f"✓ Issue identified: {matching_issue['description']}"
                
                # Bonus for good recommendation
                if action.recommendation and len(action.recommendation) > 10:
                    reward += 0.1
                    feedback += " [+0.1 for detailed recommendation]"
            else:
                # Already flagged this issue
                reward = 0.0
                feedback = "This issue was already identified."
        else:
            # False positive - flagged non-existent issue
            self._state.false_positives += 1
            self._false_positives.append({
                "action": action.action_type,
                "drug": action.drug_name,
                "claimed_issue": action.issue_type
            })
            reward = -0.2
            feedback = f"❌ False positive: No {expected_types[0]} issue with {action.drug_name or 'these drugs'}."
        
        return reward, feedback
    
    def _build_validation_results(self) -> List[Dict]:
        """
        Build validation results summary.
        
        Returns:
            List of validation check results
        """
        results = []
        
        # Check each medication
        for med in self._current_prescription.get("medications", []):
            drug = med.get("drug", "")
            
            # Dosage check
            dosage_result = DRUG_DB.check_dosage(drug, med.get("dosage_mg", 0))
            results.append({
                "check": "dosage",
                "drug": drug,
                "status": "pass" if not dosage_result else "fail",
                "message": dosage_result or "Dosage within safe range"
            })
            
            # Allergy check
            allergy_result = DRUG_DB.check_allergy(
                drug,
                self._current_patient.get("allergies", [])
            )
            results.append({
                "check": "allergy",
                "drug": drug,
                "status": "pass" if not allergy_result else "fail",
                "message": allergy_result or "No allergy concerns"
            })
            
            # Contraindication check
            contra_result = DRUG_DB.check_contraindication(
                drug,
                self._current_patient.get("conditions", [])
            )
            results.append({
                "check": "contraindication",
                "drug": drug,
                "status": "pass" if not contra_result else "fail",
                "message": contra_result or "No contraindications"
            })
        
        # Interaction checks (between drugs in prescription)
        meds = self._current_prescription.get("medications", [])
        for i, med1 in enumerate(meds):
            for med2 in meds[i+1:]:
                drug1 = med1.get("drug", "")
                drug2 = med2.get("drug", "")
                interaction = DRUG_DB.check_interaction(drug1, drug2)
                
                if interaction:
                    severity, description = interaction
                    results.append({
                        "check": "interaction",
                        "drug": f"{drug1} + {drug2}",
                        "status": "fail",
                        "message": f"{severity.upper()}: {description}"
                    })
        
        return results
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        return [
            "approve",
            "flag_interaction",
            "flag_dosage",
            "flag_contraindication",
            "flag_allergy",
            "reject"
        ]
    
    def _calculate_safety_score(self) -> float:
        """
        Calculate final safety score (0.0 to 1.0).
        
        Formula:
        - Start at 1.0 (perfect)
        - Subtract for missed issues (especially critical)
        - Subtract for false positives
        - Cap at 0.0 minimum
        """
        score = 1.0
        
        total_issues = len(self._ground_truth_issues)
        found_issues = len(self._issues_correctly_found)
        
        if total_issues > 0:
            # Penalize missed issues
            missed = total_issues - found_issues
            missed_critical = self._state.total_critical_issues - self._state.critical_issues_found
            
            score -= (missed * 0.2)  # -0.2 per missed issue
            score -= (missed_critical * 0.3)  # Additional -0.3 per missed critical
        
        # Penalize false positives
        score -= (self._state.false_positives * 0.1)
        
        # Cap at 0.0 minimum
        return max(0.0, min(1.0, score))


# ============================================================================
# EXPORT
# ============================================================================

__all__ = ["PrescriptionValidationEnvironment"]