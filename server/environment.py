"""Core environment logic for prescription validation.

Validates prescriptions against a drug database and grades agent actions
using deterministic clinical rules.
"""

import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PrescriptionAction, PrescriptionObservation, PrescriptionState
from server.drug_database import DRUG_DB, PrescriptionGenerator


class PrescriptionValidationEnvironment(Environment):
    """Medical prescription validation environment.

    Grades agent actions against ground-truth clinical issues embedded
    in each generated prescription case. Supports easy, medium, and hard
    difficulty levels with deterministic reward shaping.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 20

    def __init__(self):
        self._state = PrescriptionState()
        self._prescription_gen = PrescriptionGenerator(DRUG_DB)

        self._current_prescription: Dict = {}
        self._current_patient: Dict = {}
        self._ground_truth_issues: List[Dict] = []
        self._identified_issues: List[Dict] = []
        self._task_id = "easy"

        self._issues_correctly_found = set()
        self._false_positives = []
        self._episode_complete = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs,
    ) -> PrescriptionObservation:
        if seed is not None:
            random.seed(seed)

        self._task_id = task_id
        self._generate_case(task_id)

        self._identified_issues = []
        self._issues_correctly_found = set()
        self._false_positives = []
        self._episode_complete = False

        critical_count = sum(
            1 for issue in self._ground_truth_issues if issue.get("severity") == "critical"
        )

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
            safety_score=0.0,
        )

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
                "reject",
            ],
        )

    def step(
        self, action: PrescriptionAction, timeout_s: Optional[float] = None, **kwargs
    ) -> PrescriptionObservation:
        self._state.step_count += 1

        reward, feedback, done = self._process_action(action)

        if done:
            self._episode_complete = True
            self._state.prescription_status = (
                "approved" if action.action_type == "approve" else "rejected"
            )
            self._state.safety_score = self._calculate_safety_score()

        return PrescriptionObservation(
            done=done,
            reward=reward,
            prescription=self._current_prescription,
            patient_info=self._current_patient,
            validation_results=self._build_validation_results(),
            current_issues=self._identified_issues,
            feedback=feedback,
            task_id=self._task_id,
            step_count=self._state.step_count,
            available_actions=self._get_available_actions(),
        )

    @property
    def state(self) -> PrescriptionState:
        return self._state

    # ------------------------------------------------------------------
    # Case generation
    # ------------------------------------------------------------------

    def _generate_case(self, task_id: str):
        if task_id == "easy":
            case_type = random.choice(["safe", "interaction", "dosage", "allergy"])
            if case_type == "safe":
                prescription, patient = self._prescription_gen.generate_safe_prescription()
                issues = []
            elif case_type == "interaction":
                prescription, patient, issues = self._prescription_gen.generate_interaction_case()
            elif case_type == "dosage":
                prescription, patient, issues = self._prescription_gen.generate_dosage_error()
            else:
                prescription, patient, issues = (
                    self._prescription_gen.generate_contraindication_case()
                )

        elif task_id == "medium":
            case_type = random.choice(["interaction", "dosage", "contraindication"])
            if case_type == "interaction":
                prescription, patient, issues = self._prescription_gen.generate_interaction_case()
            elif case_type == "dosage":
                prescription, patient, issues = self._prescription_gen.generate_dosage_error()
            else:
                prescription, patient, issues = (
                    self._prescription_gen.generate_contraindication_case()
                )

        else:  # hard
            prescription, patient, issues = self._prescription_gen.generate_complex_case()

        self._current_prescription = prescription
        self._current_patient = patient
        self._ground_truth_issues = issues

    # ------------------------------------------------------------------
    # Action processing and reward calculation
    # ------------------------------------------------------------------

    def _process_action(self, action: PrescriptionAction) -> Tuple[float, str, bool]:
        reward = 0.0
        feedback = ""
        done = False

        if action.action_type == "approve":
            done = True
            if len(self._ground_truth_issues) == 0:
                reward = 1.0
                feedback = "Correct! This prescription is safe to dispense."
            else:
                missed_critical = sum(
                    1
                    for issue in self._ground_truth_issues
                    if issue.get("severity") == "critical" and issue not in self._identified_issues
                )
                reward = -1.0 * missed_critical
                self._state.false_negatives = len(self._ground_truth_issues) - len(
                    self._issues_correctly_found
                )
                feedback = (
                    f"UNSAFE! You approved a prescription with {len(self._ground_truth_issues)} issue(s). "
                    f"Missed critical issues: {missed_critical}. "
                    "Issues: "
                    + "; ".join(
                        f"{issue['type']}: {issue['description']}"
                        for issue in self._ground_truth_issues
                    )
                )

        elif action.action_type == "reject":
            done = True
            if len(self._ground_truth_issues) > 0:
                reward = 0.5 + (
                    0.5
                    * (len(self._issues_correctly_found) / max(1, len(self._ground_truth_issues)))
                )
                feedback = f"Correct rejection. Found {len(self._issues_correctly_found)}/{len(self._ground_truth_issues)} issues."
            else:
                reward = -0.5
                self._state.false_positives += 1
                feedback = "This prescription was actually safe. Unnecessary rejection delays patient care."

        elif action.action_type == "flag_interaction":
            reward, feedback = self._check_flagged_issue(action, expected_type="drug_interaction")

        elif action.action_type == "flag_dosage":
            reward, feedback = self._check_flagged_issue(
                action, expected_type=["dosage_too_high", "dosage_too_low"]
            )

        elif action.action_type == "flag_contraindication":
            reward, feedback = self._check_flagged_issue(action, expected_type="contraindication")

        elif action.action_type == "flag_allergy":
            reward, feedback = self._check_flagged_issue(action, expected_type="allergy_risk")

        elif action.action_type == "request_clarification":
            reward = 0.0
            feedback = (
                "Clarification requested. In a real system, this would contact the prescriber."
            )

        if self._state.step_count >= self.MAX_STEPS:
            done = True
            if not self._episode_complete:
                reward -= 0.3
                feedback += " [Episode ended: maximum steps reached]"

        return reward, feedback, done

    def _check_flagged_issue(
        self, action: PrescriptionAction, expected_type: Any
    ) -> Tuple[float, str]:
        expected_types = [expected_type] if isinstance(expected_type, str) else expected_type

        matching_issue = None
        for issue in self._ground_truth_issues:
            if issue["type"] in expected_types:
                if action.drug_name:
                    drug_match = (
                        issue.get("drug", "").lower() == action.drug_name.lower()
                        or issue.get("drug1", "").lower() == action.drug_name.lower()
                        or issue.get("drug2", "").lower() == action.drug_name.lower()
                    )
                    if drug_match:
                        matching_issue = issue
                        break
                else:
                    matching_issue = issue
                    break

        if matching_issue:
            issue_id = str(matching_issue)
            if issue_id not in self._issues_correctly_found:
                self._issues_correctly_found.add(issue_id)
                self._state.issues_found += 1

                self._identified_issues.append(
                    {
                        "drug": action.drug_name or matching_issue.get("drug", ""),
                        "issue": matching_issue["type"],
                        "severity": matching_issue.get("severity", "warning"),
                        "description": matching_issue["description"],
                    }
                )

                if matching_issue.get("severity") == "critical":
                    reward = 1.0
                    self._state.critical_issues_found += 1
                    feedback = f"CRITICAL ISSUE FOUND: {matching_issue['description']}"
                else:
                    reward = 0.5
                    feedback = f"Issue identified: {matching_issue['description']}"

                if action.recommendation and len(action.recommendation) > 10:
                    reward += 0.1
                    feedback += " [+0.1 for detailed recommendation]"
            else:
                reward = 0.0
                feedback = "This issue was already identified."
        else:
            self._state.false_positives += 1
            self._false_positives.append(
                {
                    "action": action.action_type,
                    "drug": action.drug_name,
                    "claimed_issue": action.issue_type,
                }
            )
            reward = -0.2
            feedback = f"False positive: No {expected_types[0]} issue with {action.drug_name or 'these drugs'}."

        return reward, feedback

    # ------------------------------------------------------------------
    # Validation and scoring helpers
    # ------------------------------------------------------------------

    def _build_validation_results(self) -> List[Dict]:
        results = []

        for med in self._current_prescription.get("medications", []):
            drug = med.get("drug", "")

            dosage_result = DRUG_DB.check_dosage(drug, med.get("dosage_mg", 0))
            results.append(
                {
                    "check": "dosage",
                    "drug": drug,
                    "status": "pass" if not dosage_result else "fail",
                    "message": dosage_result or "Dosage within safe range",
                }
            )

            allergy_result = DRUG_DB.check_allergy(drug, self._current_patient.get("allergies", []))
            results.append(
                {
                    "check": "allergy",
                    "drug": drug,
                    "status": "pass" if not allergy_result else "fail",
                    "message": allergy_result or "No allergy concerns",
                }
            )

            contra_result = DRUG_DB.check_contraindication(
                drug, self._current_patient.get("conditions", [])
            )
            results.append(
                {
                    "check": "contraindication",
                    "drug": drug,
                    "status": "pass" if not contra_result else "fail",
                    "message": contra_result or "No contraindications",
                }
            )

        meds = self._current_prescription.get("medications", [])
        for i, med1 in enumerate(meds):
            for med2 in meds[i + 1 :]:
                drug1 = med1.get("drug", "")
                drug2 = med2.get("drug", "")
                interaction = DRUG_DB.check_interaction(drug1, drug2)
                if interaction:
                    severity, description = interaction
                    results.append(
                        {
                            "check": "interaction",
                            "drug": f"{drug1} + {drug2}",
                            "status": "fail",
                            "message": f"{severity.upper()}: {description}",
                        }
                    )

        return results

    def _get_available_actions(self) -> List[str]:
        return [
            "approve",
            "flag_interaction",
            "flag_dosage",
            "flag_contraindication",
            "flag_allergy",
            "reject",
        ]

    def _calculate_safety_score(self) -> float:
        score = 1.0

        total_issues = len(self._ground_truth_issues)
        found_issues = len(self._issues_correctly_found)

        if total_issues > 0:
            missed = total_issues - found_issues
            missed_critical = self._state.total_critical_issues - self._state.critical_issues_found
            score -= missed * 0.2
            score -= missed_critical * 0.3

        score -= self._state.false_positives * 0.1
        return max(0.0, min(1.0, score))


__all__ = ["PrescriptionValidationEnvironment"]
