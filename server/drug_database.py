"""Drug database and prescription case generator.

Contains drug reference data (dosing, interactions, contraindications)
and generators for test cases of varying difficulty. Data is simplified
from FDA drug interaction guidelines for educational use.
"""

import random
from typing import Dict, List, Optional, Tuple


class DrugDatabase:
    """In-memory drug reference database with interaction and safety checking."""

    def __init__(self):
        self.drugs = {
            # Cardiovascular
            "Warfarin": {
                "class": "anticoagulant",
                "max_daily_dose_mg": 10,
                "min_daily_dose_mg": 1,
                "contraindications": ["Active bleeding", "Pregnancy"],
                "interactions": ["Aspirin", "Clopidogrel", "Ibuprofen", "NSAIDs"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            "Aspirin": {
                "class": "antiplatelet",
                "max_daily_dose_mg": 325,
                "min_daily_dose_mg": 81,
                "contraindications": ["Active bleeding", "Peptic ulcer"],
                "interactions": ["Warfarin", "Clopidogrel", "Ibuprofen"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": False,
            },
            "Lisinopril": {
                "class": "ACE inhibitor",
                "max_daily_dose_mg": 40,
                "min_daily_dose_mg": 5,
                "contraindications": [
                    "Pregnancy",
                    "Bilateral renal artery stenosis",
                    "Angioedema history",
                ],
                "interactions": ["Spironolactone", "Potassium supplements"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            "Metoprolol": {
                "class": "beta blocker",
                "max_daily_dose_mg": 200,
                "min_daily_dose_mg": 25,
                "contraindications": ["Severe bradycardia", "Heart block", "Cardiogenic shock"],
                "interactions": ["Verapamil", "Diltiazem"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            "Amlodipine": {
                "class": "calcium channel blocker",
                "max_daily_dose_mg": 10,
                "min_daily_dose_mg": 2.5,
                "contraindications": ["Severe aortic stenosis", "Cardiogenic shock"],
                "interactions": ["Simvastatin"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            # Diabetes
            "Metformin": {
                "class": "antidiabetic",
                "max_daily_dose_mg": 2550,
                "min_daily_dose_mg": 500,
                "contraindications": ["Severe kidney disease", "Metabolic acidosis"],
                "interactions": ["Contrast dye", "Alcohol"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            "Insulin": {
                "class": "antidiabetic",
                "max_daily_dose_mg": 200,  # units, not mg
                "min_daily_dose_mg": 10,
                "contraindications": ["Hypoglycemia"],
                "interactions": ["Beta blockers", "Corticosteroids"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            "Glipizide": {
                "class": "sulfonylurea",
                "max_daily_dose_mg": 40,
                "min_daily_dose_mg": 5,
                "contraindications": ["Type 1 Diabetes", "Diabetic ketoacidosis"],
                "interactions": ["Fluconazole", "Sulfamethoxazole"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": True,
            },
            # Antibiotics
            "Amoxicillin": {
                "class": "antibiotic",
                "max_daily_dose_mg": 3000,
                "min_daily_dose_mg": 500,
                "contraindications": ["Penicillin allergy"],
                "interactions": ["Warfarin"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            "Ciprofloxacin": {
                "class": "antibiotic",
                "max_daily_dose_mg": 1500,
                "min_daily_dose_mg": 500,
                "contraindications": ["Myasthenia gravis", "QT prolongation"],
                "interactions": ["Warfarin", "Theophylline", "Antacids"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            # Pain
            "Ibuprofen": {
                "class": "NSAID",
                "max_daily_dose_mg": 2400,
                "min_daily_dose_mg": 400,
                "contraindications": ["Active bleeding", "Severe kidney disease"],
                "interactions": ["Warfarin", "Aspirin", "Lisinopril"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            "Morphine": {
                "class": "opioid",
                "max_daily_dose_mg": 200,
                "min_daily_dose_mg": 10,
                "contraindications": ["Respiratory depression", "Paralytic ileus"],
                "interactions": ["Benzodiazepines", "Alcohol"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": True,
            },
            "Acetaminophen": {
                "class": "analgesic",
                "max_daily_dose_mg": 3000,
                "min_daily_dose_mg": 500,
                "contraindications": ["Severe liver disease"],
                "interactions": ["Warfarin", "Alcohol"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            # Psychiatric
            "Sertraline": {
                "class": "SSRI antidepressant",
                "max_daily_dose_mg": 200,
                "min_daily_dose_mg": 50,
                "contraindications": ["MAOI use within 14 days"],
                "interactions": ["Warfarin", "Tramadol", "MAOIs"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            "Lorazepam": {
                "class": "benzodiazepine",
                "max_daily_dose_mg": 10,
                "min_daily_dose_mg": 0.5,
                "contraindications": [
                    "Acute narrow-angle glaucoma",
                    "Severe respiratory insufficiency",
                ],
                "interactions": ["Opioids", "Alcohol"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            "Lithium": {
                "class": "mood stabilizer",
                "max_daily_dose_mg": 1800,
                "min_daily_dose_mg": 300,
                "contraindications": ["Severe kidney disease", "Dehydration"],
                "interactions": ["Ibuprofen", "Lisinopril", "Diuretics"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
            # Other
            "Omeprazole": {
                "class": "proton pump inhibitor",
                "max_daily_dose_mg": 40,
                "min_daily_dose_mg": 20,
                "contraindications": [],
                "interactions": ["Clopidogrel", "Warfarin"],
                "requires_monitoring": False,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            "Atorvastatin": {
                "class": "statin",
                "max_daily_dose_mg": 80,
                "min_daily_dose_mg": 10,
                "contraindications": ["Pregnancy", "Active liver disease"],
                "interactions": ["Gemfibrozil", "Grapefruit juice"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            "Levothyroxine": {
                "class": "thyroid hormone",
                "max_daily_dose_mg": 0.3,
                "min_daily_dose_mg": 0.025,
                "contraindications": ["Acute MI", "Thyrotoxicosis"],
                "interactions": ["Iron supplements", "Calcium", "Antacids"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": False,
            },
            "Prednisone": {
                "class": "corticosteroid",
                "max_daily_dose_mg": 80,
                "min_daily_dose_mg": 5,
                "contraindications": ["Systemic fungal infections"],
                "interactions": ["Warfarin", "Insulin", "NSAIDs"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": False,
            },
            "Digoxin": {
                "class": "cardiac glycoside",
                "max_daily_dose_mg": 0.25,
                "min_daily_dose_mg": 0.0625,
                "contraindications": ["Ventricular fibrillation", "Hypokalemia"],
                "interactions": ["Amiodarone", "Verapamil", "Diuretics"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": True,
                "requires_liver_adjustment": False,
            },
        }

        self.interaction_severity = {
            ("Warfarin", "Aspirin"): "critical",
            ("Warfarin", "Ibuprofen"): "critical",
            ("Warfarin", "Clopidogrel"): "critical",
            ("Aspirin", "Ibuprofen"): "warning",
            ("Lisinopril", "Spironolactone"): "warning",
            ("Metoprolol", "Verapamil"): "critical",
            ("Morphine", "Lorazepam"): "critical",
            ("Sertraline", "Tramadol"): "warning",
            ("Lithium", "Ibuprofen"): "critical",
            ("Lithium", "Lisinopril"): "critical",
            ("Digoxin", "Amiodarone"): "critical",
            ("Digoxin", "Verapamil"): "critical",
        }

    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        return self.drugs.get(drug_name.title())

    def check_interaction(self, drug1: str, drug2: str) -> Optional[Tuple[str, str]]:
        """Return (severity, description) if an interaction exists, else None."""
        drug1, drug2 = drug1.title(), drug2.title()

        severity = self.interaction_severity.get((drug1, drug2)) or self.interaction_severity.get(
            (drug2, drug1)
        )
        if severity:
            return (severity, f"{drug1} and {drug2} interaction")

        drug1_info = self.get_drug_info(drug1)
        drug2_info = self.get_drug_info(drug2)

        if drug1_info and drug2 in drug1_info.get("interactions", []):
            return ("warning", f"{drug1} interacts with {drug2}")
        if drug2_info and drug1 in drug2_info.get("interactions", []):
            return ("warning", f"{drug2} interacts with {drug1}")

        return None

    def check_dosage(self, drug_name: str, dosage_mg: float) -> Optional[str]:
        """Return an error message if dosage is out of range, else None."""
        drug_info = self.get_drug_info(drug_name)
        if not drug_info:
            return f"Unknown drug: {drug_name}"

        max_dose = drug_info["max_daily_dose_mg"]
        min_dose = drug_info["min_daily_dose_mg"]

        if dosage_mg > max_dose:
            return f"Dosage too high: {dosage_mg}mg exceeds maximum {max_dose}mg"
        if dosage_mg < min_dose:
            return f"Dosage too low: {dosage_mg}mg below minimum {min_dose}mg"
        return None

    def check_contraindication(
        self, drug_name: str, patient_conditions: List[str]
    ) -> Optional[str]:
        """Return an error message if the drug is contraindicated, else None."""
        drug_info = self.get_drug_info(drug_name)
        if not drug_info:
            return None

        for condition in patient_conditions:
            for contra in drug_info.get("contraindications", []):
                if contra.lower() in condition.lower() or condition.lower() in contra.lower():
                    return f"{drug_name} contraindicated in {condition}"
        return None

    def check_allergy(self, drug_name: str, patient_allergies: List[str]) -> Optional[str]:
        """Return an error message if the patient is allergic, else None."""
        drug_name = drug_name.title()

        if drug_name in [a.title() for a in patient_allergies]:
            return f"Patient allergic to {drug_name}"

        # Cross-sensitivity: Penicillin allergy covers Amoxicillin
        if "Penicillin" in patient_allergies and drug_name == "Amoxicillin":
            return f"Patient has Penicillin allergy - {drug_name} contraindicated"

        return None


# -----------------------------------------------------------------------
# Random name / demographic generators for variety
# -----------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Nancy", "Maria", "Ahmed", "Fatima", "Wei", "Priya", "Carlos",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Patel", "Chen", "Kim", "Nguyen",
]

DOCTOR_FIRST = [
    "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Amanda",
    "James", "Rachel", "Andrew", "Lisa", "Christopher", "Angela", "Mark",
    "Stephanie", "Kevin", "Priya", "Wei", "Omar", "Fatima",
]

DOCTOR_LAST = [
    "Johnson", "Chen", "Rodriguez", "Kim", "Martinez", "Patel", "Wilson",
    "Thompson", "Garcia", "Adams", "Wright", "Hall", "Young", "Lopez",
    "Rivera", "Campbell", "Mitchell", "Roberts", "Carter", "Phillips",
]


def _random_patient_id() -> str:
    return f"P{random.randint(1000, 9999)}"


def _random_doctor() -> str:
    return f"Dr. {random.choice(DOCTOR_FIRST)} {random.choice(DOCTOR_LAST)}"


def _random_date() -> str:
    year = 2024
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


class PrescriptionGenerator:
    """Generates prescription test cases with known ground-truth issues."""

    def __init__(self, drug_db: DrugDatabase):
        self.drug_db = drug_db

    # ==================================================================
    #  EASY CASES — single medication, 0-1 obvious issues
    # ==================================================================

    def generate_safe_prescription(self) -> Tuple[Dict, Dict]:
        """Generate a genuinely safe prescription with no issues."""
        case = random.choice([
            self._safe_lisinopril,
            self._safe_metoprolol,
            self._safe_omeprazole,
            self._safe_atorvastatin,
            self._safe_metformin,
            self._safe_amlodipine,
        ])
        return case()

    def _safe_lisinopril(self) -> Tuple[Dict, Dict]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Lisinopril", "dosage": "10mg", "dosage_mg": 10,
                "frequency": "once daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(40, 70), "weight_kg": random.randint(60, 95),
            "conditions": ["Hypertension"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        return prescription, patient

    def _safe_metoprolol(self) -> Tuple[Dict, Dict]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Metoprolol", "dosage": "50mg", "dosage_mg": 50,
                "frequency": "twice daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(45, 75), "weight_kg": random.randint(65, 100),
            "conditions": ["Hypertension"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        return prescription, patient

    def _safe_omeprazole(self) -> Tuple[Dict, Dict]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Omeprazole", "dosage": "20mg", "dosage_mg": 20,
                "frequency": "once daily", "route": "oral", "duration": "14 days",
            }],
        }
        patient = {
            "age": random.randint(30, 65), "weight_kg": random.randint(55, 90),
            "conditions": ["GERD"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        return prescription, patient

    def _safe_atorvastatin(self) -> Tuple[Dict, Dict]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Atorvastatin", "dosage": "20mg", "dosage_mg": 20,
                "frequency": "once daily", "route": "oral", "duration": "90 days",
            }],
        }
        patient = {
            "age": random.randint(50, 75), "weight_kg": random.randint(65, 95),
            "conditions": ["Hyperlipidemia"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        return prescription, patient

    def _safe_metformin(self) -> Tuple[Dict, Dict]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Metformin", "dosage": "500mg", "dosage_mg": 500,
                "frequency": "twice daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(35, 70), "weight_kg": random.randint(70, 110),
            "conditions": ["Type 2 Diabetes"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        return prescription, patient

    def _safe_amlodipine(self) -> Tuple[Dict, Dict]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Amlodipine", "dosage": "5mg", "dosage_mg": 5,
                "frequency": "once daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(40, 70), "weight_kg": random.randint(55, 85),
            "conditions": ["Hypertension"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        return prescription, patient

    # ------------------------------------------------------------------
    #  EASY — Interaction cases
    # ------------------------------------------------------------------

    def generate_interaction_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        case = random.choice([
            self._interaction_warfarin_aspirin,
            self._interaction_warfarin_ibuprofen,
            self._interaction_morphine_lorazepam,
            self._interaction_lithium_ibuprofen,
        ])
        return case()

    def _interaction_warfarin_aspirin(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Warfarin", "dosage": "5mg", "dosage_mg": 5,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Aspirin", "dosage": "81mg", "dosage_mg": 81,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
            ],
        }
        patient = {
            "age": random.randint(55, 80), "weight_kg": random.randint(60, 95),
            "conditions": ["Atrial fibrillation", "Coronary artery disease"],
            "allergies": [], "current_medications": ["Warfarin"],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "drug_interaction", "severity": "critical",
            "drug1": "Warfarin", "drug2": "Aspirin",
            "description": "Both are anticoagulants - major bleeding risk",
            "recommendation": "Use only one anticoagulant or use under close monitoring",
        }]
        return prescription, patient, issues

    def _interaction_warfarin_ibuprofen(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Warfarin", "dosage": "5mg", "dosage_mg": 5,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Ibuprofen", "dosage": "400mg", "dosage_mg": 400,
                 "frequency": "three times daily", "route": "oral", "duration": "7 days"},
            ],
        }
        patient = {
            "age": random.randint(50, 75), "weight_kg": random.randint(55, 90),
            "conditions": ["Atrial fibrillation", "Osteoarthritis"],
            "allergies": [], "current_medications": ["Warfarin"],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "drug_interaction", "severity": "critical",
            "drug1": "Warfarin", "drug2": "Ibuprofen",
            "description": "NSAIDs increase bleeding risk with anticoagulants",
            "recommendation": "Use acetaminophen instead of ibuprofen for pain",
        }]
        return prescription, patient, issues

    def _interaction_morphine_lorazepam(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Morphine", "dosage": "15mg", "dosage_mg": 15,
                 "frequency": "every 4 hours", "route": "oral", "duration": "5 days"},
                {"drug": "Lorazepam", "dosage": "1mg", "dosage_mg": 1,
                 "frequency": "twice daily", "route": "oral", "duration": "5 days"},
            ],
        }
        patient = {
            "age": random.randint(40, 70), "weight_kg": random.randint(55, 85),
            "conditions": ["Chronic pain", "Anxiety"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "drug_interaction", "severity": "critical",
            "drug1": "Morphine", "drug2": "Lorazepam",
            "description": "Opioid + benzodiazepine combination - respiratory depression risk",
            "recommendation": "Avoid concurrent use; consider non-benzodiazepine anxiolytic",
        }]
        return prescription, patient, issues

    def _interaction_lithium_ibuprofen(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Lithium", "dosage": "600mg", "dosage_mg": 600,
                 "frequency": "twice daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Ibuprofen", "dosage": "400mg", "dosage_mg": 400,
                 "frequency": "three times daily", "route": "oral", "duration": "7 days"},
            ],
        }
        patient = {
            "age": random.randint(25, 55), "weight_kg": random.randint(55, 85),
            "conditions": ["Bipolar disorder", "Back pain"],
            "allergies": [], "current_medications": ["Lithium"],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "drug_interaction", "severity": "critical",
            "drug1": "Lithium", "drug2": "Ibuprofen",
            "description": "NSAIDs increase lithium levels - lithium toxicity risk",
            "recommendation": "Use acetaminophen instead; monitor lithium levels if NSAID necessary",
        }]
        return prescription, patient, issues

    # ------------------------------------------------------------------
    #  EASY — Dosage error cases
    # ------------------------------------------------------------------

    def generate_dosage_error(self) -> Tuple[Dict, Dict, List[Dict]]:
        case = random.choice([
            self._dosage_metformin_high,
            self._dosage_warfarin_high,
            self._dosage_lorazepam_high,
            self._dosage_digoxin_high,
        ])
        return case()

    def _dosage_metformin_high(self) -> Tuple[Dict, Dict, List[Dict]]:
        over_dose = random.choice([3000, 3500, 4000])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Metformin", "dosage": f"{over_dose}mg", "dosage_mg": over_dose,
                "frequency": "once daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(40, 70), "weight_kg": random.randint(70, 100),
            "conditions": ["Type 2 Diabetes"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "dosage_too_high", "severity": "warning",
            "drug": "Metformin",
            "description": f"{over_dose}mg exceeds maximum daily dose of 2550mg",
            "recommendation": "Reduce to maximum 2550mg daily or split into multiple doses",
        }]
        return prescription, patient, issues

    def _dosage_warfarin_high(self) -> Tuple[Dict, Dict, List[Dict]]:
        over_dose = random.choice([15, 20, 25])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Warfarin", "dosage": f"{over_dose}mg", "dosage_mg": over_dose,
                "frequency": "once daily", "route": "oral", "duration": "ongoing",
            }],
        }
        patient = {
            "age": random.randint(55, 80), "weight_kg": random.randint(60, 90),
            "conditions": ["Atrial fibrillation"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "dosage_too_high", "severity": "critical",
            "drug": "Warfarin",
            "description": f"{over_dose}mg exceeds maximum daily dose of 10mg - serious bleeding risk",
            "recommendation": "Reduce to maximum 10mg, typical dose 2-5mg",
        }]
        return prescription, patient, issues

    def _dosage_lorazepam_high(self) -> Tuple[Dict, Dict, List[Dict]]:
        over_dose = random.choice([12, 15, 20])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Lorazepam", "dosage": f"{over_dose}mg", "dosage_mg": over_dose,
                "frequency": "daily", "route": "oral", "duration": "14 days",
            }],
        }
        patient = {
            "age": random.randint(30, 65), "weight_kg": random.randint(55, 85),
            "conditions": ["Anxiety disorder"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "dosage_too_high", "severity": "critical",
            "drug": "Lorazepam",
            "description": f"{over_dose}mg exceeds maximum daily dose of 10mg - oversedation risk",
            "recommendation": "Reduce to maximum 10mg daily; typical anxiolytic dose 1-4mg",
        }]
        return prescription, patient, issues

    def _dosage_digoxin_high(self) -> Tuple[Dict, Dict, List[Dict]]:
        over_dose = random.choice([0.5, 0.75, 1.0])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Digoxin", "dosage": f"{over_dose}mg", "dosage_mg": over_dose,
                "frequency": "once daily", "route": "oral", "duration": "ongoing",
            }],
        }
        patient = {
            "age": random.randint(60, 85), "weight_kg": random.randint(50, 80),
            "conditions": ["Heart failure"],
            "allergies": [], "current_medications": [],
            "kidney_function": random.choice(["normal", "mildly impaired"]),
            "liver_function": "normal",
        }
        issues = [{
            "type": "dosage_too_high", "severity": "critical",
            "drug": "Digoxin",
            "description": f"{over_dose}mg exceeds maximum daily dose of 0.25mg - digoxin toxicity risk",
            "recommendation": "Reduce to 0.125-0.25mg daily; monitor kidney function and levels",
        }]
        return prescription, patient, issues

    # ------------------------------------------------------------------
    #  EASY — Allergy / contraindication cases
    # ------------------------------------------------------------------

    def generate_contraindication_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        case = random.choice([
            self._allergy_penicillin,
            self._contra_lisinopril_pregnancy,
            self._contra_metformin_kidney,
            self._contra_atorvastatin_pregnancy,
        ])
        return case()

    def _allergy_penicillin(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Amoxicillin", "dosage": "500mg", "dosage_mg": 500,
                "frequency": "three times daily", "route": "oral", "duration": "10 days",
            }],
        }
        patient = {
            "age": random.randint(20, 60), "weight_kg": random.randint(50, 90),
            "conditions": ["Pneumonia"],
            "allergies": ["Penicillin"],
            "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "allergy_risk", "severity": "critical",
            "drug": "Amoxicillin",
            "description": "Patient has Penicillin allergy - Amoxicillin is a penicillin derivative",
            "recommendation": "Switch to non-penicillin antibiotic (e.g., Azithromycin, Ciprofloxacin)",
        }]
        return prescription, patient, issues

    def _contra_lisinopril_pregnancy(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Lisinopril", "dosage": "10mg", "dosage_mg": 10,
                "frequency": "once daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(22, 38), "weight_kg": random.randint(55, 80),
            "conditions": ["Hypertension", "Pregnancy"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "contraindication", "severity": "critical",
            "drug": "Lisinopril",
            "description": "ACE inhibitors are contraindicated in pregnancy - teratogenic effects",
            "recommendation": "Switch to pregnancy-safe antihypertensive (e.g., Methyldopa, Labetalol)",
        }]
        return prescription, patient, issues

    def _contra_metformin_kidney(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Metformin", "dosage": "1000mg", "dosage_mg": 1000,
                "frequency": "twice daily", "route": "oral", "duration": "30 days",
            }],
        }
        patient = {
            "age": random.randint(55, 80), "weight_kg": random.randint(60, 95),
            "conditions": ["Type 2 Diabetes", "Severe kidney disease"],
            "allergies": [], "current_medications": [],
            "kidney_function": "severely impaired", "liver_function": "normal",
        }
        issues = [{
            "type": "contraindication", "severity": "critical",
            "drug": "Metformin",
            "description": "Metformin is contraindicated in severe kidney disease - lactic acidosis risk",
            "recommendation": "Switch to renal-safe antidiabetic (e.g., insulin, DPP-4 inhibitor)",
        }]
        return prescription, patient, issues

    def _contra_atorvastatin_pregnancy(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [{
                "drug": "Atorvastatin", "dosage": "20mg", "dosage_mg": 20,
                "frequency": "once daily", "route": "oral", "duration": "90 days",
            }],
        }
        patient = {
            "age": random.randint(25, 40), "weight_kg": random.randint(55, 80),
            "conditions": ["Hyperlipidemia", "Pregnancy"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [{
            "type": "contraindication", "severity": "critical",
            "drug": "Atorvastatin",
            "description": "Statins are contraindicated in pregnancy - risk of fetal harm",
            "recommendation": "Discontinue statin during pregnancy; manage with diet modifications",
        }]
        return prescription, patient, issues

    # ==================================================================
    #  MEDIUM CASES — 2-3 medications, multiple interaction/issue types
    # ==================================================================

    def generate_medium_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        case = random.choice([
            self._medium_warfarin_aspirin_omeprazole,
            self._medium_morphine_lorazepam_high_dose,
            self._medium_lisinopril_pregnancy_metformin,
            self._medium_multi_drug_interaction,
            self._medium_dosage_plus_allergy,
        ])
        return case()

    def _medium_warfarin_aspirin_omeprazole(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Warfarin", "dosage": "5mg", "dosage_mg": 5,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Aspirin", "dosage": "325mg", "dosage_mg": 325,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Omeprazole", "dosage": "20mg", "dosage_mg": 20,
                 "frequency": "once daily", "route": "oral", "duration": "30 days"},
            ],
        }
        patient = {
            "age": random.randint(55, 80), "weight_kg": random.randint(60, 95),
            "conditions": ["Atrial fibrillation", "GERD", "History of stroke"],
            "allergies": [], "current_medications": ["Warfarin"],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Warfarin", "drug2": "Aspirin",
                "description": "Dual anticoagulant/antiplatelet - major bleeding risk",
                "recommendation": "Avoid concurrent use unless cardiologist directs",
            },
            {
                "type": "drug_interaction", "severity": "warning",
                "drug1": "Omeprazole", "drug2": "Warfarin",
                "description": "Omeprazole may increase warfarin effect (CYP2C19 interaction)",
                "recommendation": "Monitor INR closely; consider pantoprazole instead",
            },
        ]
        return prescription, patient, issues

    def _medium_morphine_lorazepam_high_dose(self) -> Tuple[Dict, Dict, List[Dict]]:
        morphine_dose = random.choice([250, 300])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Morphine", "dosage": f"{morphine_dose}mg", "dosage_mg": morphine_dose,
                 "frequency": "daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Lorazepam", "dosage": "2mg", "dosage_mg": 2,
                 "frequency": "twice daily", "route": "oral", "duration": "14 days"},
            ],
        }
        patient = {
            "age": random.randint(45, 75), "weight_kg": random.randint(55, 85),
            "conditions": ["Chronic pain", "Anxiety disorder"],
            "allergies": [], "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [
            {
                "type": "dosage_too_high", "severity": "critical",
                "drug": "Morphine",
                "description": f"{morphine_dose}mg exceeds maximum daily dose of 200mg",
                "recommendation": "Reduce morphine dose; consider multimodal pain management",
            },
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Morphine", "drug2": "Lorazepam",
                "description": "Opioid + benzodiazepine - respiratory depression risk",
                "recommendation": "Avoid concurrent use per FDA black box warning",
            },
        ]
        return prescription, patient, issues

    def _medium_lisinopril_pregnancy_metformin(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Lisinopril", "dosage": "20mg", "dosage_mg": 20,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Metformin", "dosage": "1000mg", "dosage_mg": 1000,
                 "frequency": "twice daily", "route": "oral", "duration": "ongoing"},
            ],
        }
        patient = {
            "age": random.randint(25, 38), "weight_kg": random.randint(60, 85),
            "conditions": ["Hypertension", "Type 2 Diabetes", "Pregnancy"],
            "allergies": [], "current_medications": ["Lisinopril", "Metformin"],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [
            {
                "type": "contraindication", "severity": "critical",
                "drug": "Lisinopril",
                "description": "ACE inhibitors contraindicated in pregnancy - fetal harm risk",
                "recommendation": "Switch to Labetalol or Methyldopa for blood pressure management",
            },
        ]
        return prescription, patient, issues

    def _medium_multi_drug_interaction(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Lithium", "dosage": "600mg", "dosage_mg": 600,
                 "frequency": "twice daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Lisinopril", "dosage": "10mg", "dosage_mg": 10,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Ibuprofen", "dosage": "600mg", "dosage_mg": 600,
                 "frequency": "three times daily", "route": "oral", "duration": "10 days"},
            ],
        }
        patient = {
            "age": random.randint(30, 55), "weight_kg": random.randint(55, 85),
            "conditions": ["Bipolar disorder", "Hypertension", "Knee pain"],
            "allergies": [], "current_medications": ["Lithium", "Lisinopril"],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Lithium", "drug2": "Ibuprofen",
                "description": "NSAIDs increase lithium levels - toxicity risk",
                "recommendation": "Use acetaminophen; if NSAID necessary, monitor lithium levels",
            },
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Lithium", "drug2": "Lisinopril",
                "description": "ACE inhibitors increase lithium levels - toxicity risk",
                "recommendation": "Monitor lithium levels closely; consider ARB or CCB for BP",
            },
        ]
        return prescription, patient, issues

    def _medium_dosage_plus_allergy(self) -> Tuple[Dict, Dict, List[Dict]]:
        cipro_dose = random.choice([2000, 2500])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Amoxicillin", "dosage": "500mg", "dosage_mg": 500,
                 "frequency": "three times daily", "route": "oral", "duration": "7 days"},
                {"drug": "Ciprofloxacin", "dosage": f"{cipro_dose}mg", "dosage_mg": cipro_dose,
                 "frequency": "daily", "route": "oral", "duration": "7 days"},
            ],
        }
        patient = {
            "age": random.randint(30, 60), "weight_kg": random.randint(55, 85),
            "conditions": ["Urinary tract infection", "Bronchitis"],
            "allergies": ["Penicillin"],
            "current_medications": [],
            "kidney_function": "normal", "liver_function": "normal",
        }
        issues = [
            {
                "type": "allergy_risk", "severity": "critical",
                "drug": "Amoxicillin",
                "description": "Patient has Penicillin allergy - Amoxicillin contraindicated",
                "recommendation": "Remove Amoxicillin; use Ciprofloxacin or Azithromycin alone",
            },
            {
                "type": "dosage_too_high", "severity": "warning",
                "drug": "Ciprofloxacin",
                "description": f"{cipro_dose}mg exceeds maximum daily dose of 1500mg",
                "recommendation": "Reduce to standard 500-750mg twice daily",
            },
        ]
        return prescription, patient, issues

    # ==================================================================
    #  HARD CASES — 4+ medications, compound issues, elderly/complex
    # ==================================================================

    def generate_complex_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        case = random.choice([
            self._complex_elderly_polypharmacy,
            self._complex_warfarin_nsaid_dosage,
            self._complex_psychiatric_pain,
            self._complex_cardiac_diabetes_renal,
        ])
        return case()

    def _complex_elderly_polypharmacy(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Warfarin", "dosage": "5mg", "dosage_mg": 5,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Aspirin", "dosage": "325mg", "dosage_mg": 325,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Ibuprofen", "dosage": "600mg", "dosage_mg": 600,
                 "frequency": "three times daily", "route": "oral", "duration": "14 days"},
                {"drug": "Metformin", "dosage": "1000mg", "dosage_mg": 1000,
                 "frequency": "twice daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Amoxicillin", "dosage": "500mg", "dosage_mg": 500,
                 "frequency": "three times daily", "route": "oral", "duration": "7 days"},
            ],
        }
        patient = {
            "age": random.randint(70, 88), "weight_kg": random.randint(50, 75),
            "conditions": ["Atrial fibrillation", "Type 2 Diabetes", "Osteoarthritis", "Bronchitis"],
            "allergies": ["Penicillin"],
            "current_medications": ["Warfarin", "Metformin"],
            "kidney_function": "mildly impaired", "liver_function": "normal",
        }
        issues = [
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Warfarin", "drug2": "Aspirin",
                "description": "Dual anticoagulation - severe hemorrhage risk in elderly",
                "recommendation": "Remove Aspirin or switch anticoagulation strategy",
            },
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Warfarin", "drug2": "Ibuprofen",
                "description": "NSAID with anticoagulant - compounded bleeding risk",
                "recommendation": "Replace Ibuprofen with acetaminophen",
            },
            {
                "type": "allergy_risk", "severity": "critical",
                "drug": "Amoxicillin",
                "description": "Patient has Penicillin allergy - anaphylaxis risk",
                "recommendation": "Switch to Azithromycin or Ciprofloxacin",
            },
        ]
        return prescription, patient, issues

    def _complex_warfarin_nsaid_dosage(self) -> Tuple[Dict, Dict, List[Dict]]:
        warfarin_dose = random.choice([15, 20])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Warfarin", "dosage": f"{warfarin_dose}mg", "dosage_mg": warfarin_dose,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Ibuprofen", "dosage": "600mg", "dosage_mg": 600,
                 "frequency": "three times daily", "route": "oral", "duration": "14 days"},
                {"drug": "Metformin", "dosage": "1000mg", "dosage_mg": 1000,
                 "frequency": "twice daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Sertraline", "dosage": "100mg", "dosage_mg": 100,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
            ],
        }
        patient = {
            "age": random.randint(60, 80), "weight_kg": random.randint(55, 85),
            "conditions": ["Atrial fibrillation", "Type 2 Diabetes", "Osteoarthritis", "Depression"],
            "allergies": [],
            "current_medications": ["Warfarin", "Metformin"],
            "kidney_function": "mildly impaired", "liver_function": "normal",
        }
        issues = [
            {
                "type": "dosage_too_high", "severity": "critical",
                "drug": "Warfarin",
                "description": f"{warfarin_dose}mg exceeds maximum daily dose of 10mg - serious bleeding risk",
                "recommendation": "Reduce to maximum 10mg, typical dose 2-5mg",
            },
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Warfarin", "drug2": "Ibuprofen",
                "description": "NSAIDs increase bleeding risk with anticoagulants",
                "recommendation": "Use acetaminophen instead of ibuprofen for pain",
            },
            {
                "type": "contraindication", "severity": "warning",
                "drug": "Metformin",
                "description": "Patient has impaired kidney function - metformin dose may need adjustment",
                "recommendation": "Consider dose reduction or monitor kidney function closely",
            },
        ]
        return prescription, patient, issues

    def _complex_psychiatric_pain(self) -> Tuple[Dict, Dict, List[Dict]]:
        morphine_dose = random.choice([250, 300, 350])
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Morphine", "dosage": f"{morphine_dose}mg", "dosage_mg": morphine_dose,
                 "frequency": "daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Lorazepam", "dosage": "2mg", "dosage_mg": 2,
                 "frequency": "three times daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Sertraline", "dosage": "150mg", "dosage_mg": 150,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Acetaminophen", "dosage": "1000mg", "dosage_mg": 1000,
                 "frequency": "four times daily", "route": "oral", "duration": "14 days"},
            ],
        }
        patient = {
            "age": random.randint(40, 70), "weight_kg": random.randint(55, 90),
            "conditions": ["Chronic pain", "Anxiety disorder", "Major depressive disorder"],
            "allergies": [],
            "current_medications": ["Sertraline"],
            "kidney_function": "normal",
            "liver_function": random.choice(["normal", "mildly impaired"]),
        }
        issues = [
            {
                "type": "dosage_too_high", "severity": "critical",
                "drug": "Morphine",
                "description": f"{morphine_dose}mg exceeds maximum daily dose of 200mg",
                "recommendation": "Reduce dose; consider multimodal pain approach",
            },
            {
                "type": "drug_interaction", "severity": "critical",
                "drug1": "Morphine", "drug2": "Lorazepam",
                "description": "Opioid + benzodiazepine - respiratory depression (FDA black box warning)",
                "recommendation": "Avoid concurrent use; taper one or both",
            },
            {
                "type": "dosage_too_high", "severity": "warning",
                "drug": "Acetaminophen",
                "description": "4000mg daily dose exceeds recommended 3000mg maximum",
                "recommendation": "Reduce to 3000mg max; risk of liver damage at higher doses",
            },
        ]
        return prescription, patient, issues

    def _complex_cardiac_diabetes_renal(self) -> Tuple[Dict, Dict, List[Dict]]:
        prescription = {
            "patient_id": _random_patient_id(),
            "prescriber": _random_doctor(),
            "date": _random_date(),
            "medications": [
                {"drug": "Lisinopril", "dosage": "20mg", "dosage_mg": 20,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Metformin", "dosage": "2000mg", "dosage_mg": 2000,
                 "frequency": "daily", "route": "oral", "duration": "ongoing"},
                {"drug": "Ibuprofen", "dosage": "800mg", "dosage_mg": 800,
                 "frequency": "three times daily", "route": "oral", "duration": "14 days"},
                {"drug": "Digoxin", "dosage": "0.5mg", "dosage_mg": 0.5,
                 "frequency": "once daily", "route": "oral", "duration": "ongoing"},
            ],
        }
        patient = {
            "age": random.randint(65, 85), "weight_kg": random.randint(55, 80),
            "conditions": ["Heart failure", "Type 2 Diabetes", "Severe kidney disease", "Osteoarthritis"],
            "allergies": [],
            "current_medications": ["Lisinopril", "Metformin", "Digoxin"],
            "kidney_function": "severely impaired", "liver_function": "normal",
        }
        issues = [
            {
                "type": "contraindication", "severity": "critical",
                "drug": "Metformin",
                "description": "Metformin contraindicated in severe kidney disease - lactic acidosis risk",
                "recommendation": "Switch to insulin or DPP-4 inhibitor",
            },
            {
                "type": "contraindication", "severity": "critical",
                "drug": "Ibuprofen",
                "description": "NSAIDs contraindicated in severe kidney disease - nephrotoxic",
                "recommendation": "Use acetaminophen for pain management",
            },
            {
                "type": "dosage_too_high", "severity": "critical",
                "drug": "Digoxin",
                "description": "0.5mg exceeds max 0.25mg; severe renal impairment increases toxicity",
                "recommendation": "Reduce to 0.0625-0.125mg with renal monitoring",
            },
        ]
        return prescription, patient, issues


DRUG_DB = DrugDatabase()
