"""
Medical Drug Database
================================================================================

This module contains drug information, interaction rules, and validation logic.

Data sources (simplified for hackathon):
- FDA drug interaction tables
- Clinical pharmacology guidelines
- Common contraindication patterns

In production, this would connect to:
- DrugBank API
- FDA OpenFDA API
- Clinical decision support systems

Author: Suhel Mulla
"""

from typing import Dict, List, Set, Tuple, Optional


# ============================================================================
# DRUG DATABASE - Simplified medical reference
# ============================================================================

class DrugDatabase:
    """
    Centralized drug information database.
    
    In production, this would be a proper database with thousands of drugs.
    For the hackathon, we include ~20 common drugs with real interaction data.
    """
    
    def __init__(self):
        # Drug information: max/min doses, contraindications, etc.
        self.drugs = {
            # ============ CARDIOVASCULAR DRUGS ============
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
                "contraindications": ["Pregnancy", "Bilateral renal artery stenosis", "Angioedema history"],
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
            
            # ============ DIABETES DRUGS ============
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
            
            # ============ ANTIBIOTICS ============
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
            
            # ============ PAIN MEDICATIONS ============
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
            
            # ============ PSYCHIATRIC MEDICATIONS ============
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
                "contraindications": ["Acute narrow-angle glaucoma", "Severe respiratory insufficiency"],
                "interactions": ["Opioids", "Alcohol"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": True,
            },
            
            # ============ OTHER COMMON DRUGS ============
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
                "max_daily_dose_mg": 0.3,  # mcg converted to mg
                "min_daily_dose_mg": 0.025,
                "contraindications": ["Acute MI", "Thyrotoxicosis"],
                "interactions": ["Iron supplements", "Calcium", "Antacids"],
                "requires_monitoring": True,
                "requires_kidney_adjustment": False,
                "requires_liver_adjustment": False,
            },
        }
        
        # Drug interaction severity levels
        self.interaction_severity = {
            ("Warfarin", "Aspirin"): "critical",  # Major bleeding risk
            ("Warfarin", "Ibuprofen"): "critical",
            ("Warfarin", "Clopidogrel"): "critical",
            ("Aspirin", "Ibuprofen"): "warning",
            ("Lisinopril", "Spironolactone"): "warning",  # Hyperkalemia risk
            ("Metoprolol", "Verapamil"): "critical",  # Severe bradycardia
            ("Morphine", "Lorazepam"): "critical",  # Respiratory depression
            ("Sertraline", "Tramadol"): "warning",  # Serotonin syndrome risk
        }
    
    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Get information about a specific drug"""
        return self.drugs.get(drug_name.title())
    
    def check_interaction(self, drug1: str, drug2: str) -> Optional[Tuple[str, str]]:
        """
        Check if two drugs interact.
        
        Returns:
            Tuple of (severity, description) if interaction exists, None otherwise
        """
        drug1 = drug1.title()
        drug2 = drug2.title()
        
        # Check in both directions
        key1 = (drug1, drug2)
        key2 = (drug2, drug1)
        
        severity = self.interaction_severity.get(key1) or self.interaction_severity.get(key2)
        
        if severity:
            return (severity, f"{drug1} and {drug2} interaction")
        
        # Also check if drugs are in each other's interaction lists
        drug1_info = self.get_drug_info(drug1)
        drug2_info = self.get_drug_info(drug2)
        
        if drug1_info and drug2 in drug1_info.get("interactions", []):
            return ("warning", f"{drug1} interacts with {drug2}")
        
        if drug2_info and drug1 in drug2_info.get("interactions", []):
            return ("warning", f"{drug2} interacts with {drug1}")
        
        return None
    
    def check_dosage(self, drug_name: str, dosage_mg: float) -> Optional[str]:
        """
        Check if dosage is within safe range.
        
        Returns:
            Error message if out of range, None if okay
        """
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
        self,
        drug_name: str,
        patient_conditions: List[str]
    ) -> Optional[str]:
        """
        Check if drug is contraindicated for patient's conditions.
        
        Returns:
            Error message if contraindicated, None if okay
        """
        drug_info = self.get_drug_info(drug_name)
        if not drug_info:
            return None
        
        contraindications = drug_info.get("contraindications", [])
        
        for condition in patient_conditions:
            for contraindication in contraindications:
                if contraindication.lower() in condition.lower():
                    return f"{drug_name} contraindicated in {condition}"
        
        return None
    
    def check_allergy(
        self,
        drug_name: str,
        patient_allergies: List[str]
    ) -> Optional[str]:
        """
        Check if patient is allergic to the drug.
        
        Returns:
            Error message if allergic, None if okay
        """
        drug_name = drug_name.title()
        
        # Direct allergy
        if drug_name in [a.title() for a in patient_allergies]:
            return f"Patient allergic to {drug_name}"
        
        # Cross-sensitivity (e.g., Penicillin allergy → Amoxicillin)
        if "Penicillin" in patient_allergies and drug_name == "Amoxicillin":
            return f"Patient has Penicillin allergy - {drug_name} contraindicated"
        
        return None


# ============================================================================
# PRESCRIPTION GENERATOR - Creates test cases
# ============================================================================

class PrescriptionGenerator:
    """
    Generates prescription test cases with known issues.
    
    Used to create tasks of varying difficulty.
    """
    
    def __init__(self, drug_db: DrugDatabase):
        self.drug_db = drug_db
    
    def generate_safe_prescription(self) -> Tuple[Dict, Dict]:
        """
        Generate a completely safe prescription (no issues).
        
        Returns:
            (prescription_dict, patient_dict)
        """
        prescription = {
            "patient_id": "P001",
            "prescriber": "Dr. Sarah Johnson",
            "date": "2024-04-01",
            "medications": [
                {
                    "drug": "Lisinopril",
                    "dosage": "10mg",
                    "dosage_mg": 10,
                    "frequency": "once daily",
                    "route": "oral",
                    "duration": "30 days"
                }
            ]
        }
        
        patient = {
            "age": 55,
            "weight_kg": 75,
            "conditions": ["Hypertension"],
            "allergies": [],
            "current_medications": [],
            "kidney_function": "normal",
            "liver_function": "normal"
        }
        
        return prescription, patient
    
    def generate_interaction_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        """
        Generate prescription with drug-drug interaction.
        
        Returns:
            (prescription, patient, ground_truth_issues)
        """
        prescription = {
            "patient_id": "P002",
            "prescriber": "Dr. Michael Chen",
            "date": "2024-04-01",
            "medications": [
                {
                    "drug": "Warfarin",
                    "dosage": "5mg",
                    "dosage_mg": 5,
                    "frequency": "once daily",
                    "route": "oral",
                    "duration": "ongoing"
                },
                {
                    "drug": "Aspirin",
                    "dosage": "81mg",
                    "dosage_mg": 81,
                    "frequency": "once daily",
                    "route": "oral",
                    "duration": "ongoing"
                }
            ]
        }
        
        patient = {
            "age": 68,
            "weight_kg": 80,
            "conditions": ["Atrial fibrillation", "Coronary artery disease"],
            "allergies": [],
            "current_medications": ["Warfarin"],
            "kidney_function": "normal",
            "liver_function": "normal"
        }
        
        issues = [
            {
                "type": "drug_interaction",
                "severity": "critical",
                "drug1": "Warfarin",
                "drug2": "Aspirin",
                "description": "Both are anticoagulants - major bleeding risk",
                "recommendation": "Use only one anticoagulant or use under close monitoring"
            }
        ]
        
        return prescription, patient, issues
    
    def generate_dosage_error(self) -> Tuple[Dict, Dict, List[Dict]]:
        """
        Generate prescription with dosage error.
        
        Returns:
            (prescription, patient, ground_truth_issues)
        """
        prescription = {
            "patient_id": "P003",
            "prescriber": "Dr. Emily Rodriguez",
            "date": "2024-04-01",
            "medications": [
                {
                    "drug": "Metformin",
                    "dosage": "3000mg",
                    "dosage_mg": 3000,
                    "frequency": "once daily",
                    "route": "oral",
                    "duration": "30 days"
                }
            ]
        }
        
        patient = {
            "age": 62,
            "weight_kg": 85,
            "conditions": ["Type 2 Diabetes"],
            "allergies": [],
            "current_medications": [],
            "kidney_function": "normal",
            "liver_function": "normal"
        }
        
        issues = [
            {
                "type": "dosage_too_high",
                "severity": "warning",
                "drug": "Metformin",
                "description": "3000mg exceeds maximum daily dose of 2550mg",
                "recommendation": "Reduce to maximum 2550mg daily or split into multiple doses"
            }
        ]
        
        return prescription, patient, issues
    
    def generate_contraindication_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        """
        Generate prescription with contraindication.
        
        Returns:
            (prescription, patient, ground_truth_issues)
        """
        prescription = {
            "patient_id": "P004",
            "prescriber": "Dr. David Kim",
            "date": "2024-04-01",
            "medications": [
                {
                    "drug": "Amoxicillin",
                    "dosage": "500mg",
                    "dosage_mg": 500,
                    "frequency": "three times daily",
                    "route": "oral",
                    "duration": "10 days"
                }
            ]
        }
        
        patient = {
            "age": 45,
            "weight_kg": 70,
            "conditions": ["Pneumonia"],
            "allergies": ["Penicillin"],
            "current_medications": [],
            "kidney_function": "normal",
            "liver_function": "normal"
        }
        
        issues = [
            {
                "type": "allergy_risk",
                "severity": "critical",
                "drug": "Amoxicillin",
                "description": "Patient has Penicillin allergy - Amoxicillin is a penicillin derivative",
                "recommendation": "Switch to non-penicillin antibiotic (e.g., Azithromycin, Ciprofloxacin)"
            }
        ]
        
        return prescription, patient, issues
    
    def generate_complex_case(self) -> Tuple[Dict, Dict, List[Dict]]:
        """
        Generate complex prescription with multiple issues.
        
        Returns:
            (prescription, patient, ground_truth_issues)
        """
        prescription = {
            "patient_id": "P005",
            "prescriber": "Dr. Jessica Martinez",
            "date": "2024-04-01",
            "medications": [
                {
                    "drug": "Warfarin",
                    "dosage": "15mg",  # TOO HIGH
                    "dosage_mg": 15,
                    "frequency": "once daily",
                    "route": "oral",
                    "duration": "ongoing"
                },
                {
                    "drug": "Ibuprofen",  # INTERACTS WITH WARFARIN
                    "dosage": "600mg",
                    "dosage_mg": 600,
                    "frequency": "three times daily",
                    "route": "oral",
                    "duration": "14 days"
                },
                {
                    "drug": "Metformin",
                    "dosage": "1000mg",
                    "dosage_mg": 1000,
                    "frequency": "twice daily",
                    "route": "oral",
                    "duration": "ongoing"
                }
            ]
        }
        
        patient = {
            "age": 72,
            "weight_kg": 65,
            "conditions": ["Atrial fibrillation", "Type 2 Diabetes", "Osteoarthritis"],
            "allergies": [],
            "current_medications": ["Warfarin", "Metformin"],
            "kidney_function": "mildly impaired",  # REQUIRES DOSE ADJUSTMENT
            "liver_function": "normal"
        }
        
        issues = [
            {
                "type": "dosage_too_high",
                "severity": "critical",
                "drug": "Warfarin",
                "description": "15mg exceeds maximum daily dose of 10mg - serious bleeding risk",
                "recommendation": "Reduce to maximum 10mg, typical dose 2-5mg"
            },
            {
                "type": "drug_interaction",
                "severity": "critical",
                "drug1": "Warfarin",
                "drug2": "Ibuprofen",
                "description": "NSAIDs increase bleeding risk with anticoagulants",
                "recommendation": "Use acetaminophen instead of ibuprofen for pain"
            },
            {
                "type": "contraindication",
                "severity": "warning",
                "drug": "Metformin",
                "description": "Patient has impaired kidney function - metformin dose may need adjustment",
                "recommendation": "Consider dose reduction or monitor kidney function closely"
            }
        ]
        
        return prescription, patient, issues


# Global instance
DRUG_DB = DrugDatabase()