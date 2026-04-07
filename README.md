---
title: Medical Prescription Validation
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Medical Prescription Validation Environment

**🎥 [Watch the 2-Minute Demo Video](https://drive.google.com/file/d/1iEd4d98U0d2udv3411JHuKzx-IecQP9G/view?usp=sharing)**

A reinforcement learning environment for training AI agents to detect medication errors.
Agents review prescriptions for drug interactions, dosage errors, contraindications,
and allergy risks, acting as a clinical pharmacist safety layer.

## Problem Statement

Medication errors cause an estimated 7,000–9,000 deaths annually in the United States.
Common errors include dangerous drug-drug interactions, dosage errors, prescribing
contraindicated medications, and ignoring patient allergies.

This environment provides a controlled setting to train and evaluate AI agents on
these real-world clinical safety tasks.

## Tasks

| Task   | Difficulty | Description                      | Medications | Issues | Expected Steps |
| ------ | ---------- | -------------------------------- | ----------- | ------ | -------------- |
| easy   | Low        | Single drug, 0–1 simple issues   | 1           | 0–1    | 3–5            |
| medium | Medium     | Multi-drug with interactions     | 2–3         | 1–2    | 5–10           |
| hard   | High       | Complex patient, multiple issues | 4+          | 2–4    | 10–20          |

### Examples

**Easy** — A 55-year-old patient with Hypertension prescribed Lisinopril 10mg once daily.
Expected action: `approve`.

**Medium** — A 68-year-old patient on Warfarin 5mg and Aspirin 81mg together.
Expected action: `flag_interaction` (critical bleeding risk).

**Hard** — A 72-year-old with impaired kidneys prescribed Warfarin 15mg (overdose),
Ibuprofen 600mg (interacts with Warfarin), and Metformin 1000mg (needs kidney adjustment).
Expected actions: `flag_dosage`, `flag_interaction`, `flag_contraindication`.

## Action Space

```python
class PrescriptionAction(Action):
    action_type: Literal[
        "approve",
        "flag_interaction",
        "flag_dosage",
        "flag_contraindication",
        "flag_allergy",
        "request_clarification",
        "reject"
    ]
    drug_name: Optional[str]
    issue_type: Optional[str]
    severity: Literal["critical", "warning", "info"]
    recommendation: str
    rationale: Optional[str]
```

## Observation Space

```python
class PrescriptionObservation(Observation):
    done: bool
    reward: float
    prescription: Dict
    patient_info: Dict
    validation_results: List[Dict]
    current_issues: List[Dict]
    feedback: str
    task_id: str
    step_count: int
    available_actions: List[str]
```

## Reward Structure

| Action              | Outcome           | Reward | Rationale                   |
| ------------------- | ----------------- | ------ | --------------------------- |
| Flag critical       | Correctly found   | +1.0   | Prevented patient harm      |
| Flag warning        | Correctly found   | +0.5   | Caught potential issue      |
| Approve             | Prescription safe | +1.0   | Correct clinical judgment   |
| Approve             | Has issues        | -1.0   | Patient at risk             |
| Flag issue          | False positive    | -0.2   | Unnecessary delay in care   |
| Reject              | Prescription safe | -0.5   | Unnecessary rejection       |
| Good recommendation | Detailed plan     | +0.1   | Better patient care (bonus) |

Success threshold: score >= 0.7 (70% safety rating).

## Quick Start

### Installation

```bash
pip install git+https://huggingface.co/spaces/suhellll/prescription-validator
```

### Usage

```python
from prescription_validator import PrescriptionValidationEnv, PrescriptionAction

async with PrescriptionValidationEnv(
    base_url="https://suhellll-prescription-validator.hf.space"
) as env:
    result = await env.reset(task_id="easy")

    action = PrescriptionAction(
        action_type="approve",
        recommendation="Prescription is safe"
    )
    result = await env.step(action)
    print(f"Reward: {result.reward}")
```

## Docker Deployment

```bash
docker build -t prescription-validator:latest .
docker run -d -p 7860:7860 prescription-validator:latest
curl http://localhost:7860/health   # {"status":"healthy"}
```

### Environment Variables

| Variable                       | Default   | Description     |
| ------------------------------ | --------- | --------------- |
| `PORT`                         | `7860`    | Server port     |
| `HOST`                         | `0.0.0.0` | Bind address    |
| `WORKERS`                      | `4`       | Uvicorn workers |
| `OPENENV_ENABLE_WEB_INTERFACE` | `false`   | Enable web UI   |

## Running Inference

```bash
export HF_TOKEN="your_hugging_face_token"
export IMAGE_NAME="suhellll/prescription-validator"
export TASK_NAME="medium"
python inference.py
```

Expected output:

```
[START] task=medium env=prescription_validator model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"flag_interaction","drug_name":"Warfarin"} reward=1.00 done=false error=null
[STEP] step=2 action={"action_type":"reject","recommendation":"Unsafe"} reward=0.50 done=true error=null
[END] success=true steps=2 rewards=1.00,0.50
```

## Baseline Performance

| Agent                | Easy | Medium | Hard | Avg Score |
| -------------------- | ---- | ------ | ---- | --------- |
| Random               | 0.15 | 0.08   | 0.03 | 0.09      |
| Rule-based Heuristic | 0.78 | 0.58   | 0.41 | 0.59      |
| GPT-4                | 0.94 | 0.82   | 0.68 | 0.81      |
| Qwen-2.5-72B         | 0.92 | 0.79   | 0.65 | 0.79      |
| Claude-3.5-Sonnet    | 0.96 | 0.88   | 0.74 | 0.86      |

## Drug Database

14 common medications across 6 categories:

| Category       | Drugs                                     |
| -------------- | ----------------------------------------- |
| Cardiovascular | Warfarin, Aspirin, Lisinopril, Metoprolol |
| Diabetes       | Metformin, Insulin                        |
| Antibiotics    | Amoxicillin, Ciprofloxacin                |
| Pain           | Ibuprofen, Morphine                       |
| Psychiatric    | Sertraline, Lorazepam                     |
| Other          | Omeprazole, Atorvastatin, Levothyroxine   |

Each drug includes safe dose ranges, known interactions, contraindications, and
organ function requirements. Data is simplified from FDA drug interaction guidelines
for educational use.

## Project Structure

```
prescription-validator/
├── models.py              # Type definitions (Action, Observation, State)
├── client.py              # Client interface
├── server/
│   ├── environment.py     # Core environment logic
│   ├── app.py             # FastAPI server
│   ├── drug_database.py   # Medical knowledge base
│   └── __init__.py
├── inference.py           # Evaluation script
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── openenv.yaml           # Environment metadata
├── pyproject.toml         # Package configuration
├── .dockerignore
└── README.md
```

## Disclaimer

This environment is for AI research and training only. It is not FDA-approved,
not a substitute for licensed pharmacists, and not validated for real patient care.
The drug data is simplified from real guidelines for educational purposes.

## License

MIT License — Copyright (c) 2026 Suhel Mulla

## Acknowledgments

- OpenEnv Team — framework
- Meta PyTorch — hackathon sponsorship
- Hugging Face — infrastructure and hosting
- FDA — public drug interaction data
