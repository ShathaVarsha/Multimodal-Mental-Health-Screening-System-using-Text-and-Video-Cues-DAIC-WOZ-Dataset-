# Multimodal Mental Health Screening System (Text + Video)

This project is a Flask-based backend + web client pipeline for screening depression/PTSD signals from:
- Conversational questionnaires (PHQ-9, detailed depression, PCL-5)
- Video-derived facial Action Units (AUs)
- Text responses interpreted into assessment scales
- Fused multimodal scoring and report generation

It is designed around DAIC-WOZ style participant folders and split files in this repository.

---

## 1) What is implemented

### Core capabilities
- Session-based assessment workflow (`/api/session/*`)
- Conversational questionnaire APIs with scenario personalization
- Scenario-specific PHQ-9 follow-up questions loaded from data files
- Text response interpretation (natural language -> 0..3 scale)
- Video processing + AU extraction + AU activation detection
- Video + text depression severity prediction (audio currently disabled in fusion)
- Fused risk output + report generation

### Current model/fusion behavior
- `HybridDepressionModel` uses **video + text** weighting:
  - video: `0.6`
  - text: `0.4`
- Audio score is intentionally set to `None` in prediction flow.
- AU detector reports **individual AU activations** (e.g., `AU1 - Inner brow raiser`) instead of emotion labels.

---

## 2) Repository structure

```text
ROBO_Project2/
├─ backend/
│  ├─ app.py                          # Main Flask API (all routes)
│  ├─ session_manager.py              # Session lifecycle/state
│  ├─ questionnaire.py                # PHQ-9 schema/validation/scoring
│  ├─ fusion_engine.py                # Multimodal fusion logic
│  ├─ video_analyzer.py               # Participant-level video analysis
│  └─ services/
│     ├─ assessment_data_loader.py    # Loads question bank + split divisions
│     ├─ feature_extractor.py         # Feature loading/extraction helpers
│     ├─ hybrid_model.py              # Video+text severity inference
│     ├─ llm_service.py               # Text feature analysis
│     ├─ microexpression_service.py   # AU activation detector
│     ├─ report_generator.py          # Clinical-style report payloads
│     ├─ text_interpreter.py          # NLP response interpretation + scenario detect
│     └─ video_processor.py           # Uploaded video processing path
│
├─ ml_training/
│  ├─ train_complete_pipeline.py      # Multi-model training pipeline
│  ├─ train_hybrid_42_10_2.py         # Hybrid setup training
│  ├─ train_microexpression_model.py  # AU/microexpression-related training
│  ├─ extract_features_for_participants.py
│  ├─ evaluate_models.py
│  ├─ model_usage_examples.py
│  └─ MODEL_LOADING_GUIDE.py
│
├─ data/
│  ├─ question_bank.json              # PHQ-9, detailed, PCL-5, scenario followups
│  └─ <participant_id>/               # AU/gaze/pose/transcript/etc files
│
├─ test_client.html                   # Browser UI client
├─ requirements.txt
└─ README.md
```

---

## 3) Data contracts used by code

### Questionnaire bank
`data/question_bank.json` is expected to contain sections such as:
- `phq9`
- `depression_detailed`
- `pcl5`
- `phq9_scenario_followups` (keyed by scenario, each list mapped by `phq9_index`)

### Participant split files
Read by `AssessmentDataLoader.get_divisions()`:
- `train_split_Depression_AVEC2017.csv`
- `test_split_Depression_AVEC2017.csv`
- `dev_split_Depression_AVEC2017.csv`
- `full_test_split.csv`

### Participant folder example
For participant `300`:
- `data/300/300_CLNF_AUs.txt`
- `data/300/300_CLNF_gaze.txt`
- `data/300/300_CLNF_pose.txt`
- `data/300/300_TRANSCRIPT.csv`
- (other files may exist and be used by training scripts)

---

## 4) Backend API reference (from `backend/app.py`)

### Session routes
- `POST /api/session/create`
- `GET /api/session/<session_id>`
- `POST /api/session/<session_id>/delete`

### Questionnaire routes
- `GET /api/questionnaire/questions`
- `POST /api/questionnaire/<session_id>/submit`

### Conversational assessment routes
- `POST /api/assessment/conversational/interpret`
- `GET /api/assessment/conversational/phq9`
- `GET /api/assessment/conversational/pcl5`
- `GET /api/assessment/conversational/depression-detailed`
- `GET /api/assessment/divisions`
- `POST /api/assessment/conversational/scenario-identify`
- `POST /api/assessment/conversational/submit`

### Video/text/prediction routes
- `POST /api/video/analyze/<int:participant_id>`
- `POST /api/microexpression/detect/<int:participant_id>`
- `POST /api/video/upload-and-analyze`
- `POST /api/text/analyze`
- `POST /api/predict/depression/<int:participant_id>`

### Fusion/report/health
- `POST /api/assessment/fused/<session_id>`
- `POST /api/report/generate/<session_id>`
- `GET /api/health`
- `GET /`

---

## 5) Setup (Windows PowerShell)

```powershell
cd C:\Users\abhis\OneDrive\Desktop\sem_6\ROBO_Project2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If execution policy blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 6) Run the app

### Terminal 1: Backend
```powershell
python -m backend.app
```
Expected base URL: `http://localhost:5000`

### Terminal 2: Frontend client
```powershell
python -m http.server 8000
```
Open: `http://localhost:8000/test_client.html`

---

## 7) Quick API smoke tests

### Health
```powershell
Invoke-WebRequest http://localhost:5000/api/health -UseBasicParsing | Select-Object -ExpandProperty Content
```

### Get PHQ-9 conversational questions with followups
```powershell
Invoke-WebRequest "http://localhost:5000/api/assessment/conversational/phq9?scenario=postpartum" -UseBasicParsing |
  Select-Object -ExpandProperty Content
```

### Interpret free-text response into scale
```powershell
$body = @{ response = "I feel low almost every day"; question = "How have you been feeling?" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:5000/api/assessment/conversational/interpret -Method POST -ContentType "application/json" -Body $body -UseBasicParsing |
  Select-Object -ExpandProperty Content
```

---

## 8) Training scripts guide (`ml_training`)

### Main training/evaluation scripts
- `train_complete_pipeline.py`
  - Trains multiple expert models across modalities
  - Includes feature loaders for AU, gaze, pose, landmarks, audio, transcript
  - Saves models to `ml_training/saved_models/`
- `train_hybrid_42_10_2.py`
  - Hybrid workflow using predefined participant split
- `train_microexpression_model.py`
  - Training flow related to AU/microexpression modeling
- `evaluate_models.py`
  - Evaluates saved models and prints metrics
- `extract_features_for_participants.py`
  - Batch feature extraction helper

Run examples:
```powershell
python ml_training\train_complete_pipeline.py
python ml_training\train_hybrid_42_10_2.py
python ml_training\train_microexpression_model.py
python ml_training\evaluate_models.py
```

---

## 9) AU detector behavior (current)

`backend/services/microexpression_service.py` currently detects **individual AU activations** from AU intensity time series.

### Key parameters
- `PEAK_DETECTION_THRESHOLD = 0.05`
- `PATTERN_ACTIVATION_THRESHOLD = 0.12`
- `CONFIDENCE_THRESHOLD = 0.2`
- `MINIMUM_FRAMES = 2`

### Output format (per detection)
- `expression`: e.g., `AU12 - Lip corner puller`
- `au_index`
- `au_name`
- `confidence`
- `start_time`, `peak_time`, `end_time`, `duration`
- `intensity`
- `question_id` (if provided)

---

## 10) Important implementation notes

- Audio is still present in some data/training scripts, but runtime fusion in `HybridDepressionModel` is currently video+text.
- `requirements.txt` includes both classic ML and deep learning stacks (scikit-learn, TensorFlow, PyTorch).
- `AssessmentDataLoader` gracefully falls back to empty results if files are missing.
- PHQ-9 endpoint supports scenario followups through `data/question_bank.json`.

---

## 11) Troubleshooting

### Backend exits immediately
1. Ensure venv is active.
2. Install dependencies again: `pip install -r requirements.txt`.
3. Run with full error output:
   ```powershell
   python -m backend.app
   ```
4. Check file paths exist (`data/`, split CSVs, question bank).

### Port already in use
```powershell
taskkill /F /IM python.exe
```
Then restart backend/frontend.

### No AU detections
- Verify AU file has numeric intensity columns.
- Check participant folder and filename formatting.
- Increase `time_window` or lower thresholds carefully.

---

## 12) Minimal end-to-end flow

1. Create session.
2. Load conversational PHQ-9 with scenario.
3. Submit questionnaire responses.
4. Analyze video/text.
5. Run depression prediction.
6. Generate fused assessment + report.

---

## 13) Disclaimer

This system is for research/prototyping and screening support only. It is **not** a standalone diagnostic tool and should not replace licensed clinical evaluation.
