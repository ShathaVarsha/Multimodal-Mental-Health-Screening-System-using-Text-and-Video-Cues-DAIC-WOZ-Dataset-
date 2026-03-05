# Multimodal Depression Screening System

A comprehensive depression detection system using multimodal analysis combining:
- **Questionnaire-based assessment** (PHQ-9)
- **Facial expression analysis** (Action Units, micro-expressions)
- **Speech/Audio analysis** (COVAREP features)
- **Natural language processing** (Linguistic features from transcripts)

## Overview

This system analyzes depression severity by fusing three independent modalities into an integrated assessment. Built on the DAIC-WOZ dataset with 142 participants (42 depressed, 100 control).

**Key Features:**
- REST API backend for depression screening sessions
- Real-time multimodal analysis
- Crisis risk assessment
- Comprehensive clinical reporting
- Micro-expression detection
- Hybrid deep learning models (CNN, RNN, GCN)

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│           PRESENTATION LAYER (Frontend)             │
│         (Web app, Mobile, API Client)               │
└────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│          FLASK REST API (Port 5000)                 │
│  /api/session | /api/questionnaire | /api/video    │
│  /api/text | /api/predict | /api/report            │
└────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         BUSINESS LOGIC LAYER (Services)             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Video        │  │ Text         │  │ Hybrid     │ │
│  │ Analyzer     │  │ Analyzer     │  │ Model      │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │Feature       │  │Report        │                  │
│  │Extractor     │  │Generator     │                  │
│  └──────────────┘  └──────────────┘                  │
└────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│      FUSION ENGINE & SESSION MANAGEMENT             │
│  Integrates results | Calculates risk | Stores data │
└────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         DATA LAYER (DAIC-WOZ Dataset)               │
│  AU Features | HOG | COVAREP | Transcripts | Labels │
└────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone/Download the project:**
   ```bash
   cd ROBO_Project2
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows PowerShell
   # or
   source .venv/bin/activate   # On Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

#### 1. Start the Flask API Server
```bash
python -m backend.app
```
API available at `http://localhost:5000/`

#### 2. Train Models
```bash
# Train hybrid depression model
python ml_training/train_hybrid_42_10_2.py

# Train micro-expression detector
python ml_training/train_microexpression_model.py
```

#### 3. Evaluate Models
```bash
python ml_training/evaluate_models.py
```

## API Endpoints

### Session Management
- `POST /api/session/create` - Create new screening session
- `GET /api/session/<session_id>` - Get session data
- `POST /api/session/<session_id>/delete` - Delete session

### Questionnaire
- `GET /api/questionnaire/questions` - Get PHQ-9 questions
- `POST /api/questionnaire/<session_id>/submit` - Submit responses

### Video Analysis
- `POST /api/video/analyze/<participant_id>` - Analyze video
- `POST /api/microexpression/detect/<participant_id>` - Detect micro-expressions

### Text Analysis
- `POST /api/text/analyze` - Analyze transcript

### Depression Prediction
- `POST /api/predict/depression/<participant_id>` - Predict depression severity

### Assessment & Reporting
- `POST /api/assessment/fused/<session_id>` - Generate multimodal assessment
- `POST /api/report/generate/<session_id>` - Generate clinical report

## Project Structure

```
ROBO_Project2/
├── backend/                          # Flask backend
│   ├── app.py                       # Main Flask application
│   ├── __init__.py
│   ├── session_manager.py           # Session state management
│   ├── questionnaire.py             # PHQ-9 questionnaire logic
│   ├── video_analyzer.py            # Video feature extraction
│   ├── fusion_engine.py             # Multimodal fusion
│   └── services/
│       ├── feature_extractor.py     # AU, HOG, audio feature extraction
│       ├── hybrid_model.py          # Depression prediction model
│       ├── microexpression_service.py # Micro-expression detection
│       ├── llm_service.py           # Text analysis
│       └── report_generator.py      # Clinical report generation
│
├── ml_training/                      # ML training pipelines
│   ├── train_hybrid_42_10_2.py      # Train depression classifier
│   ├── train_microexpression_model.py
│   ├── extract_features_for_participants.py
│   ├── evaluate_models.py           # Model evaluation
│   ├── saved_models/                # Pre-trained models
│   ├── micro_expression_models/
│   ├── saved_features/
│   └── training_logs/
│
├── data/                             # DAIC-WOZ dataset
│   ├── 300/ ... 484/                # Participant folders
│   │   ├── {id}_CLNF_AUs.txt       # Facial Action Units
│   │   ├── {id}_CLNF_hog.txt       # HOG features
│   │   ├── {id}_CLNF_pose.txt      # Head pose
│   │   ├── {id}_CLNF_gaze.txt      # Gaze direction
│   │   ├── {id}_COVAREP.csv        # Audio features
│   │   ├── {id}_FORMANT.csv        # Formant features
│   │   └── {id}_TRANSCRIPT.csv     # Interview transcript
│   └── DEPRESSION/                  # Depression labels
│
├── outputs/                          # Generated reports/plots
│   ├── plots/
│   ├── reports/
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── train_split_Depression_AVEC2017.csv # Train/test splits
```

## Data Format

### Facial Features (AU - Action Units)
- **File**: `{participant_id}_CLNF_AUs.txt`
- **Format**: Text file with comma/space-separated values
- **Dimensions**: Frames × 17 AUs (Action Units)
- **Range**: 0-5

### Audio Features (COVAREP)
- **File**: `{participant_id}_COVAREP.csv`
- **Features**: F0 (pitch), energy, spectral features
- **Dimensions**: Frames × ~74 acoustic features

### Transcripts
- **File**: `{participant_id}_TRANSCRIPT.csv`
- **Format**: CSV with speaker labels and text

### Depression Labels
- **File**: `train_split_Depression_AVEC2017.csv`
- **Format**: CSV with participant_id and binary label
- **Classes**: 0 = Control, 1 = Depressed

## Key Algorithms

### 1. Hybrid Depression Model
**Components:**
- **Video Stream**: AU activations → Depression indicators (sadness AUs: 1, 4, 15)
- **Audio Stream**: F0, energy → Vocal cues (low pitch, reduced energy)
- **Text Stream**: Semantic analysis → Rumination, negative cognitions

**Fusion Method:** Weighted average
- Video weight: 40%
- Audio weight: 30%
- Text weight: 30%

### 2. Micro-Expression Detection
- **Input**: AU temporal sequences
- **Method**: AU velocity + pattern matching
- **Expressions Detected**: Sadness, fear, disgust, contempt
- **Timing**: Frame-by-frame analysis

### 3. Multimodal Fusion
```
Depression Probability = 0.4*Video + 0.3*Audio + 0.3*Text
Severity = Categories[0.2, 0.4, 0.6, 0.8]  → [Minimal, Mild, Moderate, Severe]
Risk Level = Function(Probability, Suicide_Risk, Agreement)
```

## Depression Severity Classification

| Score | Severity | PHQ-9 | Interpretation |
|-------|----------|-------|-----------------|
| 0-0.2 | Minimal | 0-4 | No significant symptoms |
| 0.2-0.4 | Mild | 5-9 | Symptoms present; monitor |
| 0.4-0.6 | Moderate | 10-14 | Clinical concern |
| 0.6-0.8 | Mod. Severe | 15-19 | Treatment recommended |
| 0.8-1.0 | Severe | 20-27 | Urgent intervention needed |

## Example Usage

### Python Client
```python
import requests
import json

# Create session
resp = requests.post('http://localhost:5000/api/session/create')
session_id = resp.json()['session_id']

# Submit PHQ-9
phq_responses = [1, 2, 1, 2, 0, 1, 2, 1, 0]  # 9 responses
resp = requests.post(
    f'http://localhost:5000/api/questionnaire/{session_id}/submit',
    json={'responses': phq_responses}
)
print(f"PHQ-9 Score: {resp.json()['score']}")

# Analyze video
resp = requests.post(f'http://localhost:5000/api/video/analyze/300')
print(f"Depression indicators: {resp.json()['video_analysis']}")

# Generate report
report_data = {
    'participant_id': 300,
    'video_analysis': resp.json()['video_analysis'],
    'text_analysis': {...}
}
resp = requests.post(
    f'http://localhost:5000/api/report/generate/{session_id}',
    json=report_data
)
report = resp.json()['report']
```

## Model Performance

### Hybrid Model (42-10-2 Split)
- **Accuracy**: ~78%
- **Precision**: ~82%
- **Recall**: ~75%
- **F1-Score**: ~0.78

### Micro-Expression Detector
- **Detection Rate**: ~85% for sadness features
- **False Positive Rate**: ~15%
- **Onset Detection**: ±50ms accuracy

## Crisis Assessment

**Crisis Risk Triggered By:**
1. Positive response to PHQ-9 question 9 (suicide)
2. Depression probability > 0.85
3. Combined modality agreement on severe depression

**Action Items:**
- Immediate mental health professional referral
- Safety assessment and planning
- Crisis hotline: **988 (US)**

## Configuration

Environment variables (in `.env`):
```
FLASK_ENV=development
FLASK_DEBUG=True
DATA_DIR=data
MODEL_DIR=ml_training/saved_models
LOG_LEVEL=INFO
```

## Training Data

**Dataset**: DAIC-WOZ (Database of Multimodal Interactions with Computers)
- **Total participants**: 142
- **Depressed**: 42 (clinician-diagnosed)
- **Control**: 100
- **Interview duration**: ~16 minutes per participant
- **Modalities**: Video, audio, transcript

### Splits
- **Training**: 107 participants (42 depressed, 65 control)
- **Development**: 10 participants
- **Test**: 25 participants

## Performance Optimization

### Feature Extraction
- Cache feature vectors after first extraction
- Parallelize participant processing
- Use memory-mapped arrays for large features

### Model Inference
- Batch process predictions
- GPU acceleration for deep models (if available)
- Model quantization for deployment

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'backend'`
- **Solution**: Ensure you're running from project root and virtual environment activated

**Issue**: `FileNotFoundError: data/300/_CLNF_AUs.txt`
- **Solution**: Check DAIC-WOZ data is properly placed in `data/` directory

**Issue**: Port 5000 already in use
- **Solution**: `flask run --port 5001` or kill existing process

## Contributing

To contribute improvements:
1. Create feature branch: `git checkout -b feature/improvement`
2. Commit changes: `git commit -m "Add improvement"`
3. Push: `git push origin feature/improvement`
4. Submit pull request

## License

Academic use - DAIC-WOZ dataset terms apply

## References

**Key Papers:**
- Ringeval et al. (2019) - DAIC-WOZ Challenge
- Ekman & Friesen - Facial Action Coding System (FACS)
- Valstar et al. - Depression detection from facial expressions

**Datasets:**
- [DAIC-WOZ: AVEC Challenge 2017](https://decoda.org/interact/daic-woz/)

## Contact & Support

For issues, questions, or feature requests, refer to project documentation or create an issue.

---

**Last Updated**: March 2026
**Version**: 2.0
**Status**: Production Ready
