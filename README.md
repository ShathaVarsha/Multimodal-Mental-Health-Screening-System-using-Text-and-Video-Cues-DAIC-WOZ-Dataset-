
# Hybrid Multimodal Depression & PTSD Screening System

A comprehensive AI-powered system for mental health screening using multimodal analysis (text + facial expressions) for depression and a chat-only workflow for PTSD.


## 🆕 NEW FEATURES (2026)

### 1. Dual-Mode Screening: Depression & PTSD
- **Depression (Video + Chat):** Multimodal analysis using facial microexpressions and text.
- **PTSD (Chat Only):** Validated PCL/PCL-5 questions, option-based chat, and PDF reporting.

### 2. Empathetic Conversational Chat
- Human-like responses with acknowledgement and encouragement
- Natural conversation flow with empathetic feedback
- Example: "Thank you for sharing that with me. Take your time with your answer."

### 3. Answer Validation & Re-Prompting
- Detects vague answers (e.g., "idk", "ok", too short)
- Detects off-topic responses using keyword matching
- Re-prompts with specific guidance: "Could you tell me more specifically about sleep?"
- Yellow highlight for clarification requests

### 4. Real-Time Camera Integration (Depression Only)
- OpenCV-based facial feature extraction (no dummy data!)
- Extracts 34 features: 22 AU-like + 6 pose + 6 gaze
- Uses Haar Cascades (pre-trained, works offline)
- Fast processing: ~100ms per frame on CPU
- See [camera_utils.py](camera_utils.py) and [FEATURES_SUMMARY.md](FEATURES_SUMMARY.md)


## 📋 Overview


This system implements:
- **Depression (Video + Chat):**
   - Text: Conversational responses using DistilBERT
   - Visual: Facial expressions (Action Units), head pose, and gaze patterns
   - Multimodal Fusion: Combined analysis for depression risk assessment
- **PTSD (Chat Only):**
   - All PCL/PCL-5 questions with option-based chat
   - Automated PDF report generation


## 🏗️ System Architecture

```
Depression Workflow:
User Input (Text + Webcam)
         ↓
Model 1: Adaptive Dialogue (DistilBERT) → Predicts next question
         ↓
Model 2a: Text Embeddings (DistilBERT) → 768-dim semantic vectors
Model 2b: Visual Classifier (SVM) → Facial behavior analysis
         ↓
Model 3: Multimodal Fusion (Neural Network) → Depression score
         ↓
Model 4: Report Generator → Human-readable psychological report

PTSD Workflow:
User Input (Chat Only, Option Selection)
         ↓
Validated PCL/PCL-5 Questions (all included)
         ↓
Score Calculation & Severity Interpretation
         ↓
PTSD PDF Report Generator
```


## 📦 Installation

### Prerequisites
- Python 3.12.7
- Windows OS (paths configured for Windows)
- No GPU required (CPU training supported)

### Step 1: Clone/Setup Project
```bash
cd c:\Users\abhis\OneDrive\Desktop\sem 6\robo_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLP Models
```bash
# spaCy model
python -m spacy download en_core_web_sm

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Step 5: Install OpenFace (for facial feature extraction)

**Option A: Download Pre-built Binary (Recommended)**
1. Download OpenFace 2.2.0 for Windows from:
   https://github.com/TadasBaltrusaitis/OpenFace/releases
2. Extract to `C:\OpenFace` or similar location
3. Add to PATH:
   ```bash
   set PATH=%PATH%;C:\OpenFace
   ```

**Option B: Build from Source (Advanced)**
Follow instructions at: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation

**Verify Installation:**
```bash
# Check if FeatureExtraction.exe is accessible
FeatureExtraction.exe -h
```

## 🗂️ Data Structure

Your data should be organized as follows:
```
robo_project/
├── data/
│   ├── 300/
│   │   ├── 300_TRANSCRIPT.csv
│   │   ├── 300_CLNF_AUs.txt
│   │   ├── 300_CLNF_pose.txt
│   │   ├── 300_CLNF_gaze.txt
│   │   └── ...
│   ├── 301/ ... 317/
├── train_split_Depression_AVEC2017.csv (for sessions 303, 304, 305, 310, 312, 313, 315, 316, 317)
├── test_split_Depression_AVEC2017.csv (for sessions 300, 301)
└── ...
```

## 🚀 Training Pipeline

Run the training scripts in order:


### Step 1: Data Preparation
```bash
# (Missing: see project notes or use your own script)
# python step1_data_preparation.py
# (If you have a data preparation script, run it here. Otherwise, ensure your data is in the correct format as described above.)
```
*Note: step1_data_preparation.py is not present in this workspace. Prepare your data manually or restore this script if needed.*


### Step 2: Feature Engineering
```bash
python feature_engineering_2.py
```
- Computes text embeddings (DistilBERT)
- Extracts visual features (AU, pose, gaze)
- Creates aggregated feature vectors
- Computes sentiment features (VADER)
- **Output**: `outputs/text_embeddings.pkl`, `outputs/visual_features.pkl`, `outputs/sentiment_features.pkl`


### Step 3: Visual Classifier
```bash
python visual_classifier_5.py
```
- Trains SVM on visual features
- Predicts depression risk from facial behavior
- **Output**: `models/model2b_visual_svm.pkl`


### Step 4: Fusion Network
```bash
python fusion_model_6.py
```
- Trains multimodal fusion network
- Combines text + visual cues
- Predicts Depression/Anxiety/PTSD scores
- **Output**: `models/model3_fusion.pth`


### Step 5: Report Generator
```bash
python report_generator_7.py
```
- Creates report templates
- Generates human-readable reports
- **Output**: `outputs/report_templates.json`


### Step 6: Web Interface
```bash
python step8_web_enhanced.py
```
- Starts Flask web server
- Launches browser at http://127.0.0.1:5000
- Chat interface with webcam integration and PTSD chat

## 🎯 Complete Training Workflow

Run these commands in order (30-60 minutes total):

```bash
## 🎯 Complete Training Workflow

Run these commands in order (30-60 minutes total):

```bash
# Step 1: Prepare data (manual or custom script)
# python step1_data_preparation.py

# Step 2: Extract features (10-15 min)
python feature_engineering_2.py

# Step 3: Train visual classifier (2-5 min)
python visual_classifier_5.py

# Step 4: Train fusion network (15-30 min) - MAIN MODEL
python fusion_model_6.py

# Step 5: Generate report templates (1-2 min)
python report_generator_7.py

# Step 6: Launch web interface
python step8_web_enhanced.py
```
```

This allows you to:
- ✅ Save each model output separately
- ✅ Reuse models without retraining
- ✅ Debug individual components
- ✅ Run minimum 15 epochs with early stopping


## 🌐 Using the Web Interface


1. Start the server:
   ```bash
   python step8_web_enhanced.py 
   ```

2. Open browser: `http://127.0.0.1:5000`

3. On the start screen, select your screening type:
   - **Depression (Video + Chat):** Requires webcam for facial analysis. Adaptive questions, real-time feedback, and PDF report.
   - **PTSD (Chat Only):** No webcam required. Answer all PCL/PCL-5 questions via chat options. Receive a PTSD PDF report at the end.

4. For depression, grant webcam permissions when prompted. For PTSD, proceed with chat only.

5. Complete the session and download your report.


## 📊 Model Performance

Expected performance with 9 training sessions (303, 304, 305, 310, 312, 313, 315, 316, 317):

| Model | Metric | Expected |
|-------|--------|----------|
| Model 2b (Visual SVM) | MAE | 2-4 points |
| Model 2b (Visual SVM) | RMSE | 3-5 points |
| Model 3 (Fusion) | MAE | 2-3 points |
| Model 3 (Fusion) | RMSE | 3-5 points |
| Model 3 (Fusion) | R² | 0.3-0.7 |

**Training Details:**
- LOOCV (Leave-One-Out Cross-Validation)
- Minimum 15 epochs for fusion network
- Early stopping (patience=30, delta=0.001)
- SVM hyperparameter tuning via GridSearchCV

**Note**: Performance improves significantly with more training sessions.


## 📁 Output Files

All trained models and outputs are saved in:
- `models/` - Trained model weights (.pth, .pkl)
- `outputs/` - Feature embeddings, reports
- `checkpoints/` - Training checkpoints
- `logs/` - Training logs


## 🔧 Configuration

Edit [config.py](config.py) to customize:
- Session IDs
- Model hyperparameters
- Training settings
- File paths


## 🐛 Troubleshooting

### Issue: "No module named 'transformers'"
```bash
pip install transformers
```

### Issue: "OpenFace not found"
- Ensure FeatureExtraction.exe is in PATH
- Or provide full path in config.py

### Issue: "CUDA not available"
- System is configured for CPU training
- No GPU required

### Issue: "Not enough sessions for training"
- Download more sessions from DAIC-WOZ dataset
- Add session IDs to config.py


## 📚 References

- **DistilBERT**: Sanh et al., 2019
- **OpenFace**: Baltrusaitis et al., 2018
- **DAIC-WOZ Dataset**: Gratch et al., 2014


## 📄 License

For academic and research purposes only.


## 👨‍💻 Author

Depression Screening System v1.0  
Built for Robotics Project (Semester 6)


## 🙏 Acknowledgments

- DAIC-WOZ Dataset creators
- OpenFace project contributors
- Hugging Face transformers library

---

**⚠️ Important**: This system is for research purposes only. Not intended for clinical diagnosis. Always consult qualified mental health professionals.
