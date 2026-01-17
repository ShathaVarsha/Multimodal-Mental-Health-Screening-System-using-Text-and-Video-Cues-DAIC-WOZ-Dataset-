# 🚀 QUICK START GUIDE
## Hybrid Multimodal Depression Screening System
**Updated: December 2025** | **Python 3.12** | **CPU-Optimized**

---

## 👋 COMPLETE BEGINNER'S GUIDE (START HERE!)

**New to this project? Follow this guide step-by-step.**

### What You Need Before Starting
- ✅ **Windows PC** (you have this)
- ✅ **Python 3.12 installed** (check: `python --version`)
- ✅ **Internet connection** (for downloading models)
- ✅ **2-3 GB free disk space**
- ✅ **30-60 minutes** for training

### Your Journey: 5 Simple Steps

```
Step 1: Install Python Packages (5 min)
   ↓
Step 2: Prepare Your Data (2 min)
   ↓
Step 3: Train the AI Models (30-50 min)
   ↓
Step 4: Launch Web Interface (30 sec)
   ↓
Step 5: Start Screening! 🎉
```

---

## 📝 STEP-BY-STEP GUIDE FOR NEW USERS

### STEP 1: Open PowerShell & Navigate to Project

```powershell
# Open PowerShell (press Windows key, type "PowerShell", press Enter)

# Navigate to your project folder
cd "C:\Users\abhis\OneDrive\Desktop\sem 6\robo_project"

# Verify you're in the right place
dir
# You should see files like: step1_data_preparation.py, config.py, README.md
```

---

### STEP 2: Install Required Software (5 minutes)

**Copy and paste this entire block into PowerShell:**

```powershell
# Install essential Python packages (this will take 3-5 minutes)
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn flask nltk Pillow opencv-python

# Download NLTK language models (takes 1 minute)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

**Wait for installation to complete.** You should see "Successfully installed..." messages.

**Test installation:**
```powershell
python test_installation.py
```

Expected output: `✓ All core packages available`

---

### STEP 3: Verify Your Data (2 minutes)

**Check that you have the required data files:**

```powershell

# Check session data folders (should show .txt and .csv files)
dir data\303
dir data\304
dir data\305
dir data\310
dir data\312
dir data\313
dir data\315
dir data\316
dir data\317


# Check label files (should exist in project root)
dir train_split_Depression_AVEC2017.csv  # (for sessions 303, 304, 305, 310, 312, 313, 315, 316, 317)
dir test_split_Depression_AVEC2017.csv   # (for sessions 300, 301)
```

**If any folders/files are missing:**
- You need to download DAIC-WOZ dataset
- Contact your instructor for access
- Cannot proceed without data!

---

### STEP 4: Train the AI Models (30-50 minutes)

**Run these 5 commands ONE AT A TIME. Wait for each to complete before running the next.**

#### 4.1 Prepare Data (1-2 minutes)
```powershell
python step1_data_preparation.py
```

**What to expect:**
```
✓ Loaded 9 sessions with PHQ-8 labels
✓ Synchronized 1,234 conversation turns
✓ Saved to outputs/prepared_data.pkl
```

**If you see errors:** Check that data files exist. See [Troubleshooting](#-troubleshooting).

---

#### 4.2 Extract Features (10-15 minutes) ⏰

```powershell
python step2_feature_engineering.py
```

**What to expect:**
```
Downloading DistilBERT model... (first time only - 250MB)
✓ Generated 1,234 text embeddings (768-dim each)
✓ Computed sentiment scores
✓ Engineered 34 visual features per turn
```

**This is the longest step!** Go get coffee ☕. The first run downloads a 250MB AI model.

---

#### 4.3 Train Visual Classifier (2-5 minutes)

```powershell
python step5_model2b_visual_classifier.py
```

**What to expect:**
```
✓ Training SVM with GridSearchCV...
✓ LOOCV MAE: 3.2 points
✓ Saved to models/model2b_visual_svm.pkl
```

---

#### 4.4 Train Fusion Network (15-30 minutes) ⏰⭐

```powershell
python step6_model3_fusion.py
```

**What to expect:**
```
✓ Model: 280,193 trainable parameters
Training Fold 1/4...
  Epoch 1: train_loss=15.2, val_loss=12.3
  Epoch 50: train_loss=8.1, val_loss=9.7
  Early stopping at epoch 65
✓ LOOCV Complete: MAE=3.5, RMSE=4.2
✓ Saved to models/model3_fusion.pth
```

**This is the MAIN MODEL.** It takes 15-30 minutes. Be patient! 🕐

---

#### 4.5 Generate Report Templates (1-2 minutes)

```powershell
python step7_model4_report_generator.py
```

**What to expect:**
```
✓ Generated 5 report templates (minimal, mild, moderate, severe)
✓ Saved to outputs/report_templates.json
```

---

### STEP 5: Verify Training Completed Successfully

**Check that all required files were created:**

```powershell
# Check models folder
dir models\model3_fusion.pth       # Should show ~1-2 MB file
dir models\model2b_visual_svm.pkl  # Should show ~1 KB file

# Check outputs folder
dir outputs\text_embeddings.pkl     # Should show ~5-10 MB file
dir outputs\report_templates.json   # Should show ~5 KB file
```

**If ALL 4 files exist:** ✅ **Training successful! Proceed to Step 6.**

**If ANY file is missing:** ⚠️ That step failed. Re-run it and check `training.log` for errors.

---


### STEP 6: Launch the Web Interface (30 seconds)

```powershell
python step8_web_enhanced.py
```

**What to expect:**
```
======================================================================
      DEPRESSION & PTSD SCREENING WEB INTERFACE
======================================================================

Starting server on http://127.0.0.1:5000

Press Ctrl+C to stop the server
======================================================================
```

**Keep this PowerShell window open!** The server is running.

---

### STEP 7: Open in Your Browser 🌐

1. Open your web browser (Chrome, Edge, Firefox)
2. Type this address: **http://127.0.0.1:5000**
3. Press Enter

**You should see:** A start screen with two options:
  - **Depression (Video + Chat):** Adaptive questions, facial analysis (webcam required), and PDF report.
  - **PTSD (Chat Only):** All PCL/PCL-5 questions, option-based chat (no webcam needed), and PTSD PDF report.

---

### STEP 8: Try Your First Screening! 🎉

1. Click **"Begin Assessment"**
2. Select your screening type (Depression or PTSD)
3. For depression, grant camera permission if prompted
4. For PTSD, proceed with chat only (no camera needed)
5. Answer all questions and follow the empathetic conversation
6. Download your detailed PDF report at the end

**Tips for testing:**
- Try vague answers like "idk" to see re-prompting (depression)
- Enable webcam for facial analysis (depression)
- For PTSD, select options for each question and review the PTSD report

---

### 🎊 CONGRATULATIONS!

You've successfully:
- ✅ Installed all dependencies
- ✅ Trained 5 AI models from scratch
- ✅ Launched the web interface
- ✅ Created a working mental health screening system (Depression & PTSD)

**Next time:** Models are already trained! Just run `python step8_web_enhanced.py`

---

## 📊 TRAINING PROGRESS TRACKER

**Use this checklist to track your progress:**

```
INSTALLATION & SETUP
[ ] Opened PowerShell and navigated to project folder
[ ] Installed Python packages (pip install ...)
[ ] Downloaded NLTK data
[ ] Verified test_installation.py passes
[ ] Verified data folders exist (303, 304, 305, 310, 312, 313, 315, 316, 317, 300, 301)
[ ] Verified label CSV files exist

TRAINING (30-60 minutes total)
[ ] Step 1: Data Preparation (1-2 min)
    → outputs/prepared_data.pkl created
[ ] Step 2: Feature Engineering (10-15 min) ☕
    → outputs/text_embeddings.pkl created
    → outputs/sentiment_features.pkl created  
    → outputs/visual_features.pkl created
[ ] Step 5: Visual Classifier (2-5 min)
    → models/model2b_visual_svm.pkl created
[ ] Step 6: Fusion Network (15-30 min) ⭐☕
    → models/model3_fusion.pth created
[ ] Step 7: Report Generator (1-2 min)
    → outputs/report_templates.json created

LAUNCH & TEST
[ ] Launched web interface (python step8_web_interface.py)
[ ] Opened browser to http://127.0.0.1:5000
[ ] Completed first screening test
[ ] Tested empathetic responses (tried vague answers)
[ ] Tested webcam features (optional)
[ ] Viewed final assessment report

✅ ALL DONE! System ready for use.
```

**Save this checklist** - you can refer back if something fails.

---

## ❓ COMMON QUESTIONS FOR NEW USERS

### Q: Do I need to train every time I use it?
**A:** No! Train once (30-60 min), then just launch the web interface (30 sec).

### Q: What if I close PowerShell?
**A:** Models stay saved. Just open PowerShell again and run `python step8_web_interface.py`

### Q: Can I skip training and use someone else's models?
**A:** Yes! Copy their `models/` and `outputs/` folders to your project. Then launch interface.

### Q: What if training fails halfway?
**A:** Check which step failed, fix the issue (see Troubleshooting below), then re-run that step only.

### Q: How much disk space do models use?
**A:** About 1-2 GB total (mostly DistilBERT cache and embeddings).

### Q: Do I need internet after training?
**A:** No! Once trained, everything runs offline.

### Q: Can I train on a different computer?
**A:** Yes! Copy the `models/` and `outputs/` folders to any computer with Python installed.

### Q: What if I don't have sessions 302-304?
**A:** Download DAIC-WOZ dataset or contact your instructor. Cannot train without data.

---

## 🎯 WHAT YOU'LL BUILD

A complete AI-powered mental health screening system that:
- **Analyzes conversations** using DistilBERT (768-dim text embeddings)
- **Reads facial expressions** using OpenCV (34 features: AU, pose, gaze)
- **Predicts depression scores** (PHQ-8: 0-24 scale)
- **Generates reports** with severity classification and recommendations
- **Provides empathetic chat** with human-like responses and answer validation

---

## 🔍 DO I NEED TO TRAIN?

**Check if models already exist:**
```powershell
# Check for trained models
dir models\model3_fusion.pth
dir models\model2b_visual_svm.pkl
dir outputs\text_embeddings.pkl
```

### If Files Exist: ✅ Skip Training!
Models are already trained. Jump to [Using Pre-Trained Models](#-using-pre-trained-models)

### If Files Don't Exist: ⚠️ Train First!
Follow [Training from Scratch](#-training-from-scratch-first-time-setup)

---

## ⚡ FASTEST PATH TO RUNNING

### Option A: Training from Scratch (First Time - 30-60 min)
```powershell
# 1. Install dependencies
pip install torch transformers scikit-learn pandas flask nltk Pillow opencv-python

# 2. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# 3. Train all models (30-60 min total)
python step1_data_preparation.py
python step2_feature_engineering.py
python step5_model2b_visual_classifier.py
python step6_model3_fusion.py
python step7_model4_report_generator.py

# 4. Launch web interface
python step8_web_interface.py

# 5. Open browser → http://127.0.0.1:5000
```

### Option B: Using Pre-Trained Models (Already Trained - 30 sec)
```powershell
# 1. Verify models exist
dir models\*.pth
dir models\*.pkl

# 2. Install dependencies (if not done)
pip install flask Pillow opencv-python transformers torch

# 3. Launch web interface
python step8_web_interface.py

# 4. Open browser → http://127.0.0.1:5000
```

**Choose your path based on whether models exist!**

---

## 📦 DETAILED INSTALLATION

### Prerequisites
- **Python 3.12+** (you have 3.12.0 ✓)
- **Windows OS** (you're on Windows ✓)
- **No GPU needed** (CPU-only training ✓)
- **~2GB disk space** (models + dependencies)

### Option 1: Essential Packages Only (Recommended)
```powershell
cd "c:\Users\abhis\OneDrive\Desktop\sem 6\robo_project"

pip install torch transformers scikit-learn pandas numpy matplotlib seaborn flask nltk Pillow opencv-python
```

**Download NLTK models:**
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Option 2: Full Installation (All Features)
```powershell
pip install -r requirements.txt
```

**Includes:** spacy, textblob, opencv-contrib-python, mediapipe (5-10 min install time)

### Test Your Installation
```powershell
python test_installation.py
```

Expected output: ✓ All core packages available

---

## 🎓 TRAINING PIPELINE (Step-by-Step)

### Before You Start: Check Your Setup

**1. Verify data folder exists:**
```powershell
dir data\302\  # Should show files like 302_CLNF_AUs.txt, 302_TRANSCRIPT.csv
dir data\303\
dir data\304\
```

**2. Verify label files exist:**
```powershell
dir dev_split_Depression_AVEC2017.csv
dir train_split_Depression_AVEC2017.csv
```

**3. Check Python version:**
```powershell
python --version  # Should be 3.12+
```

If any checks fail, see [Troubleshooting](#-troubleshooting).

---

## 🚀 TRAINING FROM SCRATCH (First Time Setup)

### Complete Training Workflow

Follow these steps in order. Each step saves files that the next step needs.

---

### Overview
The system trains **5 separate models** in sequence. Each step saves outputs for the next.

| Step | Script | Time | Output | Required For |
|------|--------|------|--------|--------------|
| 1 | `step1_data_preparation.py` | 1-2 min | `prepared_data.pkl` | All steps |
| 2 | `step2_feature_engineering.py` | 10-15 min | `text_embeddings.pkl`, `visual_features.pkl` | Steps 5, 6 |
| 5 | `step5_model2b_visual_classifier.py` | 2-5 min | `model2b_visual_svm.pkl` | Step 8 (optional) |
| 6 | `step6_model3_fusion.py` | 15-30 min | `model3_fusion.pth` ⭐ | Step 8 (required) |
| 7 | `step7_model4_report_generator.py` | 1-2 min | `report_templates.json` | Step 8 (required) |

**Total Time:** 30-60 minutes (mostly Step 2 + Step 6)

### Step 1: Data Preparation
```powershell
python step1_data_preparation.py
```

**What it does:**
- Loads sessions 300-305 from `data/` folder
- Extracts PHQ-8 labels from 3 CSV files:
  - `dev_split_Depression_AVEC2017.csv` → Session 302
  - `train_split_Depression_AVEC2017.csv` → Sessions 303, 304, 305
  - `test_split_Depression_AVEC2017.csv` → Sessions 300, 301 (no labels)
- Synchronizes text transcripts with visual features (AUs, pose, gaze)
- **Saves:** `outputs/prepared_data.pkl` (4 sessions with PHQ-8 labels)

**Expected output:**
```
✓ Loaded 4 sessions with PHQ-8 labels
✓ Synchronized 1,234 conversation turns
✓ Saved to outputs/prepared_data.pkl
```

### Step 2: Feature Engineering
```powershell
python step2_feature_engineering.py
```

**What it does:**
- Extracts **DistilBERT embeddings** (768-dim) from conversation text
- Computes **VADER sentiment scores** (positive, negative, neutral, compound)
- Engineers **visual features** from raw AU/pose/gaze data (30-70 features)
- **Saves:** `text_embeddings.pkl`, `sentiment_features.pkl`, `visual_features.pkl`

**Expected output:**
```
✓ Processing with DistilBERT... (250MB download on first run)
✓ Generated 1,234 text embeddings (768-dim each)
✓ Computed sentiment scores for all turns
✓ Engineered 34 visual features per turn
```

**Note:** First run downloads DistilBERT model (~250MB) to `~/.cache/huggingface/`

---

### Step 3: Verify Step 2 Outputs

**Before continuing, check that files were created:**
```powershell
dir outputs\text_embeddings.pkl
dir outputs\sentiment_features.pkl
dir outputs\visual_features.pkl
```

**If files are missing, Step 2 failed.** Check `training.log` for errors.

---

### Step 5: Visual Classifier (SVM)
```powershell
python step5_model2b_visual_classifier.py
```

**What it does:**
- Trains **Support Vector Machine (SVM)** on visual features only
- Uses **GridSearchCV** to find best hyperparameters (C=0.1, 1, 10, 100)
- Evaluates with **LOOCV** (Leave-One-Out Cross-Validation)
- **Saves:** `models/model2b_visual_svm.pkl` (SVM + scaler)

**Expected output:**
```
✓ Training SVM with GridSearchCV...
✓ Best params: C=10, kernel=rbf
✓ LOOCV MAE: 3.2 points (on 4 sessions)
✓ Saved to models/model2b_visual_svm.pkl
```

### Step 6: Multimodal Fusion Network ⭐ (Main Model)
```powershell
python step6_model3_fusion.py
```

**What it does:**
- Trains **PyTorch neural network** combining text (768-dim) + visual (30-dim)
- Architecture:
  - Text branch: 768 → 256 (ReLU, Dropout)
  - Visual branch: 30 → 256 (ReLU, Dropout)
  - Fusion: 512 → 128 → 64 → 1 (final prediction)
  - **No BatchNorm** (incompatible with batch_size=1 in LOOCV)
- **Training:** 300 max epochs with early stopping (patience=30, delta=0.001)
- **Validation:** LOOCV (Leave-One-Out) on 4 sessions
- **Saves:** `models/model3_fusion.pth` (280,193 parameters)

**Expected output:**
```
✓ Model architecture: 280,193 trainable parameters
✓ Training Fold 1/4 (leave out session 302)...
   Epoch 1: train_loss=15.2, val_loss=12.3
   Epoch 50: train_loss=8.1, val_loss=9.7
   Early stopping at epoch 65
✓ LOOCV Complete: MAE=3.5, RMSE=4.2, R²=0.65
✓ Saved best model to models/model3_fusion.pth
```

**Note:** Runs for 15-300 epochs depending on early stopping. Expect 15-30 minutes on CPU.

### Step 7: Report Generator
```powershell
python step7_model4_report_generator.py
```

**What it does:**
- Creates **5 severity-level templates** (minimal, mild, moderate, moderately_severe, severe)
- Includes behavioral observations, recommendations, follow-up actions
- Adds **crisis resources** for severe cases (suicide hotlines)
- **Saves:** `outputs/report_templates.json`, sample reports in `outputs/sample_reports/`

**Expected output:**
```
✓ Generated 5 report templates
✓ Created sample reports for each severity level
✓ Saved to models/report_templates.json
```

---

### Step 8: Verify All Training Complete

**Check that all required files exist:**
```powershell
# Required for web interface
dir models\model3_fusion.pth          # ⭐ REQUIRED (fusion network)
dir outputs\report_templates.json     # ⭐ REQUIRED (reports)
dir outputs\text_embeddings.pkl       # ⭐ REQUIRED (DistilBERT)

# Optional but recommended
dir models\model2b_visual_svm.pkl     # Optional (visual classifier)
dir outputs\visual_features.pkl       # Optional (visual features)
```

**If all files exist:** ✅ Training complete! Proceed to [Using the Web Interface](#-using-the-web-interface)

**If files are missing:** ⚠️ Re-run the failed step and check `training.log`

---

## 💾 USING PRE-TRAINED MODELS

### If Models Already Exist

**1. Check what's available:**
```powershell
dir models\
dir outputs\
```

**2. Required files for web interface:**
- `models/model3_fusion.pth` - Fusion network (⭐ required)
- `outputs/report_templates.json` - Report templates (⭐ required)
- `outputs/text_embeddings.pkl` - DistilBERT cache (⭐ required)

**3. Launch directly:**
```powershell
python step8_web_interface.py
```

### Loading Models in Code

The web interface automatically loads models on startup:

```python
# From step8_web_interface.py (happens automatically)

# Load DistilBERT (for new text)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load SVM classifier (optional)
visual_model_data = load_pickle("models/model2b_visual_svm.pkl")

# Load Fusion Network (required)
fusion_model = MultimodalFusionNet(text_dim=768, visual_dim=30)
checkpoint = torch.load("models/model3_fusion.pth", weights_only=False)
fusion_model.load_state_dict(checkpoint["model_state_dict"])

# Load report templates (required)
report_templates = load_json("outputs/report_templates.json")
```

**You don't need to do anything!** Just run `python step8_web_interface.py`

### Re-Training Specific Models

**Re-train only one model (keeps others):**

```powershell
# Re-train fusion network only (keeps Steps 1-2 outputs)
python step6_model3_fusion.py

# Re-train visual classifier only
python step5_model2b_visual_classifier.py

# Re-generate reports only
python step7_model4_report_generator.py
```

**Re-train everything from scratch:**
```powershell
# Delete old models
Remove-Item -Recurse models\*
Remove-Item -Recurse outputs\*

# Re-run all steps
python step1_data_preparation.py
python step2_feature_engineering.py
python step5_model2b_visual_classifier.py
python step6_model3_fusion.py
python step7_model4_report_generator.py
```

---

## 🌐 USING THE WEB INTERFACE

### Launch the Server
```powershell
python step8_web_interface.py
```

**Expected output:**
```
======================================================================
              DEPRESSION SCREENING WEB INTERFACE
======================================================================

Starting server on http://127.0.0.1:5000

Press Ctrl+C to stop the server
======================================================================
```

### Open in Browser
Navigate to: **http://127.0.0.1:5000**

---

## 🎨 WEB INTERFACE FEATURES

### 1. **Empathetic Conversational Chat** 🆕
The system now responds like a human therapist:

**Example Conversation:**
```
🤖 Bot: "How has your sleep been lately?"
👤 You: "idk"

🤖 Bot: "I understand this might be difficult to talk about. 
        Could you tell me more specifically about sleep? 
        For example, how many hours do you usually sleep?"
👤 You: "I sleep around 4-5 hours and wake up a lot."

🤖 Bot: "Thank you for sharing that with me. Take your time. 
        Have you been feeling down, depressed, or hopeless?"
```

**Response Types:**
- ✅ **Acknowledgement:** "Thank you for sharing that with me."
- 💙 **Encouragement:** "Take your time with your answer."
- ⚠️ **Vague answer:** "Could you tell me more specifically about {topic}?"
- 🔄 **Off-topic:** "Let's focus on {topic}. Can you share your thoughts on that?"

### 2. **Answer Validation & Re-Prompting** 🆕
System detects and handles problematic answers:

| Issue | Detection | What Happens |
|-------|-----------|--------------|
| **Too vague** | <3 words, "idk", "ok" | Yellow message asks for more detail |
| **Off-topic** | No relevant keywords | Re-asks question with context |
| **Generic** | "not sure", single word | Encourages specific answer |

**Keyword Topics:**
- **Sleep:** sleep, insomnia, rest, tired, hours, bed
- **Mood:** sad, depressed, down, happy, feeling
- **Energy:** energy, tired, fatigue, exhausted
- **Appetite:** appetite, eat, food, hungry, weight
- **Concentration:** concentrate, focus, attention
- **Activity:** slow, moving, agitated, restless
- **Self-worth:** failure, worthless, guilty, blame
- **Thoughts:** death, suicide, harm, hurt

### 3. **Real-Time Webcam Analysis** 🆕
Enable your webcam to analyze facial expressions:

**Features Extracted (34 total):**
- **22 Action Units (AU-like):**
  - Eye openness (AU5, AU7)
  - Smile detection (AU12)
  - Brow movement (AU1, AU2, AU4)
  - Facial brightness (emotional arousal)
  
- **6 Head Pose:**
  - Pitch (up/down tilt)
  - Yaw (left/right rotation)
  - Roll (head tilt)
  - X, Y, Z position
  
- **6 Gaze Direction:**
  - Left eye: x, y, z
  - Right eye: x, y, z

**Technology:** OpenCV Haar Cascades (no external dependencies, works offline)

### 4. **Real-Time Depression Score**
As you answer questions, the system predicts your current PHQ-8 score:

```
Status bar: "Current score: 8.3/24"
```

### 5. **Final Assessment Report**
After completing all questions, you get a detailed report:

```
SEVERITY: MILD DEPRESSION
PHQ-8 SCORE: 8/24

You are experiencing symptoms consistent with mild depression...

RECOMMENDATIONS:
• Consider speaking with a mental health professional
• Maintain regular sleep schedule (7-9 hours)
• Engage in physical activity (30 min daily)
• Stay connected with supportive friends/family

FOLLOW-UP:
Schedule a follow-up screening in 2 weeks...
```

---

## 📊 EXPECTED PERFORMANCE

With your 4 training sessions (302, 303, 304):

| Model | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| **Visual SVM** | MAE | 2-4 pts | Average error 2-4 points on PHQ-8 scale |
| **Fusion Network** | MAE | 3-5 pts | Better with more training data |
| **Fusion Network** | RMSE | 4-6 pts | Root mean squared error |
| **Fusion Network** | R² | 0.5-0.7 | Moderate correlation |

**Note:** Performance improves with more training sessions. Consider downloading additional DAIC-WOZ data.

---

## 🗂️ PROJECT FILES

### Core Scripts
```
step1_data_preparation.py        # Load & sync data (1-2 min)
step2_feature_engineering.py     # DistilBERT + features (10-15 min)
step5_model2b_visual_classifier.py  # Train SVM (2-5 min)
step6_model3_fusion.py           # Train fusion network (15-30 min) ⭐
step7_model4_report_generator.py # Create templates (1-2 min)
step8_web_interface.py           # Flask web server
```

### Configuration & Utils
```
config.py                        # All hyperparameters
utils.py                         # Helper functions (logging, metrics)
camera_utils.py                  # OpenCV facial feature extraction 🆕
```

### Data & Models (Created by Training)
```
data/
  ├── 300/ ... 305/              # Raw DAIC-WOZ sessions
  
outputs/
  ├── prepared_data.pkl          # Step 1 output
  ├── text_embeddings.pkl        # Step 2: DistilBERT (768-dim)
  ├── sentiment_features.pkl     # Step 2: VADER scores
  ├── visual_features.pkl        # Step 2: Engineered features
  ├── session_aggregates.pkl     # Step 2: Session stats
  └── report_templates.json      # Step 7: 5 severity templates
  
models/
  ├── model2b_visual_svm.pkl     # Step 5: SVM classifier
  └── model3_fusion.pth          # Step 6: Fusion network ⭐
  
checkpoints/                     # Training checkpoints (auto-saved)
```

### Web Interface
```
templates/
  └── index.html                 # Chat UI with webcam
  
static/                          # CSS/JS assets (auto-created)
```

---

## 🔧 CONFIGURATION

Edit `config.py` to customize training:

```python
# Data
TRAIN_SESSIONS = ["302", "303", "304"]  # Remove "305" if folder is empty

# Model 3: Fusion Network
MODEL3_CONFIG = {
    "num_epochs": 300,              # Max epochs (early stopping may stop sooner)
    "early_stopping_patience": 30,  # Stop if no improvement for 30 epochs
    "early_stopping_delta": 0.001,  # Minimum improvement threshold
    "batch_size": 1,                # LOOCV uses batch_size=1
    "learning_rate": 0.001,
    "hidden_dims": [256, 128, 64],  # Network architecture
}

# Web Interface
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True
}
```

---

## 🐛 TROUBLESHOOTING

### "No module named 'transformers'"
```powershell
pip install transformers
```

### "Session 305 has 0 files"
Check if `data/305/` exists. If empty, edit `config.py`:
```python
TRAIN_SESSIONS = ["302", "303", "304"]  # Removed 305
```

### "No PHQ-8 labels loaded"
Verify these CSV files exist in project root:
- `dev_split_Depression_AVEC2017.csv`
- `train_split_Depression_AVEC2017.csv`

### "DistilBERT download failed"
Check internet connection. Model downloads from HuggingFace (~250MB).

### "Web interface won't start"
```powershell
pip install flask Pillow opencv-python
```

### "Camera features return all zeros"
Enable webcam permission in browser. If no face detected, features will be zeros.
### "Model not found" error in web interface
Models weren't trained. Run training steps first:
```powershell
python step1_data_preparation.py
python step2_feature_engineering.py
python step6_model3_fusion.py
python step7_model4_report_generator.py
```

### "FileNotFoundError: text_embeddings.pkl"
Step 2 didn't complete successfully. Re-run:
```powershell
python step2_feature_engineering.py
```

### Want to skip training?
You need models from someone else who already trained. Copy these folders:
- `models/` (contains model3_fusion.pth, model2b_visual_svm.pkl)
- `outputs/` (contains text_embeddings.pkl, report_templates.json)

Then just run: `python step8_web_interface.py`
### Training errors
Check `training.log` for detailed error messages.

---

## 💡 TIPS & TRICKS

### Speed Up Training
- **Use fewer epochs:** Edit `MODEL3_CONFIG["num_epochs"] = 50` in config.py
- **Skip visual model:** Only run steps 1, 2, 6, 7 (fusion uses text primarily)
- **Reduce LOOCV folds:** Not recommended, but possible in step6

### Improve Accuracy
- **Add more data:** Download additional DAIC-WOZ sessions
- **Adjust architecture:** Modify `hidden_dims` in config.py
- **Tune hyperparameters:** Experiment with learning_rate, dropout

### Test Without Webcam
- Webcam is optional - system works with text only
- To disable: Comment out webcam capture code in templates/index.html

### Save Intermediate Checkpoints
Training auto-saves checkpoints to `checkpoints/` folder. To resume:
- Check for `model3_fold_*.pth` files
- Modify step6 to load checkpoint (requires code edit)

---

## 📚 ADDITIONAL RESOURCES

- **Full Documentation:** See [README.md](README.md)
- **Feature Details:** See [FEATURES_SUMMARY.md](FEATURES_SUMMARY.md)
- **Camera Implementation:** See [camera_utils.py](camera_utils.py)
- **Training Logs:** Check `training.log` after each run
- **Model Architecture:** Documented in each step file's docstring

---

## ✨ FEATURE CHECKLIST

✅ **Data Pipeline**
- [x] Load 4 DAIC-WOZ sessions with PHQ-8 labels
- [x] Synchronize text + visual modalities
- [x] Handle missing data gracefully

✅ **Feature Engineering**
- [x] DistilBERT text embeddings (768-dim)
- [x] VADER sentiment analysis
- [x] Visual feature engineering (34 features)
- [x] Session-level aggregates

✅ **Model Training**
- [x] SVM visual classifier with GridSearchCV
- [x] Multimodal fusion network (280K parameters)
- [x] LOOCV cross-validation
- [x] Early stopping (prevents overfitting)
- [x] Model checkpointing (saves best model)

✅ **Report Generation**
- [x] 5 severity-level templates
- [x] Behavioral observations
- [x] Clinical recommendations
- [x] Crisis resources for severe cases

✅ **Web Interface**
- [x] Beautiful chat UI
- [x] Empathetic conversational responses 🆕
- [x] Answer validation & re-prompting 🆕
- [x] Real-time webcam facial analysis 🆕
- [x] Live depression score display
- [x] Final psychological report

---

## 🎯 QUICK REFERENCE

| Task | Command |
|------|---------|
| **Check if models exist** | `dir models\*.pth` and `dir outputs\*.pkl` |
| **Install dependencies** | `pip install torch transformers scikit-learn pandas flask nltk Pillow opencv-python` |
| **Train from scratch** | Run steps 1, 2, 5, 6, 7 in sequence (30-60 min) |
| **Re-train fusion network** | `python step6_model3_fusion.py` |
| **Re-train visual SVM** | `python step5_model2b_visual_classifier.py` |
| **Load pre-trained models** | `python step8_web_interface.py` (loads automatically) |
| **Web interface** | `python step8_web_interface.py` → http://127.0.0.1:5000 |
| **Test camera** | `python camera_utils.py` |
| **Check errors** | View `training.log` |
| **Reset everything** | Delete `models/` and `outputs/` folders, re-run training |

---

## 🔄 TRAINING WORKFLOW DIAGRAM

```
START
  ↓
[Check if models exist?]
  ↓              ↓
 NO             YES
  ↓              ↓
Install      Launch web
dependencies  interface
  ↓              ↓
Step 1:       DONE ✅
Prepare data
  ↓
Step 2:
Extract features
(10-15 min)
  ↓
Step 5:
Train SVM
(2-5 min)
  ↓
Step 6:
Train fusion ⭐
(15-30 min)
  ↓
Step 7:
Generate reports
(1-2 min)
  ↓
Launch web
interface
  ↓
DONE ✅
```

---

## 🚀 YOU'RE READY!

Your complete depression screening system is ready to use. 

**Start here:**
```powershell
python step8_web_interface.py
```

Then open **http://127.0.0.1:5000** and start a screening session!

Good luck with your Semester 6 Robotics Project! 🎓🤖

---

*Last updated: December 28, 2025*  
*Questions? Check [README.md](README.md) or review `training.log`*
