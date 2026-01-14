"""
=================================================================
CONFIGURATION FILE - Depression Screening System
=================================================================
All global settings, paths, and hyperparameters in one place
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Root project directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Label files
DEV_SPLIT_FILE = PROJECT_ROOT / "dev_split_Depression_AVEC2017.csv"
TRAIN_SPLIT_FILE = PROJECT_ROOT / "train_split_Depression_AVEC2017.csv"
TEST_SPLIT_FILE = PROJECT_ROOT / "test_split_Depression_AVEC2017.csv"

# =============================================================================
# SESSION CONFIGURATION
# =============================================================================

# Available sessions (based on user's data folder)
ALL_SESSIONS = ["300", "301", "302", "303", "304", "305"]

# Sessions with PHQ-8 labels
SESSIONS_WITH_LABELS = {
    "302": "dev",      # dev_split_Depression_AVEC2017.csv
    "303": "train",    # train_split_Depression_AVEC2017.csv
    "304": "train",
    "305": "train",
    "300": "test",     # test_split (NO PHQ VALUES)
    "301": "test"
}

# Training sessions (those with PHQ-8 labels)
TRAIN_SESSIONS = ["302", "303", "304", "305"]

# =============================================================================
# DATA EXTRACTION SETTINGS
# =============================================================================

# File patterns for each session
FILE_PATTERNS = {
    "transcript": "{session_id}_TRANSCRIPT.csv",
    "action_units": "{session_id}_CLNF_AUs.txt",
    "pose": "{session_id}_CLNF_pose.txt",
    "gaze": "{session_id}_CLNF_gaze.txt",
    "features_2d": "{session_id}_CLNF_features.txt",
    "features_3d": "{session_id}_CLNF_features3D.txt"
}

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.7  # For OpenFace outputs
SUCCESS_THRESHOLD = 1       # OpenFace success flag must be 1

# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================

# Action Units (Depression-relevant)
DEPRESSION_RELATED_AUS = {
    "AU01": "Inner Brow Raiser (sadness)",
    "AU04": "Brow Lowerer (anger/confusion)",
    "AU06": "Cheek Raiser (genuine smile)",
    "AU10": "Upper Lip Raiser (disgust)",
    "AU12": "Lip Corner Puller (smile/happiness)",
    "AU15": "Lip Corner Depressor (sadness)",
    "AU17": "Chin Raiser (tension)",
    "AU25": "Lips Part (surprise/emotion)"
}

# Head Pose thresholds (degrees)
YAW_THRESHOLD = 25      # Left/right head rotation
PITCH_THRESHOLD = 20    # Up/down head rotation
ROLL_THRESHOLD = 15     # Head tilt

# Gaze thresholds
GAZE_FORWARD_THRESHOLD = 0.7  # Looking at camera/screen
GAZE_DOWNWARD_THRESHOLD = -0.3  # Looking down (avoidance)

# =============================================================================
# MODEL 1: ADAPTIVE DIALOGUE MODEL (DistilBERT)
# =============================================================================

MODEL1_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 5,  # Question types: OPENING, PHQ_QUESTION, FOLLOW_UP, EMOTIONAL, CLOSING
    "max_length": 512,
    "batch_size": 4,  # Small batch for CPU
    "learning_rate": 2e-5,
    "num_epochs": 20,
    "early_stopping_patience": 5,
    "save_path": MODELS_DIR / "model1_dialogue.pth"
}

# Question type labels
QUESTION_TYPES = {
    0: "OPENING",
    1: "PHQ_QUESTION",
    2: "FOLLOW_UP",
    3: "EMOTIONAL",
    4: "CLOSING"
}

# =============================================================================
# MODEL 2a: TEXT FEATURE EXTRACTOR (DistilBERT)
# =============================================================================

MODEL2A_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "embedding_dim": 768,  # DistilBERT output dimension
    "max_length": 128,
    "batch_size": 8,
    "save_path": OUTPUTS_DIR / "text_embeddings.pkl"
}

# =============================================================================
# MODEL 2b: VISUAL FEATURE CLASSIFIER (SVM)
# =============================================================================

MODEL2B_CONFIG = {
    "classifier_type": "svm",  # Options: 'svm' or 'lstm'
    "kernel": "rbf",  # For SVM: 'linear', 'rbf', 'poly'
    "C_range": [0.1, 1, 10, 100],  # SVM regularization
    "gamma": "auto",
    "save_path": MODELS_DIR / "model2b_visual_svm.pkl"
}

# For LSTM alternative
MODEL2B_LSTM_CONFIG = {
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dropout": 0.3,
    "dense_units": 16,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "batch_size": 1,  # Small batch for limited data
    "early_stopping_patience": 20,
    "save_path": MODELS_DIR / "model2b_visual_lstm.pth"
}

# =============================================================================
# MODEL 3: MULTIMODAL FUSION NETWORK
# =============================================================================

MODEL3_CONFIG = {
    # Input dimensions
    "text_embedding_dim": 768,
    "text_sentiment_dim": 10,
    "visual_feature_dim": 70,  # AU (22) + Pose (10) + Gaze (5) + dynamics (~33)
    
    # Architecture
    "hidden_dims": [256, 128, 64],  # For fusion network
    "text_branch_dims": [256, 128],
    "visual_branch_dims": [128, 64],
    "fusion_dims": [96, 48],
    
    # Output heads
    "num_outputs": 3,  # Depression, Anxiety, PTSD scores
    
    # Training
    "learning_rate": 0.001,
    "num_epochs": 300,
    "epochs": 300,  # Alias for compatibility
    "batch_size": 2,  # Very small for limited sessions
    "dropout": 0.3,
    "early_stopping_patience": 30,
    
    # Loss weights (multi-task learning)
    "loss_weights": {
        "depression": 0.5,
        "anxiety": 0.25,
        "ptsd": 0.25
    },
    
    "save_path": MODELS_DIR / "model3_fusion.pth"
}

# =============================================================================
# MODEL 4: REPORT GENERATOR
# =============================================================================

MODEL4_CONFIG = {
    "generation_mode": "template",  # Options: 'template' or 'distilbert'
    "template_path": PROJECT_ROOT / "report_templates.json",
    "save_path": OUTPUTS_DIR / "generated_reports"
}

# PHQ-8 score interpretation
PHQ8_INTERPRETATION = {
    (0, 4): "Minimal depression",
    (5, 9): "Mild depression",
    (10, 14): "Moderate depression",
    (15, 19): "Moderately severe depression",
    (20, 24): "Severe depression"
}

# =============================================================================
# WEB INTERFACE SETTINGS
# =============================================================================

WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
    "webcam_fps": 30,
    "frame_buffer_size": 10,  # Capture 10 seconds of video per response
    "max_session_duration": 600,  # 10 minutes max per session
}

# =============================================================================
# GENERAL TRAINING SETTINGS
# =============================================================================

# Cross-validation
USE_LOOCV = True  # Leave-One-Out Cross-Validation (for small datasets)
RANDOM_SEED = 42

# Early stopping
EARLY_STOPPING_DELTA = 0.001  # Minimum improvement threshold

# Device
DEVICE = "cpu"  # User doesn't have GPU

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "training.log"

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("DEPRESSION SCREENING SYSTEM - CONFIGURATION")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Training Sessions: {TRAIN_SESSIONS}")
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Use LOOCV: {USE_LOOCV}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
