"""
=================================================================
UTILITY FUNCTIONS - Depression Screening System
=================================================================
Helper functions used across all training scripts
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: Path, level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# =============================================================================
# FILE I/O
# =============================================================================

def save_pickle(data: Any, filepath: Path):
    """Save data as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Saved: {filepath}")

def load_pickle(filepath: Path) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Loaded: {filepath}")
    return data

def save_json(data: Dict, filepath: Path):
    """Save data as JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"✓ Saved: {filepath}")

def load_json(filepath: Path) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded: {filepath}")
    return data

def save_model(model: torch.nn.Module, filepath: Path):
    """Save PyTorch model"""
    torch.save(model.state_dict(), filepath)
    print(f"✓ Model saved: {filepath}")

def load_model(model: torch.nn.Module, filepath: Path, device: str = "cpu"):
    """Load PyTorch model"""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"✓ Model loaded: {filepath}")
    return model

# =============================================================================
# DATA LOADING
# =============================================================================

def load_session_data(session_id: str, data_dir: Path, file_patterns: Dict) -> Dict:
    """
    Load all data files for a single session
    
    Args:
        session_id: Session identifier (e.g., "302")
        data_dir: Base data directory
        file_patterns: Dictionary mapping data types to file patterns
        
    Returns:
        Dictionary containing all loaded dataframes
    """
    session_path = data_dir / session_id
    data = {}
    
    if not session_path.exists():
        print(f"❌ Session path not found: {session_path}")
        return data
    
    # Load transcript
    transcript_file = session_path / file_patterns["transcript"].format(session_id=session_id)
    if transcript_file.exists():
        try:
            try:
                data["transcript"] = pd.read_csv(transcript_file, sep="\t")
            except:
                data["transcript"] = pd.read_csv(transcript_file)
            print(f"  ✓ Transcript: {len(data['transcript'])} utterances")
        except Exception as e:
            print(f"  ❌ Transcript error: {e}")
    
    # Load action units
    au_file = session_path / file_patterns["action_units"].format(session_id=session_id)
    if au_file.exists():
        try:
            data["action_units"] = pd.read_csv(au_file, sep=", ", engine="python")
            print(f"  ✓ Action Units: {len(data['action_units'])} frames")
        except Exception as e:
            print(f"  ❌ Action Units error: {e}")
    
    # Load head pose
    pose_file = session_path / file_patterns["pose"].format(session_id=session_id)
    if pose_file.exists():
        try:
            data["pose"] = pd.read_csv(pose_file, sep=", ", engine="python")
            print(f"  ✓ Head Pose: {len(data['pose'])} frames")
        except Exception as e:
            print(f"  ❌ Head Pose error: {e}")
    
    # Load gaze
    gaze_file = session_path / file_patterns["gaze"].format(session_id=session_id)
    if gaze_file.exists():
        try:
            data["gaze"] = pd.read_csv(gaze_file, sep=", ", engine="python")
            print(f"  ✓ Gaze: {len(data['gaze'])} frames")
        except Exception as e:
            print(f"  ❌ Gaze error: {e}")
    
    return data

# =============================================================================
# PHQ-8 LABEL LOADING
# =============================================================================

def load_phq8_labels(sessions_config: Dict, 
                     dev_file: Path, 
                     train_file: Path, 
                     test_file: Path) -> pd.DataFrame:
    """
    Load PHQ-8 labels for all sessions
    
    Args:
        sessions_config: Dictionary mapping session IDs to split types
        dev_file: Path to dev split CSV
        train_file: Path to train split CSV
        test_file: Path to test split CSV
        
    Returns:
        DataFrame with PHQ-8 labels for all sessions
    """
    all_labels = []
    
    # Load each split file
    split_files = {
        "dev": dev_file,
        "train": train_file,
        "test": test_file
    }
    
    for session_id, split_type in sessions_config.items():
        split_file = split_files[split_type]
        
        if not split_file.exists():
            print(f"⚠ Split file not found: {split_file}")
            continue
        
        # Load split file
        df_split = pd.read_csv(split_file)
        
        # Find session in split file
        # Column name might be 'Participant_ID', 'participant_id', or similar
        id_col = None
        for col in df_split.columns:
            if 'participant' in col.lower() or 'id' in col.lower():
                id_col = col
                break
        
        if id_col is None:
            print(f"⚠ Could not find ID column in {split_file}")
            continue
        
        # Filter for this session
        session_data = df_split[df_split[id_col].astype(str) == session_id]
        
        if len(session_data) > 0:
            session_data = session_data.copy()
            session_data["session_id"] = session_id
            all_labels.append(session_data)
            print(f"✓ Loaded PHQ-8 for session {session_id}")
        else:
            print(f"⚠ No PHQ-8 data for session {session_id}")
    
    if all_labels:
        return pd.concat(all_labels, ignore_index=True)
    else:
        return pd.DataFrame()

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute regression metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": report["accuracy"],
        "report": report,
        "confusion_matrix": cm
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict, title: str = "Training History", save_path: Optional[Path] = None):
    """
    Plot training history (loss and metrics)
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        title: Main title for the plot
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot loss
    axes[0].plot(history.get("train_loss", []), label="Train Loss", marker='o')
    axes[0].plot(history.get("val_loss", []), label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics (if available)
    if "train_metric" in history:
        axes[1].plot(history["train_metric"], label="Train Metric", marker='o')
        axes[1].plot(history["val_metric"], label="Val Metric", marker='s')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric")
        axes[1].set_title("Training & Validation Metric")
        axes[1].legend()
        axes[1].grid(True)
    elif "val_mae" in history:
        axes[1].plot(history["val_mae"], label="Validation MAE", marker='s', color='orange')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Validation MAE")
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Plot saved: {save_path}")
    
    plt.close()

def plot_predictions_vs_actual(y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                title: str = "Predictions vs Actual",
                                save_path: Optional[Path] = None):
    """
    Scatter plot of predictions vs actual values
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=100)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual PHQ-8 Score")
    plt.ylabel("Predicted PHQ-8 Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics_text = f"MAE: {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nR²: {metrics['R2']:.3f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()

# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience: int = 5, delta: float = 0.001, verbose: bool = True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss: float, model: torch.nn.Module, save_path: Path):
        """
        Check if should stop training
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improved
            save_path: Path to save best model
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, save_path: Path):
        """Save model when validation loss improves"""
        if self.verbose:
            print(f"  ✓ Validation loss improved ({self.val_loss_min:.4f} → {val_loss:.4f}). Saving model...")
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    import re
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    # text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    return text

# =============================================================================
# TIMESTAMP ALIGNMENT
# =============================================================================

def align_visual_to_text(df_transcript: pd.DataFrame, 
                         df_visual: pd.DataFrame,
                         time_col_transcript: str = "start_time",
                         time_col_visual: str = "timestamp") -> pd.DataFrame:
    """
    Align visual features to transcript timestamps
    
    Args:
        df_transcript: Transcript dataframe with timestamps
        df_visual: Visual features dataframe with timestamps
        time_col_transcript: Timestamp column in transcript
        time_col_visual: Timestamp column in visual features
        
    Returns:
        Aligned dataframe
    """
    aligned_data = []
    
    for idx, row in df_transcript.iterrows():
        t_start = row[time_col_transcript]
        t_end = row.get("stop_time", t_start + 5)  # Default 5 second window
        
        # Find visual frames in this time window
        mask = (df_visual[time_col_visual] >= t_start) & (df_visual[time_col_visual] <= t_end)
        visual_segment = df_visual[mask]
        
        if len(visual_segment) > 0:
            # Aggregate visual features over this segment
            visual_agg = visual_segment.mean(numeric_only=True).to_dict()
            
            # Combine with transcript row
            combined = {**row.to_dict(), **visual_agg}
            aligned_data.append(combined)
    
    return pd.DataFrame(aligned_data)

# =============================================================================
# PROGRESS PRINTING
# =============================================================================

def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

def print_step(step_num: int, step_name: str):
    """Print a step header"""
    print(f"\n{'─' * 70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'─' * 70}")

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print_section("UTILITY FUNCTIONS - TESTING")
    
    # Test logging
    logger = setup_logging(Path("test.log"))
    logger.info("Test log message")
    
    # Test metrics
    y_true = np.array([10, 15, 12, 8, 20])
    y_pred = np.array([11, 14, 13, 9, 19])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    print("\nRegression Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.3f}")
    
    print("\n✓ All utility functions loaded successfully")
