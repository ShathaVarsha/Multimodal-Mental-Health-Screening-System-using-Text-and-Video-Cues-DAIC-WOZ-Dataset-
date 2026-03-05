"""
Feature extraction and analysis utilities
Extracts features from participant data for model training
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

def extract_features_for_participants(participant_ids: list, data_dir: str = "data") -> Dict[int, np.ndarray]:
    """Extract multimodal features for multiple participants"""
    features_dict = {}
    
    for pid in participant_ids:
        features = extract_participant_features(pid, data_dir)
        if features is not None:
            features_dict[pid] = features
    
    return features_dict


def extract_participant_features(participant_id: int, data_dir: str = "data") -> Optional[np.ndarray]:
    """
    Extract all available features for a participant
    
    Returns concatenated feature vector from all modalities
    """
    participant_dir = Path(data_dir) / str(participant_id)
    
    if not participant_dir.exists():
        return None
    
    features = []
    
    # Load AUs
    au_file = participant_dir / f"{participant_id}_CLNF_AUs.txt"
    if au_file.exists():
        try:
            au_data = np.loadtxt(au_file, skiprows=0)
            if len(au_data.shape) == 1:
                au_data = au_data.reshape(-1, 1)
            au_features = np.array([np.mean(au_data), np.std(au_data), np.max(au_data), np.min(au_data)])
            features.extend(au_features)
        except:
            pass
    
    # Load HOG
    hog_file = participant_dir / f"{participant_id}_CLNF_hog.txt"
    if hog_file.exists():
        try:
            hog_data = np.loadtxt(hog_file, skiprows=0)
            if len(hog_data.shape) == 1:
                hog_data = hog_data.reshape(-1, 1)
            hog_features = np.array([np.mean(hog_data), np.std(hog_data), np.max(hog_data), np.min(hog_data)])
            features.extend(hog_features)
        except:
            pass
    
    # Load Pose
    pose_file = participant_dir / f"{participant_id}_CLNF_pose.txt"
    if pose_file.exists():
        try:
            pose_data = np.loadtxt(pose_file, skiprows=0)
            if len(pose_data.shape) == 1:
                pose_data = pose_data.reshape(-1, 1)
            pose_features = np.array([np.mean(pose_data), np.std(pose_data), np.max(pose_data), np.min(pose_data)])
            features.extend(pose_features)
        except:
            pass
    
    # Load audio features (COVAREP)
    covarep_file = participant_dir / f"{participant_id}_COVAREP.csv"
    if covarep_file.exists():
        try:
            covarep = pd.read_csv(covarep_file)
            # Extract F0 statistics
            if 'F0final_sma' in covarep.columns or any('F0' in c for c in covarep.columns):
                f0_cols = [c for c in covarep.columns if 'F0' in c]
                for col in f0_cols:
                    if pd.api.types.is_numeric_dtype(covarep[col]):
                        features.extend([covarep[col].mean(), covarep[col].std()])
        except:
            pass
    
    if not features:
        return None
    
    return np.array(features, dtype=np.float32)


def load_depression_labels(split_file: str) -> Dict[int, int]:
    """Load depression labels from split CSV"""
    if not Path(split_file).exists():
        print(f"Warning: {split_file} not found")
        return {}
    
    df = pd.read_csv(split_file)
    
    if 'label' in df.columns and 'participant_id' in df.columns:
        return dict(zip(df['participant_id'], df['label']))
    
    return {}


def create_train_test_datasets(train_split: str, test_split: str,
                              data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create training and testing datasets"""
    
    # Load labels
    train_labels = load_depression_labels(train_split)
    test_labels = load_depression_labels(test_split)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Extract training features
    print(f"Extracting training features from {len(train_labels)} participants...")
    for pid, label in train_labels.items():
        features = extract_participant_features(pid, data_dir)
        if features is not None:
            X_train.append(features)
            y_train.append(label)
    
    # Extract testing features
    print(f"Extracting testing features from {len(test_labels)} participants...")
    for pid, label in test_labels.items():
        features = extract_participant_features(pid, data_dir)
        if features is not None:
            X_test.append(features)
            y_test.append(label)
    
    X_train = np.array(X_train) if X_train else np.array([])
    y_train = np.array(y_train) if y_train else np.array([])
    X_test = np.array(X_test) if X_test else np.array([])
    y_test = np.array(y_test) if y_test else np.array([])
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage
    X_train, y_train, X_test, y_test = create_train_test_datasets(
        "train_split_Depression_AVEC2017.csv",
        "test_split_Depression_AVEC2017.csv"
    )
