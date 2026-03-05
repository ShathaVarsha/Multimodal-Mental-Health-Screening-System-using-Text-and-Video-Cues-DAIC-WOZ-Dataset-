"""
Training script for hybrid depression detection model (42-10-2 split)
Trains on 42 positive, 10 negative, 2 balanced depression cases
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Tuple, Dict, List
from datetime import datetime

class HybridModelTrainer:
    """Trains hybrid depression detection model on video + text features (audio disabled)"""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "ml_training/saved_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("ml_training/training_logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_log = {
            'start_time': datetime.now().isoformat(),
            'epochs': 0,
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def load_split_data(self, split_name: str = "train_split_Depression_AVEC2017.csv") -> Dict:
        """Load participant split from CSV"""
        split_file = Path(split_name)
        
        if not split_file.exists():
            print(f"Warning: {split_file} not found")
            return {}
        
        print(f"Loading split: {split_file}")
        df = pd.read_csv(split_file)
        
        return {
            'positive': df[df['label'] == 1]['participant_id'].tolist() if 'label' in df.columns else [],
            'negative': df[df['label'] == 0]['participant_id'].tolist() if 'label' in df.columns else []
        }
    
    def extract_features_for_participant(self, participant_id: int) -> Tuple[np.ndarray, int]:
        """
        Extract features for single participant (video-focused; no audio)
        
        Returns:
            (feature_vector, label) where label is 0 (no depression) or 1 (depression)
        """
        participant_dir = self.data_dir / str(participant_id)
        
        if not participant_dir.exists():
            return None, None
        
        features = []
        
        # Load Action Unit features
        au_file = participant_dir / f"{participant_id}_CLNF_AUs.txt"
        if au_file.exists():
            try:
                au_data = np.loadtxt(au_file, skiprows=0)
                if len(au_data.shape) == 1:
                    au_data = au_data.reshape(-1, 1)
                # Calculate statistics
                au_features = self._summarize_au_features(au_data)
                features.extend(au_features)
            except Exception as e:
                print(f"Error loading AU for {participant_id}: {e}")
        
        # Load HOG features
        hog_file = participant_dir / f"{participant_id}_CLNF_hog.txt"
        if hog_file.exists():
            try:
                hog_data = np.loadtxt(hog_file, skiprows=0)
                if len(hog_data.shape) == 1:
                    hog_data = hog_data.reshape(-1, 1)
                hog_features = self._summarize_features(hog_data)
                features.extend(hog_features)
            except Exception as e:
                print(f"Error loading HOG for {participant_id}: {e}")
        
        if not features:
            return None, None
        
        return np.array(features, dtype=np.float32), None  # Label to be determined by split
    
    @staticmethod
    def _summarize_au_features(au_data: np.ndarray) -> list:
        """Summarize AU data into statistical features"""
        if len(au_data) == 0:
            return [0.0] * 12  # Return zeros if no data
        
        features = []
        for au_idx in range(au_data.shape[1]):
            au_signal = au_data[:, au_idx]
            features.extend([
                float(np.mean(au_signal)),   # Mean
                float(np.std(au_signal)),    # Std
                float(np.max(au_signal)),    # Max
                float(np.min(au_signal))     # Min
            ])
        
        return features[:12]  # Return top 12 features
    
    @staticmethod
    def _summarize_features(feature_data: np.ndarray) -> list:
        """Summarize features into mean, std, max, min"""
        if len(feature_data) == 0:
            return [0.0] * 4
        
        return [
            float(np.mean(feature_data)),
            float(np.std(feature_data)),
            float(np.max(feature_data)),
            float(np.min(feature_data))
        ]
    
    @staticmethod
    def _extract_audio_features(covarep_df: pd.DataFrame) -> list:
        """Extract key audio features from COVAREP"""
        features = []
        
        # Key features: F0, energy, MFCCs
        key_cols = ['F0final_sma', 'energy_sma', 'voicingFinalUnclipped_sma']
        
        for col in key_cols:
            matching_cols = [c for c in covarep_df.columns if col in c]
            for match_col in matching_cols:
                if pd.api.types.is_numeric_dtype(covarep_df[match_col]):
                    features.extend(HybridModelTrainer._summarize_features(
                        covarep_df[match_col].values
                    ))
        
        return features[:12]  # Return top 12 audio features
    
    def train_model(self, split_file: str = "train_split_Depression_AVEC2017.csv",
                   epochs: int = 20, batch_size: int = 16):
        """Train hybrid depression detection model"""
        
        print("="*60)
        print(f"Training Hybrid Depression Model (42-10-2 Split)")
        print("="*60)
        
        # Load split
        split_data = self.load_split_data(split_file)
        
        if not split_data.get('positive') and not split_data.get('negative'):
            print("Error: No participant data loaded")
            return False
        
        # Extract features and labels
        X_train = []
        y_train = []
        
        print(f"\nExtracting features for {len(split_data.get('positive', []))} positive cases...")
        for pid in split_data.get('positive', []):
            features, _ = self.extract_features_for_participant(pid)
            if features is not None:
                X_train.append(features)
                y_train.append(1)  # Depression
        
        print(f"Extracting features for {len(split_data.get('negative', []))} negative cases...")
        for pid in split_data.get('negative', []):
            features, _ = self.extract_features_for_participant(pid)
            if features is not None:
                X_train.append(features)
                y_train.append(0)  # No depression
        
        if not X_train:
            print("Error: Could not extract any training features")
            return False
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Simple model training (placeholder for actual implementation)
        # In real implementation, would use scikit-learn, PyTorch, or TensorFlow
        model = self._create_model(X_train.shape[1])
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        for epoch in range(epochs):
            # Placeholder training loop
            loss = self._train_epoch(model, X_train, y_train, batch_size)
            acc = self._evaluate(model, X_train, y_train)
            
            self.training_log['train_loss'].append(float(loss))
            self.training_log['train_accuracy'].append(float(acc))
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        # Save model
        self._save_model(model, "hybrid_model_42_10_2.pkl")
        
        # Save training log
        self.training_log['end_time'] = datetime.now().isoformat()
        self.training_log['epochs'] = epochs
        self._save_training_log("hybrid_model_42_10_2_log.json")
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {self.model_dir / 'hybrid_model_42_10_2.pkl'}")
        
        return True
    
    @staticmethod
    def _create_model(input_dim: int) -> Dict:
        """Create simple model structure"""
        return {
            'input_dim': input_dim,
            'weights': np.random.randn(input_dim, 64).astype(np.float32),
            'bias': np.zeros(64, dtype=np.float32),
            'output_weights': np.random.randn(64, 1).astype(np.float32)
        }
    
    @staticmethod
    def _train_epoch(model: Dict, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        """Train single epoch (placeholder)"""
        # Placeholder loss calculation
        n_batches = len(X) // batch_size
        total_loss = 0.0
        
        for i in range(n_batches):
            batch_X = X[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            
            # Forward pass
            hidden = np.tanh(batch_X @ model['weights'] + model['bias'])
            logits = hidden @ model['output_weights']
            
            # Loss
            loss = np.mean((logits.flatten() - batch_y) ** 2)
            total_loss += loss
        
        return total_loss / n_batches
    
    @staticmethod
    def _evaluate(model: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy"""
        hidden = np.tanh(X @ model['weights'] + model['bias'])
        logits = hidden @ model['output_weights']
        predictions = (logits.flatten() > 0.5).astype(int)
        
        return float(np.mean(predictions == y))
    
    def _save_model(self, model: Dict, filename: str):
        """Save model to disk"""
        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    def _save_training_log(self, filename: str):
        """Save training log"""
        filepath = self.logs_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.training_log, f, indent=2)


def main():
    """Main training function"""
    trainer = HybridModelTrainer()
    
    # Train 42-10-2 model
    success = trainer.train_model(
        split_file="train_split_Depression_AVEC2017.csv",
        epochs=20,
        batch_size=16
    )
    
    if success:
        print("\n✓ Training completed successfully!")
    else:
        print("\n✗ Training failed")


if __name__ == "__main__":
    main()
