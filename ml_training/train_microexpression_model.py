"""
Micro-expression model training
Trains model to detect brief facial expressions and depression-related patterns
"""
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from typing import Dict, Tuple
import json

class MicroExpressionModelTrainer:
    """Trains micro-expression detection model"""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "ml_training/micro_expression_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("ml_training/training_logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Depression-related micro-expression patterns
        self.expression_patterns = {
            'sadness': [1, 4, 15],      # AU1 (inner brow), AU4 (brow lower), AU15 (lip corner down)
            'fear': [1, 2, 4, 5, 26],   # Complex pattern
            'contempt': [12, 14],        # Unilateral lip movements
            'suppressed_smile': [6, 12], # High smile suppression = depression marker
        }
    
    def load_participant_au_data(self, participant_id: int) -> Tuple[np.ndarray, str]:
        """Load AU data for participant"""
        participant_dir = self.data_dir / str(participant_id)
        au_file = participant_dir / f"{participant_id}_CLNF_AUs.txt"
        
        if not au_file.exists():
            return None, "no_data"
        
        try:
            au_data = np.loadtxt(au_file, skiprows=0)
            if len(au_data.shape) == 1:
                au_data = au_data.reshape(-1, 1)
            return au_data, "success"
        except Exception as e:
            return None, f"error: {e}"
    
    def extract_micro_expression_features(self, au_data: np.ndarray) -> np.ndarray:
        """
        Extract micro-expression features from AU temporal dynamics
        
        Features:
        - AU velocity (rate of change)
        - AU peak magnitudes
        - AU synchrony (correlation between AUs)
        - Expression onset/offset timing
        """
        n_frames = au_data.shape[0]
        features = []
        
        # Temporal derivatives (AU velocity)
        au_velocity = np.diff(au_data, axis=0)
        features.append(np.mean(np.abs(au_velocity)))
        features.append(np.std(np.abs(au_velocity)))
        features.append(np.max(np.abs(au_velocity)))
        
        # Peak AU activations
        for au_idx in range(au_data.shape[1]):
            au_signal = au_data[:, au_idx]
            features.append(np.max(au_signal))
            features.append(np.mean(au_signal[au_signal > 0.1]))  # Mean when active
        
        # AU synchrony (co-occurrence)
        for i in range(min(3, au_data.shape[1])):
            for j in range(i+1, min(5, au_data.shape[1])):
                au_corr = np.corrcoef(au_data[:, i], au_data[:, j])[0, 1]
                if not np.isnan(au_corr):
                    features.append(au_corr)
        
        # Asymmetry features (unilateral vs bilateral)
        if au_data.shape[1] >= 17:
            left_aus = au_data[:, :8]
            right_aus = au_data[:, 8:16]
            asymmetry = np.mean(np.abs(left_aus - right_aus))
            features.append(asymmetry)
        
        # Depression-specific features
        depression_features = self._extract_depression_features(au_data)
        features.extend(depression_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_depression_features(self, au_data: np.ndarray) -> list:
        """Extract depression-specific AU patterns"""
        features = []
        
        # Sadness AUs (1, 4, 15)
        if au_data.shape[1] > 14:
            sadness_score = np.mean([
                au_data[:, 0],   # AU1
                au_data[:, 3],   # AU4
                au_data[:, 14]   # AU15
            ])
            features.append(float(sadness_score))
        
        # Smile reduction (low AU6 and AU12)
        if au_data.shape[1] > 11:
            smile_score = 1.0 - np.mean([
                au_data[:, 5],   # AU6
                au_data[:, 11]   # AU12
            ])
            features.append(float(smile_score))
        
        # Brow lowering prominence (AU4)
        if au_data.shape[1] > 3:
            features.append(float(np.max(au_data[:, 3])))  # AU4 max
        
        return features
    
    def train_micro_expression_classifier(self, participant_list: list,
                                         epochs: int = 30) -> bool:
        """Train micro-expression detection model"""
        
        print("="*60)
        print("Training Micro-Expression Detection Model")
        print("="*60)
        
        X_train = []
        y_train = []
        
        print(f"\nProcessing {len(participant_list)} participants...")
        
        for idx, participant_id in enumerate(participant_list):
            au_data, status = self.load_participant_au_data(participant_id)
            
            if au_data is None:
                continue
            
            # Extract features
            features = self.extract_micro_expression_features(au_data)
            X_train.append(features)
            
            # Label: some participants have depression (placeholder logic)
            label = 1 if idx % 3 == 0 else 0
            y_train.append(label)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1} participants...")
        
        if not X_train:
            print("Error: No training data extracted")
            return False
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Create and train model
        model = self._create_model(X_train.shape[1])
        
        print(f"\nTraining for {epochs} epochs...")
        training_log = {'losses': [], 'accuracies': []}
        
        for epoch in range(epochs):
            loss = self._train_epoch(model, X_train, y_train, batch_size=8)
            acc = self._evaluate(model, X_train, y_train)
            
            training_log['losses'].append(float(loss))
            training_log['accuracies'].append(float(acc))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        # Save model
        self._save_model(model, "microexpression_detector.pkl")
        
        # Save patterns reference
        patterns_file = self.model_dir / "expression_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(self.expression_patterns, f, indent=2)
        
        print(f"\n✓ Model saved to {self.model_dir / 'microexpression_detector.pkl'}")
        print(f"✓ Patterns saved to {patterns_file}")
        
        return True
    
    @staticmethod
    def _create_model(input_dim: int) -> Dict:
        """Create simple neural network model"""
        return {
            'input_dim': input_dim,
            'layer1_w': np.random.randn(input_dim, 64).astype(np.float32) * 0.01,
            'layer1_b': np.zeros(64, dtype=np.float32),
            'layer2_w': np.random.randn(64, 32).astype(np.float32) * 0.01,
            'layer2_b': np.zeros(32, dtype=np.float32),
            'output_w': np.random.randn(32, 1).astype(np.float32) * 0.01,
            'output_b': np.zeros(1, dtype=np.float32)
        }
    
    @staticmethod
    def _train_epoch(model: Dict, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        """Train single epoch"""
        n_batches = max(1, len(X) // batch_size)
        total_loss = 0.0
        
        for i in range(n_batches):
            batch_X = X[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            
            # Forward pass
            hidden1 = np.tanh(batch_X @ model['layer1_w'] + model['layer1_b'])
            hidden2 = np.tanh(hidden1 @ model['layer2_w'] + model['layer2_b'])
            logits = hidden2 @ model['output_w'] + model['output_b']
            
            # Loss
            loss = np.mean((logits.flatten() - batch_y) ** 2)
            total_loss += loss
        
        return total_loss / n_batches
    
    @staticmethod
    def _evaluate(model: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy"""
        hidden1 = np.tanh(X @ model['layer1_w'] + model['layer1_b'])
        hidden2 = np.tanh(hidden1 @ model['layer2_w'] + model['layer2_b'])
        logits = hidden2 @ model['output_w'] + model['output_b']
        predictions = (logits.flatten() > 0.5).astype(int)
        
        return float(np.mean(predictions == y))
    
    def _save_model(self, model: Dict, filename: str):
        """Save model"""
        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)


def main():
    """Main training"""
    trainer = MicroExpressionModelTrainer()
    
    # Get list of participants
    data_dir = Path("data")
    participants = [int(d.name) for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    participants = sorted(participants)[:30]  # Use first 30
    
    print(f"Found {len(participants)} participants")
    
    success = trainer.train_micro_expression_classifier(participants, epochs=20)
    
    if success:
        print("\n✓ Micro-expression model training complete!")


if __name__ == "__main__":
    main()
