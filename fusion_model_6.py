"""
=================================================================
STEP 6: MODEL 3 - MULTIMODAL FUSION NETWORK
=================================================================
Train fusion network combining text and visual features

This script:
1. Loads text embeddings and visual features
2. Defines multimodal fusion architecture
3. Trains with early stopping (min 15 epochs)
4. Uses LOOCV for evaluation
5. Saves best model checkpoints

Inputs:
  - outputs/text_embeddings.pkl (from Step 2)
  - outputs/session_aggregates.pkl (from Step 2)

Outputs:
  - models/model3_fusion.pth (trained fusion network)
  - outputs/model3_predictions.pkl (predictions)
  - outputs/model3_training_history.json (loss curves)
  - outputs/model3_predictions.png (scatter plot)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from utils import *

# Check if torch is installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch not installed. Install with: pip install torch")
    TORCH_AVAILABLE = False

# =============================================================================
# FUSION NETWORK ARCHITECTURE
# =============================================================================

class MultimodalFusionNet(nn.Module):
    """Fusion network combining text and visual features"""
    
    def __init__(self, text_dim, visual_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Text branch (removed BatchNorm for small batch compatibility)
        self.text_fc1 = nn.Linear(text_dim, hidden_dims[0])
        self.text_dropout1 = nn.Dropout(0.3)
        
        # Visual branch (removed BatchNorm for small batch compatibility)
        self.visual_fc1 = nn.Linear(visual_dim, hidden_dims[0])
        self.visual_dropout1 = nn.Dropout(0.3)
        
        # Fusion layers (removed BatchNorm for small batch compatibility)
        fusion_input_dim = hidden_dims[0] * 2
        self.fusion_fc1 = nn.Linear(fusion_input_dim, hidden_dims[1])
        self.fusion_dropout1 = nn.Dropout(0.2)
        
        self.fusion_fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fusion_dropout2 = nn.Dropout(0.1)
        
        # Output layer
        self.output = nn.Linear(hidden_dims[2], 1)
    
    def forward(self, text_features, visual_features):
        # Text branch
        text_out = F.relu(self.text_fc1(text_features))
        text_out = self.text_dropout1(text_out)
        
        # Visual branch
        visual_out = F.relu(self.visual_fc1(visual_features))
        visual_out = self.visual_dropout1(visual_out)
        
        # Fusion
        fused = torch.cat([text_out, visual_out], dim=1)
        
        fused = F.relu(self.fusion_fc1(fused))
        fused = self.fusion_dropout1(fused)
        
        fused = F.relu(self.fusion_fc2(fused))
        fused = self.fusion_dropout2(fused)
        
        # Output
        output = self.output(fused)
        
        return output.squeeze()

# =============================================================================
# DATASET CLASS
# =============================================================================

class MultimodalDataset(Dataset):
    """Dataset for multimodal fusion"""
    
    def __init__(self, text_embeddings, visual_features, labels):
        self.text = torch.FloatTensor(text_embeddings)
        self.visual = torch.FloatTensor(visual_features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text[idx], self.visual[idx], self.labels[idx]

# =============================================================================
# FUSION TRAINER CLASS
# =============================================================================

class FusionNetworkTrainer:
    """Trainer for fusion network"""
    
    def __init__(self):
        """Initialize trainer"""
        self.logger = setup_logging(LOG_FILE, LOG_LEVEL)
        print_section("STEP 6: MODEL 3 - MULTIMODAL FUSION NETWORK")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.session_data = None
        self.text_embeddings = None
        
        self.visual_feature_cols = []
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": []
        }
    
    def load_data(self):
        """Load text embeddings and visual features"""
        print_step(1, "Loading Data")
        
        # Load session aggregates
        session_file = OUTPUTS_DIR / "session_aggregates.pkl"
        if not session_file.exists():
            print(f"❌ Session aggregates not found: {session_file}")
            return False
        
        self.session_data = load_pickle(session_file)
        
        # Load text embeddings
        text_file = OUTPUTS_DIR / "text_embeddings.pkl"
        if not text_file.exists():
            print(f"❌ Text embeddings not found: {text_file}")
            return False
        
        text_data = load_pickle(text_file)
        
        # Check if it's a dict (from step2) or DataFrame
        if isinstance(text_data, dict):
            # Extract from dict format
            embeddings = text_data["embeddings"]
            session_ids = text_data["session_ids"]
            
            # Aggregate embeddings by session
            session_embeddings = {}
            for i, (session_id, embedding) in enumerate(zip(session_ids, embeddings)):
                if session_id not in session_embeddings:
                    session_embeddings[session_id] = []
                session_embeddings[session_id].append(embedding)
        else:
            # DataFrame format (legacy)
            session_embeddings = {}
            for idx, row in text_data.iterrows():
                session_id = row["session_id"]
                embedding = row["text_embedding"]
                
                if session_id not in session_embeddings:
                    session_embeddings[session_id] = []
                session_embeddings[session_id].append(embedding)
        
        # Average embeddings per session
        avg_embeddings = {}
        for session_id, embeddings in session_embeddings.items():
            avg_embeddings[session_id] = np.mean(embeddings, axis=0)
        
        # Add to session data
        self.session_data["text_embedding"] = self.session_data["session_id"].map(avg_embeddings)
        
        # Remove sessions without embeddings or PHQ scores
        self.session_data = self.session_data.dropna(subset=["text_embedding", "phq8_score"])
        
        print(f"✓ Data loaded")
        print(f"  Sessions: {len(self.session_data)}")
        if len(avg_embeddings) > 0:
            first_key = list(avg_embeddings.keys())[0]
            print(f"  Text embedding dim: {len(avg_embeddings[first_key])}")
        else:
            print(f"  ❌ No embeddings found")
        
        return True
    
    def prepare_features(self):
        """Prepare feature matrices"""
        print_step(2, "Preparing Features")
        
        # Get visual features
        au_cols = [c for c in self.session_data.columns if c.startswith("au_")]
        pose_cols = [c for c in self.session_data.columns if c.startswith("pose_")]
        gaze_cols = [c for c in self.session_data.columns if c.startswith("gaze_")]
        
        self.visual_feature_cols = au_cols + pose_cols + gaze_cols
        
        print(f"  Feature dimensions:")
        print(f"    Text: 768")
        print(f"    Visual: {len(self.visual_feature_cols)}")
        
        # Extract features
        X_text = np.stack(self.session_data["text_embedding"].values)
        X_visual = self.session_data[self.visual_feature_cols].fillna(0).values
        y = self.session_data["phq8_score"].values
        
        self.X_text = X_text
        self.X_visual = X_visual
        self.y = y
        
        print(f"\n  Data shapes:")
        print(f"    Text: {X_text.shape}")
        print(f"    Visual: {X_visual.shape}")
        print(f"    Labels: {y.shape}")
        
        return True
    
    def build_model(self):
        """Build fusion network"""
        print_step(3, "Building Fusion Network")
        
        text_dim = self.X_text.shape[1]
        visual_dim = self.X_visual.shape[1]
        
        # Get hidden dims from config (use [256, 128, 64] as default)
        hidden_dims = MODEL3_CONFIG.get("hidden_dims", [256, 128, 64])
        
        self.model = MultimodalFusionNet(
            text_dim=text_dim,
            visual_dim=visual_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Model architecture:")
        print(f"    Input: Text({text_dim}) + Visual({visual_dim})")
        print(f"    Hidden: {hidden_dims}")
        print(f"    Output: 1 (PHQ-8 score)")
        print(f"\n  Parameters:")
        print(f"    Total: {total_params:,}")
        print(f"    Trainable: {trainable_params:,}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=MODEL3_CONFIG["learning_rate"],
            weight_decay=1e-5
        )
        
        print(f"\n  Optimizer: Adam")
        print(f"    Learning rate: {MODEL3_CONFIG['learning_rate']}")
        print(f"    Weight decay: 1e-5")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for text, visual, labels in dataloader:
            text = text.to(self.device)
            visual = visual.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(text, visual)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for text, visual, labels in dataloader:
                text = text.to(self.device)
                visual = visual.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(text, visual)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Handle both scalar and array outputs
                outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Convert to 1D arrays if scalar
                if outputs_np.ndim == 0:
                    outputs_np = np.array([outputs_np])
                if labels_np.ndim == 0:
                    labels_np = np.array([labels_np])
                
                predictions.extend(outputs_np)
                actuals.extend(labels_np)
        
        avg_loss = total_loss / len(dataloader)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        return avg_loss, mae, predictions, actuals
    
    def train_with_loocv(self):
        """Train using Leave-One-Out Cross-Validation"""
        print_step(4, f"Training with LOOCV (min {MODEL3_CONFIG.get('epochs', MODEL3_CONFIG.get('num_epochs', 300))} epochs)")
        
        from sklearn.model_selection import LeaveOneOut
        
        loo = LeaveOneOut()
        
        fold_predictions = []
        fold_actuals = []
        fold_metrics = []
        
        n_samples = len(self.y)
        
        for fold_idx, (train_idx, test_idx) in enumerate(loo.split(self.X_text)):
            print(f"\n  Fold {fold_idx+1}/{n_samples}")
            
            # Split data
            X_text_train = self.X_text[train_idx]
            X_text_test = self.X_text[test_idx]
            X_visual_train = self.X_visual[train_idx]
            X_visual_test = self.X_visual[test_idx]
            y_train = self.y[train_idx]
            y_test = self.y[test_idx]
            
            # Create datasets
            train_dataset = MultimodalDataset(X_text_train, X_visual_train, y_train)
            test_dataset = MultimodalDataset(X_text_test, X_visual_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=MODEL3_CONFIG["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            # Reset model
            self.build_model()
            
            # Early stopping
            early_stopping = EarlyStopping(
                patience=MODEL3_CONFIG["early_stopping_patience"],
                delta=0.001,
                verbose=False
            )
            
            # Train
            best_loss = float('inf')
            best_pred = None
            
            for epoch in range(MODEL3_CONFIG.get("epochs", MODEL3_CONFIG.get("num_epochs", 300))):
                train_loss = self.train_epoch(train_loader)
                val_loss, val_mae, val_preds, val_actuals = self.validate(test_loader)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_pred = val_preds[0]
                
                # Print every 5 epochs
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{MODEL3_CONFIG.get('epochs', MODEL3_CONFIG.get('num_epochs', 300))}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                
                # Early stopping check (save to temp path for LOOCV fold)
                fold_save_path = CHECKPOINTS_DIR / f"fold_{fold_idx+1}_temp.pth"
                early_stopping(val_loss, self.model, fold_save_path)
                if early_stopping.early_stop and epoch >= 15:  # Minimum 15 epochs
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            # Store results
            fold_predictions.append(best_pred)
            fold_actuals.append(y_test[0])
            
            session_id = self.session_data.iloc[test_idx[0]]["session_id"]
            error = abs(best_pred - y_test[0])
            
            fold_metrics.append({
                "fold": fold_idx + 1,
                "session_id": int(session_id),
                "actual": float(y_test[0]),
                "predicted": float(best_pred),
                "error": float(error)
            })
            
            print(f"    Result: Actual={y_test[0]:.1f}, Pred={best_pred:.1f}, Error={error:.2f}")
        
        # Overall metrics
        fold_predictions = np.array(fold_predictions)
        fold_actuals = np.array(fold_actuals)
        
        mae = np.mean(np.abs(fold_predictions - fold_actuals))
        rmse = np.sqrt(np.mean((fold_predictions - fold_actuals)**2))
        
        from sklearn.metrics import r2_score
        r2 = r2_score(fold_actuals, fold_predictions)
        
        print(f"\n  ✓ LOOCV Complete")
        print(f"    MAE: {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    R²: {r2:.3f}")
        
        self.loocv_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "fold_metrics": fold_metrics
        }
        
        self.loocv_predictions = fold_predictions
        self.loocv_actuals = fold_actuals
    
    def train_final_model(self):
        """Train final model on all data"""
        print_step(5, f"Training Final Model (min {MODEL3_CONFIG.get('epochs', MODEL3_CONFIG.get('num_epochs', 300))} epochs)")
        
        # Create dataset
        dataset = MultimodalDataset(self.X_text, self.X_visual, self.y)
        dataloader = DataLoader(dataset, batch_size=MODEL3_CONFIG["batch_size"], shuffle=True)
        
        # Reset model
        self.build_model()
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=MODEL3_CONFIG["early_stopping_patience"],
            delta=0.001,
            verbose=True
        )
        
        # Training loop
        print(f"\n  Training...")
        
        for epoch in range(MODEL3_CONFIG.get("epochs", MODEL3_CONFIG.get("num_epochs", 300))):
            train_loss = self.train_epoch(dataloader)
            val_loss, val_mae, _, _ = self.validate(dataloader)
            
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_mae"].append(val_mae)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{MODEL3_CONFIG.get('epochs', MODEL3_CONFIG.get('num_epochs', 300))}: Loss={train_loss:.4f}, MAE={val_mae:.2f}")
            
            # Early stopping
            early_stopping(val_loss, self.model, MODEL3_CONFIG["save_path"])
            if early_stopping.early_stop and epoch >= 15:  # Minimum 15 epochs
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n  ✓ Training complete")
        print(f"    Final loss: {train_loss:.4f}")
        print(f"    Final MAE: {val_mae:.2f}")
    
    def save_model(self):
        """Save trained model"""
        print_step(6, "Saving Model")
        
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "text_dim": self.X_text.shape[1],
            "visual_dim": self.X_visual.shape[1],
            "visual_feature_cols": self.visual_feature_cols,
            "config": MODEL3_CONFIG,
            "loocv_metrics": self.loocv_metrics
        }
        
        # Save full checkpoint with metadata
        torch.save(model_data, MODEL3_CONFIG["save_path"])
        print(f"✓ Model saved: {MODEL3_CONFIG['save_path']}")
        
        # Save predictions
        predictions_data = {
            "session_ids": self.session_data["session_id"].tolist(),
            "actuals": self.loocv_actuals.tolist(),
            "predictions": self.loocv_predictions.tolist()
        }
        save_pickle(predictions_data, OUTPUTS_DIR / "model3_predictions.pkl")
        
        # Save training history
        save_json(self.training_history, OUTPUTS_DIR / "model3_training_history.json")
        save_json(self.loocv_metrics, OUTPUTS_DIR / "model3_evaluation.json")
        
        print(f"\n✓ Model saved successfully!")
        print(f"  → Next step: python report_generator_7.py")
    
    def visualize_results(self):
        """Visualize training and results"""
        print_step(7, "Visualizing Results")
        
        # Plot training history
        plot_training_history(
            self.training_history,
            title="Model 3: Fusion Network Training",
            save_path=OUTPUTS_DIR / "model3_training.png"
        )
        
        # Plot predictions
        plot_predictions_vs_actual(
            self.loocv_actuals,
            self.loocv_predictions,
            title="Model 3: Fusion Network Predictions (LOOCV)",
            save_path=OUTPUTS_DIR / "model3_predictions.png"
        )

# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_model():
    """Test the trained model"""
    print_section("TESTING MODEL 3")
    
    # Load model checkpoint (weights_only=False since we save metadata too)
    checkpoint = torch.load(MODEL3_CONFIG["save_path"], weights_only=False)
    
    text_dim = checkpoint["text_dim"]
    visual_dim = checkpoint["visual_dim"]
    
    model = MultimodalFusionNet(
        text_dim=text_dim,
        visual_dim=visual_dim,
        hidden_dims=MODEL3_CONFIG.get("hidden_dims", [256, 128, 64])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Text dim: {text_dim}")
    print(f"  Visual dim: {visual_dim}")
    
    # Test prediction
    dummy_text = torch.randn(1, text_dim)
    dummy_visual = torch.randn(1, visual_dim)
    
    with torch.no_grad():
        prediction = model(dummy_text, dummy_visual)
    
    print(f"\nTest prediction on random features:")
    print(f"  Predicted PHQ-8 score: {prediction.item():.2f}")
    
    print("\n✓ Model test complete")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install:")
        print("   pip install torch")
        return
    
    # Initialize
    trainer = FusionNetworkTrainer()
    
    # Load data
    if not trainer.load_data():
        return
    
    # Prepare features
    if not trainer.prepare_features():
        return
    
    # Build model
    trainer.build_model()
    
    # Train with LOOCV
    trainer.train_with_loocv()
    
    # Train final model
    trainer.train_final_model()
    
    # Visualize
    trainer.visualize_results()
    
    # Save
    trainer.save_model()
    
    # Test
    test_model()
    
    print_section("MODEL 3 TRAINING COMPLETE")

if __name__ == "__main__":
    main()
