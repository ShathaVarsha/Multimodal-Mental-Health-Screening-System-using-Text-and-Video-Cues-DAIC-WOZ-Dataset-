"""
=================================================================
STEP 5: MODEL 2B - VISUAL CLASSIFIER (SVM)
=================================================================
Train SVM classifier on visual features to predict depression

This script:
1. Loads engineered features from Step 2
2. Extracts visual features (AUs, pose, gaze)
3. Trains SVM with cross-validation
4. Hyperparameter tuning (C, gamma, kernel)
5. Evaluates on each session (LOOCV)
6. Saves trained model

Inputs:
  - outputs/engineered_features.pkl (from Step 2)
  - outputs/session_aggregates.pkl (from Step 2)

Outputs:
  - models/model2b_visual_svm.pkl (trained SVM + scaler)
  - outputs/model2b_predictions.pkl (predictions)
  - outputs/model2b_evaluation.json (metrics)
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

# Check if sklearn is installed
try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("❌ scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# =============================================================================
# MODEL 2B: VISUAL CLASSIFIER CLASS
# =============================================================================

class VisualClassifierSVM:
    """SVM-based visual classifier for depression detection"""
    
    def __init__(self):
        """Initialize visual classifier"""
        self.logger = setup_logging(LOG_FILE, LOG_LEVEL)
        print_section("STEP 5: MODEL 2B - VISUAL CLASSIFIER (SVM)")
        
        self.session_data = None
        self.visual_feature_cols = []
        
        self.scaler = StandardScaler()
        self.model = None
        
        self.X_train = None
        self.y_train = None
        
        self.predictions = []
        self.metrics = {}
    
    def load_session_data(self):
        """Load session-aggregated features"""
        print_step(1, "Loading Session Data")
        
        session_file = OUTPUTS_DIR / "session_aggregates.pkl"
        
        if not session_file.exists():
            print(f"❌ Session aggregates not found: {session_file}")
            print(f"   Please run: python feature_engineering_2.py")
            return False
        
        self.session_data = load_pickle(session_file)
        
        print(f"✓ Loaded session data")
        print(f"  Total sessions: {len(self.session_data)}")
        print(f"  Columns: {len(self.session_data.columns)}")
        
        # Check for PHQ-8 scores
        if "phq8_score" not in self.session_data.columns:
            print(f"❌ No PHQ-8 scores found in session data")
            return False
        
        # Remove sessions without PHQ-8 scores
        self.session_data = self.session_data.dropna(subset=["phq8_score"])
        
        print(f"  Sessions with PHQ-8 scores: {len(self.session_data)}")
        
        if len(self.session_data) < 2:
            print(f"❌ Need at least 2 sessions for training")
            return False
        
        return True
    
    def prepare_visual_features(self):
        """Prepare visual feature matrix"""
        print_step(2, "Preparing Visual Features")
        
        # Get all visual feature columns
        au_cols = [c for c in self.session_data.columns if c.startswith("au_")]
        pose_cols = [c for c in self.session_data.columns if c.startswith("pose_")]
        gaze_cols = [c for c in self.session_data.columns if c.startswith("gaze_")]
        
        self.visual_feature_cols = au_cols + pose_cols + gaze_cols
        
        print(f"  Visual features found:")
        print(f"    - Action Units: {len(au_cols)}")
        print(f"    - Head Pose: {len(pose_cols)}")
        print(f"    - Gaze: {len(gaze_cols)}")
        print(f"    - Total: {len(self.visual_feature_cols)}")
        
        if len(self.visual_feature_cols) == 0:
            print(f"❌ No visual features found")
            return False
        
        # Extract feature matrix
        X = self.session_data[self.visual_feature_cols].fillna(0).values
        y = self.session_data["phq8_score"].values
        
        print(f"\n  Feature matrix shape: {X.shape}")
        print(f"  Target range: [{y.min():.1f}, {y.max():.1f}]")
        print(f"  Target mean: {y.mean():.1f} ± {y.std():.1f}")
        
        self.X_train = X
        self.y_train = y
        
        return True
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print_step(3, "Scaling Features")
        
        # Fit scaler on all data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        print(f"  ✓ Features scaled")
        print(f"    Mean: {self.X_train_scaled.mean(axis=0).mean():.3f}")
        print(f"    Std: {self.X_train_scaled.std(axis=0).mean():.3f}")
    
    def hyperparameter_tuning(self):
        """Grid search for best hyperparameters"""
        print_step(4, "Hyperparameter Tuning")
        
        # Parameter grid
        param_grid = {
            'C': MODEL2B_CONFIG["C_range"],
            'gamma': ['auto', 'scale', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        print(f"  Testing {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])} combinations...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            SVR(),
            param_grid,
            cv=min(3, len(self.X_train_scaled)),  # 3-fold or less if not enough samples
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\n  ✓ Best parameters found:")
        print(f"    C: {grid_search.best_params_['C']}")
        print(f"    Gamma: {grid_search.best_params_['gamma']}")
        print(f"    Kernel: {grid_search.best_params_['kernel']}")
        print(f"    MAE: {-grid_search.best_score_:.2f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def train_with_loocv(self):
        """Train and evaluate using Leave-One-Out Cross-Validation"""
        print_step(5, "Training with Leave-One-Out Cross-Validation")
        
        loo = LeaveOneOut()
        
        predictions = []
        actuals = []
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(loo.split(self.X_train_scaled)):
            # Split data
            X_train_fold = self.X_train_scaled[train_idx]
            X_test_fold = self.X_train_scaled[test_idx]
            y_train_fold = self.y_train[train_idx]
            y_test_fold = self.y_train[test_idx]
            
            # Train model
            model_fold = SVR(
                C=self.model.C,
                gamma=self.model.gamma,
                kernel=self.model.kernel
            )
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred = model_fold.predict(X_test_fold)
            
            predictions.append(y_pred[0])
            actuals.append(y_test_fold[0])
            
            # Fold metrics
            mae = abs(y_pred[0] - y_test_fold[0])
            fold_metrics.append({
                "fold": fold_idx + 1,
                "actual": float(y_test_fold[0]),
                "predicted": float(y_pred[0]),
                "error": float(mae)
            })
            
            session_id = self.session_data.iloc[test_idx[0]]["session_id"]
            print(f"  Fold {fold_idx+1}/{len(self.y_train)}: Session {session_id} - Actual={y_test_fold[0]:.1f}, Pred={y_pred[0]:.1f}, Error={mae:.2f}")
        
        # Overall metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        print(f"\n  ✓ LOOCV Complete")
        print(f"    MAE: {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    R²: {r2:.3f}")
        
        self.predictions = predictions
        self.actuals = actuals
        self.metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "fold_metrics": fold_metrics
        }
        
        return self.metrics
    
    def train_final_model(self):
        """Train final model on all data"""
        print_step(6, "Training Final Model on All Data")
        
        # Train on all data
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Predict on all data
        y_pred_all = self.model.predict(self.X_train_scaled)
        
        # Metrics
        mae = mean_absolute_error(self.y_train, y_pred_all)
        rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_all))
        r2 = r2_score(self.y_train, y_pred_all)
        
        print(f"  ✓ Final model trained")
        print(f"    Training MAE: {mae:.2f}")
        print(f"    Training RMSE: {rmse:.2f}")
        print(f"    Training R²: {r2:.3f}")
        
        # Feature importance (for linear kernel)
        if self.model.kernel == 'linear':
            feature_importance = np.abs(self.model.coef_[0])
            top_features_idx = np.argsort(feature_importance)[-10:]
            
            print(f"\n  Top 10 Most Important Features:")
            for idx in reversed(top_features_idx):
                print(f"    {self.visual_feature_cols[idx]}: {feature_importance[idx]:.3f}")
    
    def save_model(self):
        """Save trained model and scaler"""
        print_step(7, "Saving Model")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.visual_feature_cols,
            "metrics": self.metrics,
            "config": MODEL2B_CONFIG
        }
        
        save_pickle(model_data, MODEL2B_CONFIG["save_path"])
        
        # Save predictions
        predictions_data = {
            "session_ids": self.session_data["session_id"].tolist(),
            "actuals": self.actuals.tolist() if hasattr(self, 'actuals') else [],
            "predictions": self.predictions.tolist() if hasattr(self, 'predictions') else []
        }
        save_pickle(predictions_data, OUTPUTS_DIR / "model2b_predictions.pkl")
        
        # Save metrics as JSON
        save_json(self.metrics, OUTPUTS_DIR / "model2b_evaluation.json")
        
        print(f"\n✓ Model saved successfully!")
        print(f"  → Next step: python step6_model3_fusion.py")
    
    def visualize_results(self):
        """Visualize predictions vs actuals"""
        print_step(8, "Visualizing Results")
        
        if not hasattr(self, 'actuals') or not hasattr(self, 'predictions'):
            print("  ⚠ No predictions available")
            return
        
        # Plot
        plot_predictions_vs_actual(
            self.actuals,
            self.predictions,
            title="Model 2b: Visual Classifier Predictions",
            save_path=OUTPUTS_DIR / "model2b_predictions.png"
        )

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_model():
    """Test the trained model"""
    print_section("TESTING MODEL 2B")
    
    # Load model
    model_data = load_pickle(MODEL2B_CONFIG["save_path"])
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]
    
    print(f"Model loaded successfully")
    print(f"  Kernel: {model.kernel}")
    print(f"  C: {model.C}")
    print(f"  Gamma: {model.gamma}")
    print(f"  Features: {len(feature_cols)}")
    
    # Test prediction on dummy data
    dummy_features = np.random.rand(1, len(feature_cols))
    dummy_features_scaled = scaler.transform(dummy_features)
    prediction = model.predict(dummy_features_scaled)
    
    print(f"\nTest prediction on random features:")
    print(f"  Predicted PHQ-8 score: {prediction[0]:.2f}")
    
    print("\n✓ Model test complete")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available. Please install:")
        print("   pip install scikit-learn")
        return
    
    # Initialize
    svm = VisualClassifierSVM()
    
    # Load data
    if not svm.load_session_data():
        return
    
    # Prepare features
    if not svm.prepare_visual_features():
        return
    
    # Scale features
    svm.scale_features()
    
    # Hyperparameter tuning
    svm.hyperparameter_tuning()
    
    # LOOCV evaluation
    svm.train_with_loocv()
    
    # Train final model
    svm.train_final_model()
    
    # Visualize
    svm.visualize_results()
    
    # Save
    svm.save_model()
    
    # Test
    test_model()
    
    print_section("MODEL 2B TRAINING COMPLETE")

if __name__ == "__main__":
    main()
