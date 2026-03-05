"""
Complete Model Inventory & Loading Guide
Shows all available models, their architectures, and how to load/use them
"""
import pickle
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

# ============================================================================
# MODEL INVENTORY
# ============================================================================

MODEL_REGISTRY = {
    # ---- EXPERT MODELS (Single Modality) ----
    "au_expert": {
        "path": "ml_training/saved_models/au_expert.pkl",
        "type": "Expert - Action Units",
        "modality": "Video (Facial Action Units)",
        "input_dims": "17 AUs",
        "architecture": "RandomForest Classifier",
        "description": "Specialized model trained only on facial action units"
    },
    
    "gaze_expert": {
        "path": "ml_training/saved_models/gaze_expert.pkl",
        "type": "Expert - Gaze",
        "modality": "Video (Gaze Direction)",
        "input_dims": "4 (x, y angles + validity)",
        "architecture": "RandomForest Classifier",
        "description": "Specialized model trained only on gaze features"
    },
    
    "pose_expert": {
        "path": "ml_training/saved_models/pose_expert.pkl",
        "type": "Expert - Head Pose",
        "modality": "Video (Head Pose)",
        "input_dims": "6 (3 angles + 3D position)",
        "architecture": "RandomForest Classifier",
        "description": "Specialized model trained only on head pose features"
    },
    
    "landmark_expert": {
        "path": "ml_training/saved_models/landmark_expert.pkl",
        "type": "Expert - Facial Landmarks",
        "modality": "Video (Facial Landmarks)",
        "input_dims": "~70 (facial landmark coordinates)",
        "architecture": "RandomForest Classifier",
        "description": "Specialized model trained on facial landmark features"
    },
    
    "transcript_expert": {
        "path": "ml_training/saved_models/transcript_expert.pkl",
        "type": "Expert - Text/Transcript",
        "modality": "Text (Linguistic Features)",
        "input_dims": "30-50 (linguistic features)",
        "architecture": "RandomForest Classifier",
        "description": "Specialized model trained only on transcript/linguistic features"
    },
    
    # ---- GLOBAL ENSEMBLE MODELS (Multi-Modality) ----
    "rf_global": {
        "path": "ml_training/saved_models/rf_global.pkl",
        "type": "Global Ensemble - RandomForest",
        "modality": "Multi-modal (all features combined)",
        "input_dims": "~120+ (combined from all modalities)",
        "architecture": "RandomForest on all features",
        "description": "RandomForest trained on concatenated features from all modalities"
    },
    
    "xgb_global": {
        "path": "ml_training/saved_models/xgb_global.pkl",
        "type": "Global Ensemble - XGBoost",
        "modality": "Multi-modal (all features combined)",
        "input_dims": "~120+ (combined from all modalities)",
        "architecture": "XGBoost on all features",
        "description": "XGBoost trained on concatenated features from all modalities"
    },
    
    "lgbm_global": {
        "path": "ml_training/saved_models/lgbm_global.pkl",
        "type": "Global Ensemble - LightGBM",
        "modality": "Multi-modal (all features combined)",
        "input_dims": "~120+ (combined from all modalities)",
        "architecture": "LightGBM on all features",
        "description": "LightGBM trained on concatenated features from all modalities"
    },
    
    "mlp_global": {
        "path": "ml_training/saved_models/mlp_global.pkl",
        "type": "Global Ensemble - Neural Network",
        "modality": "Multi-modal (all features combined)",
        "input_dims": "~120+ (combined from all modalities)",
        "architecture": "MLP (Multi-Layer Perceptron)",
        "description": "Neural network trained on concatenated features from all modalities"
    },
    
    # ---- META-LEARNER (Stacking/Ensemble) ----
    "meta_learner": {
        "path": "ml_training/saved_models/meta_learner.pkl",
        "type": "Meta-Learner (Stacking)",
        "modality": "Multi-modal (expert predictions combined)",
        "input_dims": "5 (predictions from 5 expert models)",
        "architecture": "Meta-learner combining expert predictions",
        "description": "Stacking model that learns optimal weights for expert models"
    },
    
    # ---- PREPROCESSING ----
    "scaler": {
        "path": "ml_training/saved_models/scaler.pkl",
        "type": "Preprocessing - StandardScaler",
        "description": "Feature scaling/normalization (StandardScaler)"
    },
    
    # ---- HYBRID MODELS (42-10-2 Split) ----
    "hybrid_42_10_2": {
        "path": "ml_training/saved_models/hybrid_42_10_2/",
        "type": "Hybrid Model - 42-10-2 Split",
        "train_split": "42 depressed participants",
        "test_split": "10 control participants",
        "val_split": "2 balanced participants",
        "description": "Hybrid model trained with proper train/test/val split"
    },
    
    # ---- EXPRESSION-BASED MODELS ----
    "hybrid_107_expression": {
        "path": "ml_training/saved_models/hybrid_107_expression/",
        "type": "Hybrid Model - Expression Based (107 participants)",
        "contains": ["model.pkl", "scaler.pkl", "imputer.pkl"],
        "description": "Hybrid depression detection using facial expression features"
    },
    
    # ---- OPTIMIZED VERSIONS ----
    "optimized": {
        "path": "ml_training/saved_models/optimized/",
        "type": "Optimized Models (hyperparameter tuned)",
        "description": "Models with optimized hyperparameters"
    },
    
    # ---- REGRESSION RESULTS ----
    "regression_results": {
        "path": "ml_training/saved_models/regression_results.pkl",
        "type": "Regression Analysis Results",
        "description": "Regression analysis results for depression severity prediction"
    },
    
    # ---- PYTORCH MODELS ----
    "improved_microexpression_pytorch": {
        "path": "ml_training/saved_models/improved_microexpression_model_best.pth",
        "type": "PyTorch Model - Micro-expression Detection",
        "framework": "PyTorch",
        "description": "PyTorch deep learning model for micro-expression detection"
    },
    
    # ---- MICRO-EXPRESSION MODELS ----
    "au_gcn": {
        "path": "ml_training/micro_expression_models/au_gcn/",
        "type": "Graph Convolutional Network - AU",
        "description": "GCN model for AU-based depression detection"
    },
    
    "hybrid_microexpression": {
        "path": "ml_training/micro_expression_models/hybrid/",
        "contains": ["hybrid_best_model.pth", "hybrid_depression_classifier.pkl"],
        "type": "Hybrid Micro-expression Model",
        "description": "Hybrid model combining traditional ML and deep learning for micro-expressions"
    },
    
    # ---- ENSEMBLE MODELS ----
    "depression_microexpression": {
        "path": "ml_training/saved_models/depression_microexpression/",
        "type": "Ensemble - Depression & Micro-expression",
        "description": "Ensemble combining depression classification with micro-expression detection"
    },
    
    "ensemble_discriminative": {
        "path": "ml_training/saved_models/ensemble_discriminative/",
        "type": "Discriminative Ensemble",
        "description": "Discriminative ensemble model"
    },
    
    "hybrid_proper_split": {
        "path": "ml_training/saved_models/hybrid_proper_split/",
        "type": "Hybrid Model - Proper Train/Test Split",
        "description": "Hybrid model with properly separated train/test data"
    },
    
    "improved_ensemble": {
        "path": "ml_training/saved_models/improved_ensemble/",
        "type": "Improved Ensemble",
        "description": "Enhanced ensemble model with better architecture"
    }
}

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_pickle_model(model_path: str) -> Any:
    """Load a pickle model from disk"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def load_pytorch_model(model_path: str):
    """Load a PyTorch model"""
    try:
        import torch
        model = torch.load(model_path)
        print(f"✓ Loaded PyTorch model from {model_path}")
        return model
    except ImportError:
        print("✗ PyTorch not installed. Install with: pip install torch")
        return None
    except Exception as e:
        print(f"✗ Error loading PyTorch model: {e}")
        return None


def load_ensemble_models() -> Dict[str, Any]:
    """Load all ensemble expert models"""
    experts = {}
    
    expert_models = ['au_expert', 'gaze_expert', 'pose_expert', 'landmark_expert', 'transcript_expert']
    
    for expert_name in expert_models:
        model_info = MODEL_REGISTRY.get(expert_name)
        if model_info:
            model = load_pickle_model(model_info['path'])
            if model:
                experts[expert_name] = model
    
    return experts


def load_global_models() -> Dict[str, Any]:
    """Load all global ensemble models"""
    global_models = {}
    
    global_model_names = ['rf_global', 'xgb_global', 'lgbm_global', 'mlp_global']
    
    for model_name in global_model_names:
        model_info = MODEL_REGISTRY.get(model_name)
        if model_info:
            model = load_pickle_model(model_info['path'])
            if model:
                global_models[model_name] = model
    
    return global_models


# ============================================================================
# HOW MODELS WERE TRAINED/CREATED
# ============================================================================

def print_model_registry():
    """Print all available models"""
    print("\n" + "="*80)
    print("COMPLETE MODEL REGISTRY - ALL TRAINED MODELS")
    print("="*80)
    
    # Group by type
    expert_models = {}
    ensemble_models = {}
    specialized_models = {}
    
    for name, info in MODEL_REGISTRY.items():
        model_type = info.get('type', 'Unknown')
        
        if 'Expert' in model_type:
            expert_models[name] = info
        elif 'Ensemble' in model_type or 'Global' in model_type:
            ensemble_models[name] = info
        else:
            specialized_models[name] = info
    
    # Print expert models
    print("\n🎯 EXPERT MODELS (Single Modality - Specialized)")
    print("-" * 80)
    for name, info in expert_models.items():
        print(f"\n  {name.upper()}")
        print(f"    Type: {info['type']}")
        print(f"    Modality: {info.get('modality', 'N/A')}")
        print(f"    Input: {info.get('input_dims', 'N/A')}")
        print(f"    Path: {info['path']}")
        print(f"    Description: {info.get('description', 'N/A')}")
    
    # Print ensemble models
    print("\n\n🔗 GLOBAL ENSEMBLE MODELS (Multi-Modal - Combined)")
    print("-" * 80)
    for name, info in ensemble_models.items():
        print(f"\n  {name.upper()}")
        print(f"    Type: {info['type']}")
        print(f"    Modality: {info.get('modality', 'N/A')}")
        print(f"    Input: {info.get('input_dims', 'N/A')}")
        print(f"    Path: {info['path']}")
        print(f"    Description: {info.get('description', 'N/A')}")
    
    # Print meta-learner & specialized
    print("\n\n⚙️ SPECIALIZED & META-MODELS")
    print("-" * 80)
    for name, info in specialized_models.items():
        print(f"\n  {name.upper()}")
        print(f"    Type: {info['type']}")
        print(f"    Path: {info['path']}")
        print(f"    Description: {info.get('description', 'N/A')}")


# ============================================================================
# PREDICTION WITH DIFFERENT MODEL TYPES
# ============================================================================

def predict_with_expert_models(features: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Get predictions from all expert models
    
    Features dict should contain:
    - 'au': Action unit features
    - 'gaze': Gaze features
    - 'pose': Head pose features
    - 'landmarks': Facial landmarks
    - 'text': Linguistic features
    """
    predictions = {}
    
    # Load experts
    au_expert = load_pickle_model(MODEL_REGISTRY['au_expert']['path'])
    gaze_expert = load_pickle_model(MODEL_REGISTRY['gaze_expert']['path'])
    pose_expert = load_pickle_model(MODEL_REGISTRY['pose_expert']['path'])
    landmark_expert = load_pickle_model(MODEL_REGISTRY['landmark_expert']['path'])
    transcript_expert = load_pickle_model(MODEL_REGISTRY['transcript_expert']['path'])
    
    # Get predictions
    if au_expert and 'au' in features:
        au_pred = au_expert.predict_proba(features['au'].reshape(1, -1))[0][1]
        predictions['au_expert'] = au_pred
    
    if gaze_expert and 'gaze' in features:
        gaze_pred = gaze_expert.predict_proba(features['gaze'].reshape(1, -1))[0][1]
        predictions['gaze_expert'] = gaze_pred
    
    if pose_expert and 'pose' in features:
        pose_pred = pose_expert.predict_proba(features['pose'].reshape(1, -1))[0][1]
        predictions['pose_expert'] = pose_pred
    
    if landmark_expert and 'landmarks' in features:
        landmark_pred = landmark_expert.predict_proba(features['landmarks'].reshape(1, -1))[0][1]
        predictions['landmark_expert'] = landmark_pred
    
    if transcript_expert and 'text' in features:
        text_pred = transcript_expert.predict_proba(features['text'].reshape(1, -1))[0][1]
        predictions['transcript_expert'] = text_pred
    
    return predictions


def predict_with_meta_learner(expert_predictions: Dict[str, float]) -> float:
    """
    Combine expert predictions using meta-learner
    
    Args:
        expert_predictions: Dict with expert predictions
        
    Returns:
        Final depression probability [0-1]
    """
    meta_learner = load_pickle_model(MODEL_REGISTRY['meta_learner']['path'])
    
    if not meta_learner:
        # Fallback: simple averaging
        return np.mean(list(expert_predictions.values()))
    
    # Stack expert predictions in order
    prediction_vector = np.array([
        expert_predictions.get('au_expert', 0.5),
        expert_predictions.get('gaze_expert', 0.5),
        expert_predictions.get('pose_expert', 0.5),
        expert_predictions.get('landmark_expert', 0.5),
        expert_predictions.get('transcript_expert', 0.5)
    ]).reshape(1, -1)
    
    # Get meta-learner prediction
    final_prediction = meta_learner.predict_proba(prediction_vector)[0][1]
    
    return final_prediction


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Print all models
    print_model_registry()
    
    print("\n\n" + "="*80)
    print("TOTAL MODELS: " + str(len(MODEL_REGISTRY)))
    print("="*80)
    
    # Count by type
    print("\nModel Breakdown:")
    print(f"  Expert Models: 5")
    print(f"  Global Ensemble Models: 4")
    print(f"  Meta-Learner: 1")
    print(f"  Hybrid Models: 4")
    print(f"  Micro-expression Models: 2")
    print(f"  Ensemble Models: 3")
    print(f"  Preprocessing: 1")
    print(f"  Specialized: 2")
    print(f"  ──────────────────")
    print(f"  TOTAL: {len(MODEL_REGISTRY)} saved models")
    
    print("\n\nTo load a specific model:")
    print("  model = load_pickle_model(MODEL_REGISTRY['model_name']['path'])")
    
    print("\nTo load all expert models:")
    print("  experts = load_ensemble_models()")
    
    print("\nTo load all global models:")
    print("  globals = load_global_models()")
