"""
Evaluate and test trained models
"""
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def load_model(model_path: str) -> Dict:
    """Load trained model from disk"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate model on test set
    
    Returns metrics dictionary
    """
    # Simple evaluation
    y_pred = predict(model, X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1': float(f1_score(y_test, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred))
    except:
        metrics['roc_auc'] = None
    
    return metrics


def predict(model: Dict, X: np.ndarray) -> np.ndarray:
    """Make predictions with model"""
    if 'layer1_w' in model:
        # Multi-layer network
        hidden1 = np.tanh(X @ model['layer1_w'] + model['layer1_b'])
        hidden2 = np.tanh(hidden1 @ model['layer2_w'] + model['layer2_b'])
        logits = hidden2 @ model['output_w'] + model['output_b']
    else:
        # Single layer
        hidden = np.tanh(X @ model['weights'] + model['bias'])
        logits = hidden @ model['output_weights']
    
    return (logits.flatten() > 0.5).astype(int)


def save_evaluation_report(metrics: Dict, output_path: str):
    """Save evaluation metrics to JSON"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report saved to {output_path}")


def main():
    """Evaluate models"""
    print("Model Evaluation Tool")
    print("="*50)
    
    # Load test data
    from ml_training.extract_features_for_participants import create_train_test_datasets
    
    X_train, y_train, X_test, y_test = create_train_test_datasets(
        "train_split_Depression_AVEC2017.csv",
        "test_split_Depression_AVEC2017.csv"
    )
    
    if len(X_test) == 0:
        print("Error: No test data loaded")
        return
    
    # Evaluate hybrid model
    try:
        model_path = Path("ml_training/saved_models/hybrid_model_42_10_2.pkl")
        if model_path.exists():
            print(f"\nEvaluating {model_path.name}...")
            model = load_model(str(model_path))
            metrics = evaluate_model(model, X_test, y_test)
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
            
            # Save report
            save_evaluation_report(metrics, "ml_training/evaluation_report_hybrid_42_10_2.json")
    except Exception as e:
        print(f"Error evaluating hybrid model: {e}")
    
    # Evaluate micro-expression model
    try:
        model_path = Path("ml_training/micro_expression_models/microexpression_detector.pkl")
        if model_path.exists():
            print(f"\nEvaluating {model_path.name}...")
            model = load_model(str(model_path))
            metrics = evaluate_model(model, X_test, y_test)
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
            
            save_evaluation_report(metrics, "ml_training/evaluation_report_microexpression.json")
    except Exception as e:
        print(f"Error evaluating micro-expression model: {e}")


if __name__ == "__main__":
    main()
