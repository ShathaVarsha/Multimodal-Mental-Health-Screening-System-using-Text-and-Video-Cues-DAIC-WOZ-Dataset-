import joblib
import numpy as np

# Load model and scaler
ensemble_model = joblib.load('outputs/depression_ensemble_model.joblib')
scaler = joblib.load('outputs/depression_ensemble_scaler.joblib')

def predict_depression_score(facial_features):
    """
    Predict depression score using the ensemble model.
    Expects facial_features as a 1D numpy array or list (length must match training).
    """
    X = np.array(facial_features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    score = ensemble_model.predict(X_scaled)[0]
    return float(score)
