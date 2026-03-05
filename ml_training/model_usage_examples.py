"""
PRACTICAL EXAMPLES: Loading and Using All Depression Detection Models

This file shows exact code snippets for using each model type
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


# ============================================================================
# EXAMPLE 1: LOAD & PREDICT WITH SINGLE EXPERT MODEL
# ============================================================================

def example_au_expert_only():
    """
    Simple example: Use only AU_EXPERT for depression detection
    This is how to use a single-modality model
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Using Single Expert (AU_EXPERT) for Prediction")
    print("="*70)
    
    # Load participant 300's features
    participant_id = 300
    
    # Load AU features from file
    au_path = f"data/{participant_id}/{participant_id}_CLNF_AUs.txt"
    au_data = np.loadtxt(au_path)
    if len(au_data.shape) == 1:
        au_data = au_data.reshape(1, -1)
    
    # Average across video frames to get one feature vector
    au_features = np.mean(au_data[:, :17], axis=0).reshape(1, -1)
    
    # Load the AU expert model
    with open('ml_training/saved_models/au_expert.pkl', 'rb') as f:
        au_expert = pickle.load(f)
    
    # Get prediction
    depression_probability = au_expert.predict_proba(au_features)[0][1]
    depression_class = au_expert.predict(au_features)[0]
    
    print(f"\nParticipant {participant_id}:")
    print(f"  AU Features: {au_features.shape[1]} dimensions")
    print(f"  Depression Probability: {depression_probability:.4f}")
    print(f"  Classification: {'DEPRESSED' if depression_class == 1 else 'CONTROL'}")
    
    return au_expert, au_features


# ============================================================================
# EXAMPLE 2: COMPARE MULTIPLE EXPERTS
# ============================================================================

def example_compare_all_experts():
    """
    Load all 5 experts and get their individual predictions
    Shows agreement/disagreement between modalities
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Predictions from All 5 Experts")
    print("="*70)
    
    participant_id = 300
    
    # Define function to load each modality
    def load_features(pid):
        # AU
        au_path = f"data/{pid}/{pid}_CLNF_AUs.txt"
        au = np.mean(np.loadtxt(au_path)[:, :17], axis=0)
        
        # Gaze
        gaze_path = f"data/{pid}/{pid}_CLNF_gaze.txt"
        gaze = np.mean(np.loadtxt(gaze_path)[:, :4], axis=0)
        
        # Pose
        pose_path = f"data/{pid}/{pid}_CLNF_pose.txt"
        pose = np.mean(np.loadtxt(pose_path)[:, :6], axis=0)
        
        # Landmarks
        landmark_path = f"data/{pid}/{pid}_CLNF_features3D.txt"
        landmark = np.mean(np.loadtxt(landmark_path)[:, :70], axis=0)
        
        return au, gaze, pose, landmark
    
    # Load features
    au, gaze, pose, landmark, = load_features(participant_id)
    
    # Load all experts
    with open('ml_training/saved_models/au_expert.pkl', 'rb') as f:
        au_expert = pickle.load(f)
    with open('ml_training/saved_models/gaze_expert.pkl', 'rb') as f:
        gaze_expert = pickle.load(f)
    with open('ml_training/saved_models/pose_expert.pkl', 'rb') as f:
        pose_expert = pickle.load(f)
    with open('ml_training/saved_models/landmark_expert.pkl', 'rb') as f:
        landmark_expert = pickle.load(f)
    with open('ml_training/saved_models/transcript_expert.pkl', 'rb') as f:
        transcript_expert = pickle.load(f)
    
    # Get predictions
    predictions = {}
    predictions['AU'] = au_expert.predict_proba(au.reshape(1, -1))[0][1]
    predictions['Gaze'] = gaze_expert.predict_proba(gaze.reshape(1, -1))[0][1]
    predictions['Pose'] = pose_expert.predict_proba(pose.reshape(1, -1))[0][1]
    predictions['Landmark'] = landmark_expert.predict_proba(landmark.reshape(1, -1))[0][1]
    
    # Print results
    print(f"\nParticipant {participant_id} - Expert Predictions:")
    print("-" * 70)
    print(f"{'Expert':<20} {'Depression Probability':<25} {'Classification':<15}")
    print("-" * 70)
    for expert_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        classification = "DEPRESSED" if prob > 0.5 else "CONTROL"
        print(f"{expert_name:<20} {prob:<25.4f} {classification:<15}")
    
    avg_prob = np.mean(list(predictions.values()))
    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_prob:<25.4f} {'DEPRESSED' if avg_prob > 0.5 else 'CONTROL':<15}")
    
    return predictions


# ============================================================================
# EXAMPLE 3: USE META-LEARNER (STACKING)
# ============================================================================

def example_meta_learner():
    """
    Combine expert predictions using meta-learner (stacking)
    The meta-learner learns optimal weights for experts
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Using Meta-Learner to Combine Experts (Stacking)")
    print("="*70)
    
    participant_id = 300
    
    # Get expert predictions (reuse from example 2)
    expert_preds = example_compare_all_experts()
    
    # Stack into vector: [AU_prob, Gaze_prob, Pose_prob, Landmark_prob, Text_prob]
    meta_input = np.array([[
        expert_preds['AU'],
        expert_preds['Gaze'],
        expert_preds['Pose'],
        expert_preds['Landmark'],
        expert_preds.get('Transcript', 0.5)  # placeholder if not loaded
    ]])
    
    # Load meta-learner
    with open('ml_training/saved_models/meta_learner.pkl', 'rb') as f:
        meta_learner = pickle.load(f)
    
    # Get final prediction
    final_prob = meta_learner.predict_proba(meta_input)[0][1]
    final_class = meta_learner.predict(meta_input)[0]
    
    print(f"\nMeta-Learner Result:")
    print(f"  Input: Expert predictions [AU, Gaze, Pose, Landmark, Text]")
    print(f"  Meta-Input Vector: {meta_input[0]}")
    print(f"  Final Depression Probability: {final_prob:.4f}")
    print(f"  Final Classification: {'DEPRESSED' if final_class == 1 else 'CONTROL'}")
    
    return final_prob


# ============================================================================
# EXAMPLE 4: USE GLOBAL MODELS (ALL FEATURES COMBINED)
# ============================================================================

def example_global_models():
    """
    Use models trained on concatenated features from all modalities
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Global Models (All Features Combined)")
    print("="*70)
    
    participant_id = 300
    
    # Load all features
    def load_all_features(pid):
        # AU
        au = np.mean(np.loadtxt(f"data/{pid}/{pid}_CLNF_AUs.txt")[:, :17], axis=0)
        # Gaze
        gaze = np.mean(np.loadtxt(f"data/{pid}/{pid}_CLNF_gaze.txt")[:, :4], axis=0)
        # Pose
        pose = np.mean(np.loadtxt(f"data/{pid}/{pid}_CLNF_pose.txt")[:, :6], axis=0)
        # Landmarks
        landmark = np.mean(np.loadtxt(f"data/{pid}/{pid}_CLNF_features3D.txt")[:, :70], axis=0)
        # Audio
        audio = np.mean(pd.read_csv(f"data/{pid}/{pid}_COVAREP.csv", header=0).select_dtypes(include=[np.number]).values, axis=0)[:74]
        
        # Concatenate
        all_features = np.concatenate([au, gaze, pose, landmark, audio])
        return all_features
    
    features = load_all_features(participant_id)
    
    # Load scaler
    with open('ml_training/saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    print(f"\nParticipant {participant_id} - Global Model Predictions:")
    print(f"  Feature Dimensions: {features.shape[0]}")
    print("-" * 70)
    
    # Try loading each global model
    models = {
        'RandomForest': 'ml_training/saved_models/rf_global.pkl',
        'XGBoost': 'ml_training/saved_models/xgb_global.pkl',
        'LightGBM': 'ml_training/saved_models/lgbm_global.pkl',
        'MLP': 'ml_training/saved_models/mlp_global.pkl'
    }
    
    predictions = {}
    for model_name, model_path in models.items():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            prob = model.predict_proba(features_scaled)[0][1]
            predictions[model_name] = prob
            print(f"{model_name:<15} {prob:.4f}  {'DEPRESSED' if prob > 0.5 else 'CONTROL'}")
        except FileNotFoundError:
            print(f"{model_name:<15} [Model not found]")
    
    if predictions:
        avg = np.mean(list(predictions.values()))
        print("-" * 70)
        print(f"{'AVERAGE':<15} {avg:.4f}  {'DEPRESSED' if avg > 0.5 else 'CONTROL'}")
    
    return predictions


# ============================================================================
# EXAMPLE 5: HYBRID MULTIMODAL FUSION
# ============================================================================

def example_hybrid_fusion():
    """
    Use weighted multimodal fusion
    Weights: Video 40%, Audio 35%, Text 20%, Questionnaire 5%
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Hybrid Multimodal Fusion (Weighted)")
    print("="*70)
    
    participant_id = 300
    
    # Step 1: Get expert predictions for each modality
    au = np.mean(np.loadtxt(f"data/{participant_id}/{participant_id}_CLNF_AUs.txt")[:, :17], axis=0)
    gaze = np.mean(np.loadtxt(f"data/{participant_id}/{participant_id}_CLNF_gaze.txt")[:, :4], axis=0)
    pose = np.mean(np.loadtxt(f"data/{participant_id}/{participant_id}_CLNF_pose.txt")[:, :6], axis=0)
    landmark = np.mean(np.loadtxt(f"data/{participant_id}/{participant_id}_CLNF_features3D.txt")[:, :70], axis=0)
    
    with open('ml_training/saved_models/au_expert.pkl', 'rb') as f:
        au_expert = pickle.load(f)
    with open('ml_training/saved_models/gaze_expert.pkl', 'rb') as f:
        gaze_expert = pickle.load(f)
    with open('ml_training/saved_models/pose_expert.pkl', 'rb') as f:
        pose_expert = pickle.load(f)
    with open('ml_training/saved_models/landmark_expert.pkl', 'rb') as f:
        landmark_expert = pickle.load(f)
    
    # Get video prediction (average of facial experts)
    video_probs = [
        au_expert.predict_proba(au.reshape(1, -1))[0][1],
        gaze_expert.predict_proba(gaze.reshape(1, -1))[0][1],
        pose_expert.predict_proba(pose.reshape(1, -1))[0][1],
        landmark_expert.predict_proba(landmark.reshape(1, -1))[0][1]
    ]
    video_score = np.mean(video_probs)
    
    # Get audio prediction (placeholder)
    audio_score = 0.45
    
    # Get text prediction (placeholder)
    text_score = 0.52
    
    # Get questionnaire score (placeholder PHQ-9)
    questionnaire_score = 0.3  # 0.0-1.0 based on PHQ-9 severity
    
    # Weighted fusion
    fusion_weights = {
        'video': 0.40,
        'audio': 0.35,
        'text': 0.20,
        'questionnaire': 0.05
    }
    
    final_score = (
        fusion_weights['video'] * video_score +
        fusion_weights['audio'] * audio_score +
        fusion_weights['text'] * text_score +
        fusion_weights['questionnaire'] * questionnaire_score
    )
    
    print(f"\nParticipant {participant_id} - Multimodal Fusion:")
    print("-" * 70)
    print(f"{'Modality':<20} {'Score':<15} {'Weight':<10} {'Weighted':<15}")
    print("-" * 70)
    print(f"{'Video (Facial)':<20} {video_score:<15.4f} {fusion_weights['video']:<10} {fusion_weights['video']*video_score:<15.4f}")
    print(f"{'Audio (COVAREP)':<20} {audio_score:<15.4f} {fusion_weights['audio']:<10} {fusion_weights['audio']*audio_score:<15.4f}")
    print(f"{'Text (Linguistic)':<20} {text_score:<15.4f} {fusion_weights['text']:<10} {fusion_weights['text']*text_score:<15.4f}")
    print(f"{'Questionnaire':<20} {questionnaire_score:<15.4f} {fusion_weights['questionnaire']:<10} {fusion_weights['questionnaire']*questionnaire_score:<15.4f}")
    print("-" * 70)
    print(f"{'FUSION RESULT':<20} {final_score:<15.4f}")
    print(f"  Classification: {'DEPRESSED' if final_score > 0.5 else 'CONTROL'}")
    print(f"  Confidence: {max(final_score, 1-final_score)*100:.1f}%")


# ============================================================================
# EXAMPLE 6: BATCH PREDICTION (MULTIPLE PARTICIPANTS)
# ============================================================================

def example_batch_prediction():
    """
    Get predictions for multiple participants at once
    Useful for evaluation on test set
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Batch Prediction on Multiple Participants")
    print("="*70)
    
    # Get test participants
    test_participants = [302, 307, 331, 335, 346]
    
    # Load AU expert
    with open('ml_training/saved_models/au_expert.pkl', 'rb') as f:
        au_expert = pickle.load(f)
    
    print(f"\nPredictions for {len(test_participants)} test participants:")
    print("-" * 70)
    print(f"{'Participant':<15} {'AU Probability':<20} {'Classification':<15}")
    print("-" * 70)
    
    for pid in test_participants:
        try:
            au = np.mean(np.loadtxt(f"data/{pid}/{pid}_CLNF_AUs.txt")[:, :17], axis=0)
            prob = au_expert.predict_proba(au.reshape(1, -1))[0][1]
            classification = "DEPRESSED" if prob > 0.5 else "CONTROL"
            print(f"{pid:<15} {prob:<20.4f} {classification:<15}")
        except FileNotFoundError:
            print(f"{pid:<15} {'[Data not found]':<20}")


# ============================================================================
# EXAMPLE 7: MODEL COMPARISON
# ============================================================================

def example_model_comparison():
    """
    Compare predictions from different models on same input
    Shows which model is most confident/disagreeing
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Comparing Predictions Across All Model Types")
    print("="*70)
    
    participant_id = 300
    
    print(f"\nParticipant {participant_id} - All Model Comparisons:")
    print("-" * 80)
    print(f"{'Model Type':<30} {'Probability':<20} {'Classification':<15}")
    print("-" * 80)
    
    # Define models to try
    models_to_test = {
        'AU Expert Only': 'ml_training/saved_models/au_expert.pkl',
        'Transcript Expert Only': 'ml_training/saved_models/transcript_expert.pkl',
        'RandomForest Global': 'ml_training/saved_models/rf_global.pkl',
        'XGBoost Global': 'ml_training/saved_models/xgb_global.pkl',
        'Meta-Learner Stacking': 'ml_training/saved_models/meta_learner.pkl'
    }
    
    for model_name, model_path in models_to_test.items():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Create appropriate input based on model type
            if 'Expert' in model_name:
                if 'AU' in model_name:
                    au = np.mean(np.loadtxt(f"data/{participant_id}/{participant_id}_CLNF_AUs.txt")[:, :17], axis=0)
                    input_data = au.reshape(1, -1)
                else:  # Transcript
                    input_data = np.zeros((1, 50))  # placeholder
            else:
                input_data = np.zeros((1, 5))  # placeholder for simplified examples
            
            prob = model.predict_proba(input_data)[0][1]
            classification = "DEPRESSED" if prob > 0.5 else "CONTROL"
            
            print(f"{model_name:<30} {prob:<20.4f} {classification:<15}")
        except Exception as e:
            print(f"{model_name:<30} {'[Error]':<20}")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PRACTICAL EXAMPLES: LOADING & USING ALL DEPRESSION DETECTION MODELS")
    print("="*80)
    
    # Run examples
    try:
        example_au_expert_only()
    except Exception as e:
        print(f"Example 1 Error: {e}")
    
    try:
        example_compare_all_experts()
    except Exception as e:
        print(f"Example 2 Error: {e}")
    
    try:
        example_meta_learner()
    except Exception as e:
        print(f"Example 3 Error: {e}")
    
    try:
        example_global_models()
    except Exception as e:
        print(f"Example 4 Error: {e}")
    
    try:
        example_hybrid_fusion()
    except Exception as e:
        print(f"Example 5 Error: {e}")
    
    try:
        example_batch_prediction()
    except Exception as e:
        print(f"Example 6 Error: {e}")
    
    try:
        example_model_comparison()
    except Exception as e:
        print(f"Example 7 Error: {e}")
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nFor more details, see:")
    print("  - MODEL_LOADING_GUIDE.py")
    print("  - train_complete_pipeline.py")
    print("  - MODEL_INVENTORY.md")
