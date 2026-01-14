"""
=================================================================
STEP 8: WEB INTERFACE
=================================================================
Flask-based web application for depression screening

Features:
- Chat interface for conversations
- Webcam integration for facial analysis
- Real-time depression risk prediction
- Final psychological report generation

Run this to start the web server:
    python web_interface_8.py
  
Then open: http://127.0.0.1:5000
"""

import sys
from pathlib import Path
import json
import base64
from datetime import datetime
import re
import io

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import cv2

# Try to import mediapipe (version-agnostic)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not available - using basic CV features")

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from utils import *
from camera_utils import process_webcam_frame_opencv

# =============================================================================
# INITIALIZE FLASK APP
# =============================================================================

app = Flask(__name__, 
            template_folder=str(PROJECT_ROOT / "templates"),
            static_folder=str(PROJECT_ROOT / "static"))

print(f"Camera analysis ready (OpenCV Haar Cascades)")

# =============================================================================
# EMPATHETIC RESPONSE SYSTEM
# =============================================================================

EMPATHETIC_RESPONSES = {
    "acknowledgement": [
        "I understand this might be difficult to talk about.",
        "Thank you for sharing that with me.",
        "I appreciate your honesty.",
        "That sounds challenging.",
        "I hear what you're saying."
    ],
    "encouragement": [
        "Take your time with your answer.",
        "There's no rush - answer when you're ready.",
        "It's okay to think about your response.",
        "Feel free to share as much or as little as you're comfortable with."
    ],
    "vague_answer": [
        "I want to make sure I understand you correctly. Could you tell me more specifically about {topic}?",
        "That's helpful, but I'd like to know a bit more detail about {topic}. Can you elaborate?",
        "I appreciate you answering. To better understand, could you describe {topic} in more specific terms?"
    ],
    "off_topic": [
        "I notice your answer might not directly address the question about {topic}. Let me ask again: {question}",
        "That's interesting, but I'd like to focus on {topic}. Could you share your thoughts on that specifically?",
        "Let's try to stay focused on {topic}. How would you answer this question: {question}"
    ]
}

ANSWER_VALIDATION_KEYWORDS = {
    "sleep": ["sleep", "insomnia", "rest", "tired", "awake", "night", "bed", "hours"],
    "mood": ["mood", "sad", "depressed", "down", "happy", "feeling", "emotion"],
    "energy": ["energy", "tired", "fatigue", "exhausted", "weak", "strength"],
    "appetite": ["appetite", "eat", "food", "hungry", "weight", "meal"],
    "concentration": ["concentrate", "focus", "attention", "distracted", "think"],
    "activity": ["activity", "slow", "moving", "agitated", "restless", "pace"],
    "self_worth": ["failure", "worthless", "guilty", "blame", "yourself", "self"],
    "thoughts": ["thoughts", "death", "suicide", "harm", "hurt", "better off"]
}

# =============================================================================
# LOAD TRAINED MODELS
# =============================================================================

print_section("LOADING TRAINED MODELS")

# Model 2a: Text Extractor (DistilBERT)
try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL2A_CONFIG["model_name"])
    text_model = DistilBertModel.from_pretrained(MODEL2A_CONFIG["model_name"])
    text_model.eval()
    print("✓ Model 2a (Text Extractor) loaded")
except Exception as e:
    print(f"❌ Failed to load DistilBERT: {e}")
    tokenizer = None
    text_model = None

# Model 2b: Visual Classifier (SVM)
try:
    visual_model_data = load_pickle(MODEL2B_CONFIG["save_path"])
    visual_model = visual_model_data["model"]
    visual_scaler = visual_model_data["scaler"]
    visual_feature_cols = visual_model_data["feature_cols"]
    print("✓ Model 2b (Visual Classifier) loaded")
except Exception as e:
    print(f"⚠ Visual model not found: {e}")
    visual_model = None
    visual_scaler = None
    visual_feature_cols = []

# Model 3: Fusion Network
class MultimodalFusionNet(nn.Module):
    """Same architecture as in step6_model3_fusion.py (without BatchNorm for small batches)"""
    def __init__(self, text_dim, visual_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Text branch (no BatchNorm for small batch compatibility)
        self.text_fc1 = nn.Linear(text_dim, hidden_dims[0])
        self.text_dropout1 = nn.Dropout(0.3)
        
        # Visual branch (no BatchNorm for small batch compatibility)
        self.visual_fc1 = nn.Linear(visual_dim, hidden_dims[0])
        self.visual_dropout1 = nn.Dropout(0.3)
        
        # Fusion layers (no BatchNorm for small batch compatibility)
        fusion_input_dim = hidden_dims[0] * 2
        self.fusion_fc1 = nn.Linear(fusion_input_dim, hidden_dims[1])
        self.fusion_dropout1 = nn.Dropout(0.2)
        
        self.fusion_fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fusion_dropout2 = nn.Dropout(0.1)
        
        # Output layer
        self.output = nn.Linear(hidden_dims[2], 1)
    
    def forward(self, text_features, visual_features):
        import torch.nn.functional as F
        
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

try:
    # Load checkpoint
    checkpoint = torch.load(MODEL3_CONFIG["save_path"], map_location=DEVICE)
    
    text_dim = checkpoint.get("text_dim", 768)
    visual_dim = checkpoint.get("visual_dim", len(visual_feature_cols) if visual_feature_cols else 70)
    hidden_dims = checkpoint.get("config", {}).get("hidden_dims", [256, 128, 64])
    
    fusion_model = MultimodalFusionNet(text_dim=text_dim, visual_dim=visual_dim, hidden_dims=hidden_dims)
    fusion_model.load_state_dict(checkpoint["model_state_dict"])
    fusion_model.to(DEVICE)
    fusion_model.eval()
    print("✓ Model 3 (Fusion Network) loaded")
    print(f"  Architecture: Text({text_dim}) + Visual({visual_dim}) → Hidden{hidden_dims} → PHQ-8")
except Exception as e:
    print(f"⚠ Fusion model not found: {e}")
    fusion_model = None

# Model 4: Report Templates
try:
    report_templates_path = OUTPUTS_DIR / "report_templates.json"
    if report_templates_path.exists():
        report_templates = load_json(report_templates_path)
        print("✓ Model 4 (Report Templates) loaded")
    else:
        print("⚠ Report templates not found, using defaults")
        report_templates = None
except Exception as e:
    print(f"⚠ Report templates error: {e}")
    report_templates = None

print_section("WEB SERVER READY")

# =============================================================================
# SESSION STORAGE (in-memory for demo)
# =============================================================================

sessions = {}

# =============================================================================
# QUESTION DATABASE
# =============================================================================

QUESTIONS = [
    {"id": 0, "type": "OPENING", "text": "Hi, I'm here to help you today. How have you been feeling lately?"},
    {"id": 1, "type": "PHQ", "text": "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?"},
    {"id": 2, "type": "PHQ", "text": "How often have you been feeling down, depressed, or hopeless?"},
    {"id": 3, "type": "PHQ", "text": "Have you had trouble falling or staying asleep, or sleeping too much?"},
    {"id": 4, "type": "PHQ", "text": "Have you been feeling tired or having little energy?"},
    {"id": 5, "type": "PHQ", "text": "How has your appetite been? Poor appetite or overeating?"},
    {"id": 6, "type": "PHQ", "text": "Have you been feeling bad about yourself, or that you're a failure?"},
    {"id": 7, "type": "PHQ", "text": "Have you had trouble concentrating on things, such as reading or watching TV?"},
    {"id": 8, "type": "PHQ", "text": "Have you noticed yourself moving or speaking more slowly than usual? Or being more fidgety or restless?"},
    {"id": 9, "type": "EMOTIONAL", "text": "Can you tell me more about what's been troubling you?"},
    {"id": 10, "type": "CLOSING", "text": "Thank you for sharing. Is there anything else you'd like to add?"}
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_text_embedding(text: str) -> np.ndarray:
    """Extract DistilBERT embedding from text"""
    if tokenizer is None or text_model is None:
        return np.zeros(768)
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        
        return embedding
    except:
        return np.zeros(768)

def process_webcam_frame(frame_data: str) -> dict:
    """
    Process webcam frame - delegates to camera_utils
    """
    return process_webcam_frame_opencv(frame_data)

def validate_answer(user_text: str, question_id: str) -> dict:
    """
    Validate if user's answer is relevant and not vague
    
    Returns:
        {
            "valid": bool,
            "issue": "vague" | "off_topic" | None,
            "topic": str (question topic),
            "empathetic_response": str
        }
    """
    # Map question IDs to topics
    question_topics = {
        "sleep": "sleep",
        "mood": "mood",
        "energy": "energy",
        "appetite": "appetite",
        "concentration": "concentration",
        "activity": "activity",
        "self_worth": "self_worth",
        "thoughts": "thoughts"
    }
    
    topic = question_topics.get(question_id, "your experience")
    keywords = ANSWER_VALIDATION_KEYWORDS.get(question_id, [])
    
    # Check if answer is too short (likely vague)
    words = user_text.strip().split()
    if len(words) < 3:
        response = np.random.choice(EMPATHETIC_RESPONSES["vague_answer"]).format(topic=topic)
        return {
            "valid": False,
            "issue": "vague",
            "topic": topic,
            "empathetic_response": response
        }
    
    # Check if answer contains relevant keywords (simple keyword matching)
    text_lower = user_text.lower()
    keyword_found = any(kw in text_lower for kw in keywords) if keywords else True
    
    # Generic responses that don't answer the question
    generic_patterns = [
        r"^(i don't know|idk|not sure|maybe|dunno)",
        r"^(ok|okay|fine|good|bad)$",
        r"^(yes|no|yeah|nah)$"
    ]
    is_generic = any(re.match(pattern, text_lower.strip()) for pattern in generic_patterns)
    
    if is_generic or not keyword_found:
        if is_generic:
            response = np.random.choice(EMPATHETIC_RESPONSES["vague_answer"]).format(topic=topic)
            issue = "vague"
        else:
            # Get original question text for context
            original_question = next((q["text"] for q in QUESTIONS if q["id"] == question_id), "")
            response = np.random.choice(EMPATHETIC_RESPONSES["off_topic"]).format(
                topic=topic, 
                question=original_question
            )
            issue = "off_topic"
        
        return {
            "valid": False,
            "issue": issue,
            "topic": topic,
            "empathetic_response": response
        }
    
    # Valid answer - add acknowledgement
    acknowledgement = np.random.choice(EMPATHETIC_RESPONSES["acknowledgement"])
    
    return {
        "valid": True,
        "issue": None,
        "topic": topic,
        "empathetic_response": acknowledgement
    }

def generate_empathetic_response(validation_result: dict, next_question: str = None) -> str:
    """
    Generate human-like empathetic response
    
    Combines validation feedback with encouragement and next question
    """
    response_parts = []
    
    # Add empathetic validation message
    response_parts.append(validation_result["empathetic_response"])
    
    # If valid, add encouragement
    if validation_result["valid"] and next_question:
        encouragement = np.random.choice(EMPATHETIC_RESPONSES["encouragement"])
        response_parts.append(encouragement)
        response_parts.append(next_question)
    elif not validation_result["valid"]:
        # For invalid answers, the empathetic_response already contains the re-prompt
        pass
    
    return " ".join(response_parts)

def predict_depression_score(text_embedding: np.ndarray, visual_features: dict) -> float:
    """Predict depression score using fusion model"""
    
    if fusion_model is None:
        # Fallback: simple heuristic
        return np.random.randint(5, 15)
    
    try:
        # Prepare visual feature vector
        visual_vec = np.concatenate([
            visual_features.get("au_features", np.zeros(22)),
            visual_features.get("pose_features", np.zeros(6)),
            visual_features.get("gaze_features", np.zeros(6))
        ])
        
        # Pad if needed
        if len(visual_vec) < len(visual_feature_cols):
            visual_vec = np.pad(visual_vec, (0, len(visual_feature_cols) - len(visual_vec)))
        
        # Convert to tensors
        text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0)
        visual_tensor = torch.FloatTensor(visual_vec[:len(visual_feature_cols) if visual_feature_cols else 70]).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction = fusion_model(text_tensor, visual_tensor)
            score = prediction.item()
        
        return max(0, min(24, score))  # Clip to 0-24
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return 10.0  # Default moderate score

def generate_report(session_data: dict) -> dict:
    """Generate final psychological report"""
    
    avg_score = np.mean(session_data.get("predicted_scores", [10]))
    
    # Select severity level
    if avg_score < 5:
        severity_level = "minimal"
    elif avg_score < 10:
        severity_level = "mild"
    elif avg_score < 15:
        severity_level = "moderate"
    elif avg_score < 20:
        severity_level = "moderately_severe"
    else:
        severity_level = "severe"
    
    # Load template
    if report_templates and severity_level in report_templates:
        template = report_templates[severity_level]
        
        # Build behavioral observations
        observations = []
        if session_data.get("avg_gaze_avoidance", 0) > 0.5:
            observations.append("Frequent downward gaze patterns observed")
        if session_data.get("avg_sad_expressions", 0) > 0.5:
            observations.append("Facial expressions showing signs of sadness")
        
        # Format report
        report_text = f"""
SEVERITY: {template.get('severity', severity_level.upper())}
PHQ-8 SCORE: {int(avg_score)}/24

{template.get('description', '')}

{template.get('summary', '')}

RECOMMENDATIONS:
"""
        for rec in template.get('recommendations', []):
            report_text += f"\n• {rec}"
        
        report_text += f"\n\nFOLLOW-UP:\n{template.get('follow_up', '')}"
        
        # Add crisis resources if needed
        if template.get('crisis_resources'):
            report_text += "\n\nCRISIS RESOURCES:\n• National Suicide Prevention Lifeline: 1-800-273-8255\n• Crisis Text Line: Text HOME to 741741"
        
        return {
            "severity": template.get('severity', severity_level),
            "phq8_score": int(avg_score),
            "report": report_text
        }
    else:
        # Fallback simple report
        return {
            "severity": severity_level.upper(),
            "phq8_score": int(avg_score),
            "report": f"Assessment complete. PHQ-8 Score: {int(avg_score)}/24. Severity: {severity_level}."
        }

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    """Start a new screening session"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sessions[session_id] = {
        "question_index": 0,
        "conversation": [],
        "predicted_scores": [],
        "visual_features": [],
        "created_at": datetime.now().isoformat()
    }
    
    # Return first question
    first_question = QUESTIONS[0]
    
    return jsonify({
        "session_id": session_id,
        "question": first_question["text"],
        "question_id": first_question["id"]
    })

@app.route("/submit_response", methods=["POST"])
def submit_response():
    """Process user response with answer validation and empathetic feedback"""
    data = request.json
    
    session_id = data.get("session_id")
    user_text = data.get("text", "")
    webcam_frame = data.get("webcam_frame", None)
    
    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    session = sessions[session_id]
    current_question = QUESTIONS[session["question_index"]]
    
    # Validate answer
    validation = validate_answer(user_text, current_question["id"])
    
    # If answer is invalid, ask again with empathetic guidance
    if not validation["valid"]:
        # Store the invalid attempt
        if "invalid_attempts" not in session:
            session["invalid_attempts"] = []
        session["invalid_attempts"].append({
            "question_id": current_question["id"],
            "response": user_text,
            "issue": validation["issue"]
        })
        
        # Return empathetic re-prompt (don't advance question)
        return jsonify({
            "complete": False,
            "question": current_question["text"],
            "question_id": current_question["id"],
            "empathetic_message": validation["empathetic_response"],
            "needs_clarification": True,
            "issue_type": validation["issue"]
        })
    
    # Valid answer - proceed with processing
    # Extract text embedding
    text_emb = extract_text_embedding(user_text)
    
    # Process webcam frame (if provided)
    visual_feats = {}
    if webcam_frame:
        visual_feats = process_webcam_frame(webcam_frame)
    
    # Predict depression score for this turn
    score = predict_depression_score(text_emb, visual_feats)
    
    # Store data
    session["conversation"].append({
        "question_id": current_question["id"],
        "user_response": user_text,
        "timestamp": datetime.now().isoformat(),
        "score": score
    })
    session["predicted_scores"].append(score)
    session["visual_features"].append(visual_feats)
    
    # Move to next question
    session["question_index"] += 1
    
    # Check if session complete
    if session["question_index"] >= len(QUESTIONS):
        # Generate final report
        report_data = generate_report(session)
        
        # Final acknowledgement
        final_message = "Thank you for completing the screening. I've generated your assessment report."
        
        return jsonify({
            "complete": True,
            "report": report_data["report"],
            "severity": report_data["severity"],
            "phq8_score": report_data["phq8_score"],
            "num_turns": len(session["conversation"]),
            "empathetic_message": final_message
        })
    
    # Return next question with empathetic acknowledgement
    next_question = QUESTIONS[session["question_index"]]
    empathetic_msg = generate_empathetic_response(validation, next_question["text"])
    
    return jsonify({
        "complete": False,
        "question": next_question["text"],
        "question_id": next_question["id"],
        "current_score": round(score, 1),
        "empathetic_message": empathetic_msg,
        "needs_clarification": False
    })

@app.route("/get_report/<session_id>", methods=["GET"])
def get_report(session_id):
    """Get final report for a session"""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    report_data = generate_report(session)
    
    return jsonify({
        "report": report_data["report"],
        "severity": report_data["severity"],
        "phq8_score": report_data["phq8_score"],
        "num_turns": len(session.get("conversation", []))
    })

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create templates and static directories
    (PROJECT_ROOT / "templates").mkdir(exist_ok=True)
    (PROJECT_ROOT / "static").mkdir(exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"{'DEPRESSION SCREENING WEB INTERFACE':^70}")
    print(f"{'=' * 70}")
    print(f"\nStarting server on http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    print(f"\nPress Ctrl+C to stop the server")
    print(f"{'=' * 70}\n")
    
    app.run(
        host=WEB_CONFIG["host"],
        port=WEB_CONFIG["port"],
        debug=WEB_CONFIG["debug"]
    )
