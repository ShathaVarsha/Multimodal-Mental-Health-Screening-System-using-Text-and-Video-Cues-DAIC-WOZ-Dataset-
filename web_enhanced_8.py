"""
Enhanced Web Interface with Camera UI, Facial Tracking, and PDF Reports
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import pickle
import os
import uuid
from datetime import datetime
import json

# Import utilities
from camera_utils import process_webcam_frame_opencv
from report_generator import generate_clinical_report
from utils import load_pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'depression-screening-secret-key-2025'

# Load models on startup
print("=" * 70)
print(" " * 15 + "LOADING MODELS...")
print("=" * 70)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load fusion model
from fusion_model_6 import MultimodalFusionNet

fusion_model = MultimodalFusionNet(text_dim=768, visual_dim=30)
checkpoint = torch.load('models/model3_fusion.pth', weights_only=False)
fusion_model.load_state_dict(checkpoint['model_state_dict'])
fusion_model.eval()

# Load report templates
with open('outputs/report_templates.json', 'r') as f:
    report_templates = json.load(f)

print("✓ Models loaded successfully")
print("=" * 70)

# Session storage
sessions = {}

# PHQ-8 Questions - Clear and conversational
PHQ8_QUESTIONS = [
    "Over the last 2 weeks, how often have you had little interest or pleasure in doing things? (Not at all, several days, more than half the days, or nearly every day?)",
    "Over the last 2 weeks, how often have you been feeling down, depressed, or hopeless?",
    "Over the last 2 weeks, how has your sleep been? Any trouble falling asleep, staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how has your energy level been? Have you been feeling tired or had little energy?",
    "Over the last 2 weeks, what's your appetite been like? Any poor appetite or overeating?",
    "Over the last 2 weeks, how have you been feeling about yourself? Any feelings of being a failure or letting yourself or your family down?",
    "Over the last 2 weeks, have you had trouble concentrating on things, such as reading the newspaper or watching television?",
    "Over the last 2 weeks, have you been moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
]

# Empathetic responses
EMPATHETIC_RESPONSES = {
    'acknowledgement': [
        "Thank you for sharing that with me.",
        "I appreciate you being open about this.",
        "That's very helpful to know.",
        "I understand. Thank you for telling me."
    ],
    'encouragement': [
        "Take your time with your answer.",
        "There are no right or wrong answers here.",
        "Whatever you're feeling is valid.",
        "It's okay to take a moment to think about this."
    ],
    'vague_answer': [
        "I understand this might be difficult to talk about. Could you tell me more specifically about {topic}?",
        "Thank you for sharing. Can you help me understand a bit more about {topic}?",
        "I'd like to understand better. Could you give me some more details about {topic}?",
        "That's a start. Can you elaborate on how {topic} has affected you?"
    ],
    'off_topic': [
        "I appreciate your response. Let's focus on {topic}. Can you share your thoughts on that?",
        "I understand. For this question, I'm specifically asking about {topic}. How has that been for you?",
        "Thank you. Could we talk more specifically about {topic}?",
        "I hear you. Let's return to {topic} - how have you been experiencing that?"
    ]
}

# Keywords for validation
ANSWER_VALIDATION_KEYWORDS = {
    'sleep': ['sleep', 'insomnia', 'rest', 'tired', 'hours', 'bed', 'wake', 'night', 'asleep'],
    'mood': ['sad', 'depressed', 'down', 'happy', 'feeling', 'mood', 'hopeless', 'blue'],
    'energy': ['energy', 'tired', 'fatigue', 'exhausted', 'weak', 'strength'],
    'appetite': ['appetite', 'eat', 'food', 'hungry', 'weight', 'meal'],
    'concentration': ['concentrate', 'focus', 'attention', 'think', 'read', 'watch', 'distracted'],
    'activity': ['slow', 'moving', 'agitated', 'restless', 'fidgety', 'pace'],
    'self_worth': ['failure', 'worthless', 'guilty', 'blame', 'disappointed', 'letting'],
    'thoughts': ['death', 'suicide', 'harm', 'hurt', 'die', 'better off']
}


def validate_answer(answer, question_idx):
    """Validate if answer is sufficient and relevant"""
    import random
    
    answer_lower = answer.lower()
    
    # Too short - only flag if REALLY short
    if len(answer.split()) < 2:
        topic = get_question_topic(question_idx)
        return {
            'valid': False,
            'empathetic_message': random.choice(EMPATHETIC_RESPONSES['vague_answer']).format(topic=topic),
            'needs_clarification': True
        }
    
    # Generic/vague responses - only flag if very generic AND short
    generic_phrases = ['idk', 'i dont know', 'dunno']
    if any(phrase in answer_lower for phrase in generic_phrases) and len(answer.split()) < 3:
        topic = get_question_topic(question_idx)
        return {
            'valid': False,
            'empathetic_message': random.choice(EMPATHETIC_RESPONSES['vague_answer']).format(topic=topic),
            'needs_clarification': True
        }
    
    # Don't check relevance - too strict and annoying
    # Accept most answers as long as they're not extremely vague
    
    # Valid answer
    return {
        'valid': True,
        'empathetic_message': random.choice(EMPATHETIC_RESPONSES['acknowledgement']),
        'needs_clarification': False
    }


def get_question_topic(idx):
    """Get human-readable topic for question"""
    topics = ['interest/pleasure', 'mood', 'sleep', 'energy', 'appetite', 
              'self-worth', 'concentration', 'movement/speech']
    return topics[idx] if idx < len(topics) else 'this'


def get_topic_key(idx):
    """Get topic key for keywords"""
    keys = ['activity', 'mood', 'sleep', 'energy', 'appetite', 
            'self_worth', 'concentration', 'activity']
    return keys[idx] if idx < len(keys) else 'mood'


@app.route('/')
def index():
    return render_template('index_new.html')


@app.route('/start_session', methods=['POST'])
def start_session():
    """Initialize a new screening session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'session_id': session_id,
        'started_at': datetime.now().isoformat(),
        'current_question': 0,
        'responses': [],
        'facial_features': [],
        'text_embeddings': [],
        'scores_history': []
    }
    
    return jsonify({'session_id': session_id, 'status': 'started'})


@app.route('/get_question', methods=['GET'])
def get_question():
    """Get next question or completion status"""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    q_idx = session['current_question']
    
    if q_idx >= len(PHQ8_QUESTIONS):
        return jsonify({'complete': True})
    
    return jsonify({
        'complete': False,
        'question': PHQ8_QUESTIONS[q_idx],
        'progress': {
            'current': q_idx + 1,
            'total': len(PHQ8_QUESTIONS),
            'current_score': session['scores_history'][-1] if session['scores_history'] else 0.0
        }
    })


@app.route('/submit_response', methods=['POST'])
def submit_response():
    """Process user response with validation and facial analysis"""
    import random
    
    data = request.json
    response_text = data.get('response', '')
    webcam_frame = data.get('webcam_frame')
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    q_idx = session['current_question']
    
    # Check if we're in follow-up mode
    if session.get('awaiting_followup'):
        # This is a follow-up answer - process it and move to next question
        session['awaiting_followup'] = False
        follow_up_text = session.get('followup_context', '')
        combined_answer = session.get('initial_answer', '') + " " + response_text
        
        # Process the combined answer
        return process_complete_answer(session, q_idx, combined_answer, webcam_frame)
    
    # Validate initial answer
    validation = validate_answer(response_text, q_idx)
    
    if not validation['valid']:
        return jsonify({
            'empathetic_message': validation['empathetic_message'],
            'needs_clarification': True,
            'question_text': PHQ8_QUESTIONS[q_idx]
        })
    
    # Check if we need a follow-up
    followup = generate_adaptive_followup(response_text, q_idx)
    
    if followup:
        # Store initial answer and ask follow-up
        session['awaiting_followup'] = True
        session['initial_answer'] = response_text
        session['followup_context'] = followup
        
        return jsonify({
            'empathetic_message': validation['empathetic_message'],
            'followup_question': followup,
            'needs_clarification': True,  # Keep on same question
            'question_text': PHQ8_QUESTIONS[q_idx]
        })
    
    # No follow-up needed, process normally
    return process_complete_answer(session, q_idx, response_text, webcam_frame)


def generate_adaptive_followup(answer, question_idx):
    """Generate contextual follow-up question based on answer"""
    answer_lower = answer.lower()
    
    # Only ask follow-up if answer suggests a problem (not for "fine" or "no issues")
    positive_indicators = ['fine', 'good', 'great', 'normal', 'no problem', 'not really', 'rarely', 'not at all']
    if any(indicator in answer_lower for indicator in positive_indicators):
        return None  # No follow-up needed
    
    # Question-specific follow-ups
    if question_idx == 0:  # Interest/pleasure
        if any(word in answer_lower for word in ['lost', 'no', 'little', 'none', 'not interested']):
            return "That sounds difficult. What activities have you lost interest in?"
    
    elif question_idx == 1:  # Feeling down
        if any(word in answer_lower for word in ['down', 'depressed', 'hopeless', 'sad', 'low']):
            return "I understand. Can you tell me what that feels like for you?"
    
    elif question_idx == 2:  # Sleep
        if any(word in answer_lower for word in ['trouble', 'insomnia', 'cant sleep', 'wake', 'too much']):
            return "Sleep issues can be really tough. About how many hours are you getting per night?"
    
    elif question_idx == 3:  # Energy
        if any(word in answer_lower for word in ['tired', 'exhausted', 'no energy', 'fatigue', 'drained']):
            return "Low energy can be challenging. Does this affect your daily activities?"
    
    elif question_idx == 4:  # Appetite
        if any(word in answer_lower for word in ['no appetite', 'overeating', 'too much', 'too little', 'weight']):
            return "Changes in appetite can be concerning. Have you noticed any weight changes?"
    
    elif question_idx == 5:  # Self-worth
        if any(word in answer_lower for word in ['failure', 'worthless', 'disappointed', 'guilty', 'bad']):
            return "Those feelings sound really hard. What makes you feel this way?"
    
    elif question_idx == 6:  # Concentration
        if any(word in answer_lower for word in ['trouble', 'cant focus', 'distracted', 'hard to', 'difficult']):
            return "Concentration difficulties can be frustrating. When do you notice this most?"
    
    elif question_idx == 7:  # Psychomotor
        if any(word in answer_lower for word in ['slow', 'restless', 'fidgety', 'agitated', 'moving']):
            return "Have others mentioned noticing these changes in how you move or speak?"
    
    return None  # No follow-up needed


def process_complete_answer(session, q_idx, response_text, webcam_frame):
    """Process complete answer (after any follow-ups) and advance to next question"""
    import random
def process_complete_answer(session, q_idx, response_text, webcam_frame):
    """Process complete answer (after any follow-ups) and advance to next question"""
    import random
    
    # Process webcam if available
    facial_features = None
    facial_feedback = None
    confidence = 0.0
    
    if webcam_frame:
        result = process_webcam_frame_opencv(webcam_frame)
        if result.get('face_detected'):
            # Combine all features into single array
            facial_features = np.concatenate([
                result['au_features'],
                result['pose_features'],
                result['gaze_features']
            ])
            confidence = result.get('confidence', 0.0)
            
            # Generate real-time feedback
            facial_feedback = generate_facial_feedback(result)
        else:
            facial_features = np.zeros(34)
            facial_feedback = "No face detected in this frame"
    else:
        facial_features = np.zeros(34)
        facial_feedback = "Camera not enabled"
    
    # Store facial features
    session['facial_features'].append(facial_features.tolist() if isinstance(facial_features, np.ndarray) else facial_features)
    
    # Get text embeddings
    with torch.no_grad():
        tokens = tokenizer(response_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        text_embedding = text_model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
    
    session['text_embeddings'].append(text_embedding.tolist())
    
    # Predict current score
    visual_features = np.array(facial_features[:30])
    
    with torch.no_grad():
        text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0)
        visual_tensor = torch.FloatTensor(visual_features).unsqueeze(0)
        current_score = fusion_model(text_tensor, visual_tensor).item()
    
    session['scores_history'].append(current_score)
    
    # Store response
    session['responses'].append({
        'question': PHQ8_QUESTIONS[q_idx],
        'answer': response_text,
        'timestamp': datetime.now().isoformat(),
        'confidence': float(confidence)
    })
    
    # Advance question
    session['current_question'] += 1
    
    # Generate empathetic acknowledgment
    acknowledgments = [
        "Thank you for sharing that with me.",
        "I appreciate you being open about this.",
        "That's very helpful to know."
    ]
    
    return jsonify({
        'empathetic_message': random.choice(acknowledgments),
        'needs_clarification': False,
        'facial_feedback': facial_feedback,
        'current_score': round(current_score, 1),
        'question_text': PHQ8_QUESTIONS[q_idx]
    })
    facial_features = None
    facial_feedback = None
    confidence = 0.0
    
    if webcam_frame:
        result = process_webcam_frame_opencv(webcam_frame)
        if result.get('face_detected'):
            # Combine all features into single array
            facial_features = np.concatenate([
                result['au_features'],
                result['pose_features'],
                result['gaze_features']
            ])
            confidence = result.get('confidence', 0.0)
            
            # Generate real-time feedback
            facial_feedback = generate_facial_feedback(result)
        else:
            facial_features = np.zeros(34)  # Placeholder
            facial_feedback = "No face detected in this frame"
    else:
        facial_features = np.zeros(34)
        facial_feedback = "Camera not enabled"
    
    # Store facial features
    session['facial_features'].append(facial_features.tolist() if isinstance(facial_features, np.ndarray) else facial_features)
    
    # Get text embeddings
    with torch.no_grad():
        tokens = tokenizer(response_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        text_embedding = text_model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
    
    session['text_embeddings'].append(text_embedding.tolist())
    
    # Predict current score (running prediction)
    visual_features = np.array(facial_features[:30])  # First 30 features
    
    with torch.no_grad():
        text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0)
        visual_tensor = torch.FloatTensor(visual_features).unsqueeze(0)
        current_score = fusion_model(text_tensor, visual_tensor).item()
    
    session['scores_history'].append(current_score)
    
    # Store response
    session['responses'].append({
        'question': PHQ8_QUESTIONS[q_idx],
        'answer': response_text,
        'timestamp': datetime.now().isoformat(),
        'confidence': float(confidence)
    })
    
    # Advance question
    session['current_question'] += 1
    
    return jsonify({
        'empathetic_message': validation['empathetic_message'],
        'needs_clarification': False,
        'facial_feedback': facial_feedback,
        'current_score': round(current_score, 1),
        'question_text': PHQ8_QUESTIONS[q_idx]
    })


def generate_facial_feedback(facial_result):
    """Generate real-time facial feedback"""
    au_features = facial_result['au_features']
    
    feedback_parts = []
    
    # Smile
    if au_features[4] > 0.5:
        feedback_parts.append("Positive expression detected")
    elif au_features[4] < 0.1:
        feedback_parts.append("Minimal facial expression")
    
    # Eye openness
    eye_avg = (au_features[0] + au_features[1]) / 2
    if eye_avg < 0.3:
        feedback_parts.append("Low eye openness (possible fatigue)")
    elif eye_avg > 0.7:
        feedback_parts.append("Alert and engaged")
    
    # Overall expressiveness
    expressiveness = np.std(au_features)
    if expressiveness < 0.1:
        feedback_parts.append("Flat affect")
    elif expressiveness > 0.3:
        feedback_parts.append("Expressive communication")
    
    if not feedback_parts:
        return "Neutral facial expression"
    
    return " | ".join(feedback_parts)


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate final assessment report"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    # Calculate final score (average of last 3 predictions for stability)
    final_scores = session['scores_history'][-3:] if len(session['scores_history']) >= 3 else session['scores_history']
    final_score = np.mean(final_scores)
    
    # Determine severity
    if final_score < 5:
        severity = "Minimal Depression"
        template_key = "minimal"
    elif final_score < 10:
        severity = "Mild Depression"
        template_key = "mild"
    elif final_score < 15:
        severity = "Moderate Depression"
        template_key = "moderate"
    elif final_score < 20:
        severity = "Moderately Severe Depression"
        template_key = "moderately_severe"
    else:
        severity = "Severe Depression"
        template_key = "severe"
    
    # Get template
    template = report_templates.get(template_key, report_templates['mild'])
    
    # Build HTML report
    report_html = f"""
    <h3>{severity}</h3>
    <p><strong>PHQ-8 Score:</strong> {final_score:.1f} / 24</p>
    <hr>
    <h4>Assessment Summary</h4>
    <p>{template.get('summary', template.get('description', ''))}</p>
    <h4>Recommendations</h4>
    <ul>
    """
    
    for rec in template.get('recommendations', []):
        report_html += f"<li>{rec}</li>"
    
    report_html += """
    </ul>
    <h4>Follow-Up</h4>
    <p>{}</p>
    """.format(template.get('follow_up', 'Please consult with a healthcare provider.'))
    
    if template.get('crisis_resources'):
        report_html += """
        <hr>
        <h4 style="color: red;">CRISIS RESOURCES</h4>
        <p><strong>{}</strong></p>
        """.format(template['crisis_resources'])
    
    # Store for PDF generation
    session['final_report'] = {
        'phq8_score': final_score,
        'severity': severity,
        'template': template
    }
    
    return jsonify({
        'phq8_score': round(final_score, 1),
        'severity': severity,
        'report_html': report_html
    })


@app.route('/download_report/<session_id>', methods=['GET'])
def download_report(session_id):
    """Generate and download detailed PDF report"""
    if session_id not in sessions:
        return "Session not found", 404
    
    session = sessions[session_id]
    
    # Generate PDF
    pdf_path = generate_clinical_report(
        session_data={'session_id': session_id, 'started_at': session['started_at']},
        responses=session['responses'],
        phq8_score=session['final_report']['phq8_score'],
        facial_features=session['facial_features']
    )
    
    return send_file(pdf_path, as_attachment=True, download_name=f'depression_screening_{session_id[:8]}.pdf')


if __name__ == '__main__':
    print("=" * 70)
    print(" " * 15 + "DEPRESSION SCREENING WEB INTERFACE")
    print("=" * 70)
    print()
    print("✓ Models loaded")
    print("✓ Camera support enabled")
    print("✓ PDF reports enabled")
    print()
    print("Starting server on http://127.0.0.1:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(host='127.0.0.1', port=5000, debug=True)
