
import pandas as pd
import uuid
from datetime import datetime
import json
from flask import Flask, request, jsonify, send_file, render_template
# Add missing imports for numpy and torch
import numpy as np
import torch

# Define Flask app
app = Flask(__name__)
from transformers import DistilBertTokenizer, DistilBertModel
from ptsd_chat_agent import PCL_QUESTIONS, PCL5_QUESTIONS, PCL_OPTIONS, PCL5_OPTIONS, interpret_pcl_score, interpret_pcl5_score

# Ensure DEPRESSION_OPTIONS is defined before use
DEPRESSION_OPTIONS = {
    'PHQ-9': [
        ["Not at all", "Several days", "More than half the days", "Nearly every day"]
    ] * 9,
    'BDI-II': [
        ["None", "Mild", "Moderate", "Severe"]
    ] * 21,
    'CES-D': [
        ["Rarely or none of the time", "Some or a little of the time", "Occasionally or moderate amount of time", "Most or all of the time"]
    ] * 20,
    'HAM-D': [
        ["None", "Mild", "Moderate", "Severe", "Very severe"]
    ] * 17,
    'GDS': [
        ["Yes", "No"]
    ] * 15,
    'EPDS': [
        ["Never", "Sometimes", "Quite often", "Very often"]
    ] * 10,
    'CDI': [
        ["Never", "Sometimes", "Always"]
    ] * 10
}


# Narrative templates for each PCL-5 item, with {occupation} and {trauma} placeholders
PCL5_NARRATIVE_TEMPLATES = [
    "Sometimes, memories of difficult events can pop up when we least expect it. In your work as a {occupation}, have you found yourself having repeated, unwanted memories of the stressful experience?",
    "Dreams can be powerful. Since the event, have you had repeated, disturbing dreams about what happened?",
    "Some people feel as if they're reliving the experience, even for a moment. Has this happened to you since the event?",
    "Reminders can trigger strong feelings. How often have you felt very upset when something reminded you of the experience?",
    "Our bodies can react, too. Have you had strong physical reactions (like heart pounding, trouble breathing, sweating) when reminded of the event?",
    "People sometimes try to avoid thinking or talking about what happened. Have you found yourself avoiding memories, thoughts, or feelings related to the experience?",
    "Some avoid places, people, or activities that bring back reminders. Have you avoided external reminders of the experience (like people, places, conversations, or situations)?",
    "After trauma, it can be hard to remember everything. Have you had trouble remembering important parts of the experience?",
    "Sometimes, people develop negative beliefs about themselves or the world. Have you noticed strong negative beliefs (like 'I am bad', 'no one can be trusted', 'the world is dangerous')?",
    "Guilt and blame are common. Have you blamed yourself or someone else for what happened or what happened after?",
    "Strong feelings like fear, anger, guilt, or shame can linger. How much have you experienced these since the event?",
    "Things you used to enjoy may lose their appeal. Have you lost interest in activities you used to enjoy?",
    "Some people feel distant or cut off from others. Have you felt this way since the experience?",
    "It can be hard to feel positive emotions. Have you had trouble experiencing positive feelings, like happiness or love for people close to you?",
    "Irritability or anger can show up unexpectedly. Have you had irritable behavior, angry outbursts, or acted aggressively?",
    "Some people take more risks after trauma. Have you found yourself taking too many risks or doing things that could cause you harm?",
    "Being on guard is common. Have you felt 'superalert' or watchful or on guard?",
    "Feeling jumpy or easily startled can happen. How often have you felt this way?",
    "Concentration can be difficult. Have you had trouble concentrating?",
    "Sleep is often affected. Have you had trouble falling or staying asleep?"
]

# Helper to generate personalized PCL-5 questions for a session
def generate_personalized_pcl5_questions(occupation, trauma=None, time_since=None, symptom_timeframe=None):
    occ = occupation if occupation else "person"
    trauma_str = trauma if trauma else "the stressful experience"
    time_str = ''
    if time_since:
        time_str = f" ({time_since} ago)"
    timeframe_str = ''
    if symptom_timeframe:
        if symptom_timeframe == 'week':
            timeframe_str = "in the past week"
        elif symptom_timeframe == 'month':
            timeframe_str = "in the past month"
        else:
            timeframe_str = "recently"
    else:
        timeframe_str = "in the past month"
    questions = []
    for template in PCL5_NARRATIVE_TEMPLATES:
        # Add timeframe and trauma context to each question
        q = template.format(occupation=occ, trauma=trauma_str)
        if timeframe_str:
            q = q.rstrip('?') + f" {timeframe_str}?"
        if time_str:
            q += f" (Event occurred{time_str})"
DEPRESSION_TOOLS = {
    'PHQ-9': {
        'questions': [
            "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?",
            "Over the last 2 weeks, how often have you been feeling down, depressed, or hopeless?",
            "Over the last 2 weeks, how has your sleep been? Any trouble falling asleep, staying asleep, or sleeping too much?",
            "Over the last 2 weeks, how has your energy level been? Have you been feeling tired or had little energy?",
            "Over the last 2 weeks, what's your appetite been like? Any poor appetite or overeating?",
            "Over the last 2 weeks, how have you been feeling about yourself? Any feelings of being a failure or letting yourself or your family down?",
            "Over the last 2 weeks, have you had trouble concentrating on things, such as reading the newspaper or watching television?",
            "Over the last 2 weeks, have you been moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
            "Over the last 2 weeks, have you had thoughts that you would be better off dead, or of hurting yourself in some way?"
        ],
        'score_range': (0, 27),
        'severity': [(5, "Minimal"), (10, "Mild"), (15, "Moderate"), (20, "Moderately Severe"), (27, "Severe")],
        'report_key': 'phq9'
    },
    'BDI-II': {
        'questions': [
            "Sadness: Have you felt sad or down?",
            "Pessimism: Have you felt pessimistic about the future?",
            "Past failure: Have you felt like a failure?",
            "Loss of pleasure: Have you lost interest in things you used to enjoy?",
            "Guilty feelings: Have you felt guilty?",
            "Punishment feelings: Have you felt you should be punished?",
            "Self-dislike: Have you disliked yourself?",
            "Self-criticalness: Have you been critical of yourself?",
            "Suicidal thoughts: Have you had thoughts of suicide?",
            "Crying: Have you cried more than usual?",
            "Agitation: Have you felt agitated?",
            "Loss of interest: Have you lost interest in people or activities?",
            "Indecisiveness: Have you had trouble making decisions?",
            "Worthlessness: Have you felt worthless?",
            "Loss of energy: Have you felt low on energy?",
            "Changes in sleep: Have you had trouble sleeping or slept too much?",
            "Irritability: Have you felt irritable?",
            "Changes in appetite: Have you had changes in appetite?",
            "Concentration difficulty: Have you had trouble concentrating?",
            "Tiredness or fatigue: Have you felt fatigued?",
            "Loss of interest in sex: Have you lost interest in sex?"
        ],
        'score_range': (0, 63),
        'severity': [(13, "Minimal"), (19, "Mild"), (28, "Moderate"), (63, "Severe")],
        'report_key': 'bdi2'
    },
    'CES-D': {
        'questions': [
            "I was bothered by things that usually don't bother me.",
            "I did not feel like eating; my appetite was poor.",
            "I felt that I could not shake off the blues even with help from my family or friends.",
            "I felt I was just as good as other people.",
            "I had trouble keeping my mind on what I was doing.",
            "I felt depressed.",
            "I felt that everything I did was an effort.",
            "I felt hopeful about the future.",
            "I thought my life had been a failure.",
            "I felt fearful.",
            "My sleep was restless.",
            "I was happy.",
            "I talked less than usual.",
            "I felt lonely.",
            "People were unfriendly.",
            "I enjoyed life.",
            "I had crying spells.",
            "I felt sad.",
            "I felt that people dislike me.",
            "I could not get going."
        ],
        'score_range': (0, 60),
        'severity': [(15, "Minimal"), (21, "Mild"), (30, "Moderate"), (60, "Severe")],
        'report_key': 'cesd'
    },
    'HAM-D': {
        'questions': [
            "Depressed mood: Have you felt sad, hopeless, or discouraged?",
            "Feelings of guilt: Have you felt guilty or blamed yourself?",
            "Suicide: Have you had thoughts of death or suicide?",
            "Insomnia (early): Have you had trouble falling asleep?",
            "Insomnia (middle): Have you had trouble staying asleep?",
            "Insomnia (late): Have you woken up too early?",
            "Work and activities: Have you lost interest in work or activities?",
            "Retardation: Have you felt slowed down in your movements or thinking?",
            "Agitation: Have you felt restless or agitated?",
            "Anxiety (psychic): Have you felt anxious or worried?",
            "Anxiety (somatic): Have you had physical symptoms of anxiety (e.g., stomach upset, headache)?",
            "Somatic symptoms (gastrointestinal): Any stomach or digestive issues?",
            "Somatic symptoms (general): Any other physical symptoms?",
            "Genital symptoms: Any changes in sexual interest or function?",
            "Hypochondriasis: Have you worried about your health?",
            "Loss of weight: Any recent weight loss?",
            "Insight: Do you feel you understand your condition?"
        ],
        'score_range': (0, 52),
        'severity': [(7, "Minimal"), (17, "Mild"), (24, "Moderate"), (52, "Severe")],
        'report_key': 'hamd'
    },
    'GDS': {
        'questions': [
            "Are you basically satisfied with your life?",
            "Have you dropped many of your activities and interests?",
            "Do you feel that your life is empty?",
            "Do you often get bored?",
            "Are you hopeful about the future?",
            "Are you afraid that something bad is going to happen to you?",
            "Do you feel happy most of the time?",
            "Do you often feel helpless?",
            "Do you prefer to stay at home, rather than going out and doing new things?",
            "Do you feel you have more problems with memory than most?",
            "Do you think it is wonderful to be alive now?",
            "Do you feel pretty worthless the way you are now?",
            "Do you feel full of energy?",
            "Do you feel that your situation is hopeless?",
            "Do you think that most people are better off than you are?"
        ],
        'score_range': (0, 15),
        'severity': [(5, "Minimal"), (10, "Mild"), (15, "Severe")],
        'report_key': 'gds'
    },
    'EPDS': {
        'questions': [
            "I have been able to laugh and see the funny side of things.",
            "I have looked forward with enjoyment to things.",
            "I have blamed myself unnecessarily when things went wrong.",
            "I have been anxious or worried for no good reason.",
            "I have felt scared or panicky for no very good reason.",
            "Things have been getting on top of me.",
            "I have been so unhappy that I have had difficulty sleeping.",
            "I have felt sad or miserable.",
            "I have been so unhappy that I have been crying.",
            "The thought of harming myself has occurred to me."
        ],
        'score_range': (0, 30),
        'severity': [(10, "Minimal"), (13, "Mild"), (30, "Severe")],
        'report_key': 'epds'
    },
    'CDI': {
        'questions': [
            "I am sad all the time.",
            "Nothing will ever work out for me.",
            "I do not like myself.",
            "I feel alone all the time.",
            "I have trouble sleeping.",
            "I am tired all the time.",
            "I do not want to do anything.",
            "I have trouble making decisions.",
            "I am not interested in anything.",
            "I feel like I am not as good as other kids."
        ],
        'score_range': (0, 20),
        'severity': [(7, "Minimal"), (13, "Mild"), (20, "Severe")],
        'report_key': 'cdi'
    }
}

# Helper to select tools based on intake
def select_depression_tools(intake):
    age = intake.get('age')
    gender = intake.get('gender', '').lower()
    postpartum = intake.get('postpartum', False)
    elderly = intake.get('elderly', False)
    child = intake.get('child', False)
    treatment = intake.get('treatment', False)
    tools = ['PHQ-9']
    if elderly:
        tools.append('GDS')
    # EPDS only for postpartum or females of childbearing age
    if postpartum or (gender == 'female' and age is not None and 18 <= age <= 45):
        tools.append('EPDS')
    # CDI only for children (under 18)
    if child or (age is not None and age < 18):
        tools.append('CDI')
    if treatment:
        tools.append('HAM-D')
    # Always include BDI-II and CES-D for adults only
    if not (child or (age is not None and age < 18)):
        tools += ['BDI-II', 'CES-D']
    # Remove duplicates
    return list(dict.fromkeys(tools))

# --- Multi-tool depression session endpoints ---
@app.route('/start_depression_session', methods=['POST'])
def start_depression_session():
    """Initialize a new multi-tool depression screening session"""
    data = request.json if request.is_json else {}
    session_id = str(uuid.uuid4())
    intake = {
        'age': int(data.get('age', 0)),
        'gender': data.get('gender', '').lower(),
        'postpartum': bool(data.get('postpartum', False)),
        'elderly': bool(data.get('elderly', False)),
        'child': bool(data.get('child', False)),
        'treatment': bool(data.get('treatment', False))
    }
    tools = select_depression_tools(intake)
    session_tools = {tool: {
        'questions': DEPRESSION_TOOLS[tool]['questions'],
        'responses': [],
        'current_question': 0
    } for tool in tools}
    sessions[session_id] = {
        'session_id': session_id,
        'started_at': datetime.now().isoformat(),
        'intake': intake,
        'tools': tools,
        'session_tools': session_tools,
        'current_tool_idx': 0,
        'completed': False
    }
    return jsonify({'session_id': session_id, 'tools': tools, 'status': 'started'})

@app.route('/get_depression_question', methods=['GET'])
def get_depression_question():
    """Get next question for current tool, or completion status"""
    session_id = request.args.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    session = sessions[session_id]
    if session['completed']:
        return jsonify({'complete': True})
    tools = session['tools']
    current_tool_idx = session['current_tool_idx']
    current_tool = tools[current_tool_idx]
    tool_data = session['session_tools'][current_tool]
    q_idx = tool_data['current_question']
    questions = tool_data['questions']
    options = DEPRESSION_OPTIONS.get(current_tool, [None] * len(questions))[q_idx]
    return jsonify({
        'complete': False,
        'tool': current_tool,
        'question': questions[q_idx],
        'options': options,
        'progress': {
            'current': q_idx + 1,
            'total': len(questions),
            'tool': current_tool,
            'tool_idx': current_tool_idx + 1,
            'tool_total': len(tools)
        }
    })

@app.route('/submit_depression_response', methods=['POST'])
def submit_depression_response():
    """Process user response for current tool/question"""
    data = request.json
    session_id = data.get('session_id')
    response = data.get('response', '')
    option_idx = None
    current_tool = None
    q_idx = None
    tool_data = None
    if session_id in sessions:
        session = sessions[session_id]
        tools = session['tools']
        current_tool_idx = session['current_tool_idx']
        current_tool = tools[current_tool_idx]
        tool_data = session['session_tools'][current_tool]
        q_idx = tool_data['current_question']
        options = DEPRESSION_OPTIONS.get(current_tool, [None] * len(tool_data['questions']))[q_idx]
        ans = response.lower() if isinstance(response, str) else ''
        if isinstance(response, int) or (isinstance(response, str) and response.isdigit()):
            option_idx = int(response)
        else:
            if options:
                for idx, opt in enumerate(options):
                    if opt and opt.lower() in ans:
                        option_idx = idx
                        break
            if option_idx is None:
                # Fallback: assign mild (index 1) if not matched
                option_idx = 1
    else:
        return jsonify({'error': 'Invalid session'}), 400
    # Store both raw response and decoded option index
    tool_data['responses'].append({'text': response, 'option_idx': option_idx})
    tool_data['current_question'] += 1
    # Check if done with tool
    if tool_data['current_question'] >= len(tool_data['questions']):
        if current_tool_idx + 1 < len(tools):
            session['current_tool_idx'] += 1
            return jsonify({'complete': False, 'switch_tool': True, 'next_tool': tools[session['current_tool_idx']]})
        else:
            session['completed'] = True
            return jsonify({'complete': True})
    return jsonify({'complete': False})

@app.route('/generate_depression_report', methods=['POST'])
def generate_depression_report():
    """Generate multi-tool depression report"""
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    session = sessions[session_id]
    tools = session['tools']
    report = {}
    for tool in tools:
        responses = session['session_tools'][tool]['responses']
        # Scoring logic per tool
        score = 0
        if tool == 'PHQ-9':
            # Each question: 0 (not at all), 1 (several days), 2 (more than half), 3 (nearly every day)
            # For demo, map keywords to scores
            for r in responses:
                if not r or not r.strip():
                    continue
                ans = r.lower()
                if 'not at all' in ans:
                    score += 0
                elif 'several' in ans:
                    score += 1
                elif 'half' in ans:
                    score += 2
                elif 'nearly' in ans or 'every day' in ans:
                    score += 3
                else:
                    score += 1  # Default mild
        elif tool == 'BDI-II':
            # Each question: 0-3, map keywords
            for r in responses:
                if not r or not r.strip():
                    continue
                ans = r.lower()
                if 'none' in ans or 'no' in ans:
                    score += 0
                elif 'mild' in ans or 'sometimes' in ans:
                    score += 1
                elif 'moderate' in ans or 'often' in ans:
                    score += 2
                elif 'severe' in ans or 'always' in ans:
                    score += 3
                else:
                    score += 1
        elif tool == 'CES-D':
            # Each question: 0-3, similar mapping
            for r in responses:
                if not r or not r.strip():
                    continue
                ans = r.lower()
                if 'rarely' in ans or 'none' in ans:
                    score += 0
                elif 'some' in ans or 'sometimes' in ans:
                    score += 1
                elif 'occasionally' in ans or 'moderate' in ans:
                    score += 2
                elif 'most' in ans or 'always' in ans:
                    score += 3
                else:
                    score += 1
        elif tool == 'HAM-D':
            # Each question: 0-4, map keywords
            for r in responses:
                if not r or not r.strip():
                    continue
                ans = r.lower()
                if 'none' in ans or 'no' in ans:
                    score += 0
                elif 'mild' in ans or 'sometimes' in ans:
                    score += 1
                elif 'moderate' in ans or 'often' in ans:
                    score += 2
                elif 'severe' in ans or 'always' in ans:
                    score += 3
                elif 'very severe' in ans:
                    score += 4
                else:
                    score += 1
        elif tool == 'GDS':
            # Yes/No, 1 point for depressive answer
            for idx, r in enumerate(responses):
                if not r or not r.strip():
                    continue
                ans = r.lower()
                # For GDS, some questions are reverse scored
                reverse = [0, 4, 6, 10, 12]  # Example indices
                if idx in reverse:
                    if 'no' in ans:
                        score += 1
                else:
                    if 'yes' in ans:
                        score += 1
        elif tool == 'EPDS':
            # Each question: 0-3, map keywords
            for r in responses:
                if not r or not r.strip():
                    continue
                ans = r.lower()
                if 'never' in ans or 'no' in ans:
                    score += 0
                elif 'sometimes' in ans or 'occasionally' in ans:
                    score += 1
                elif 'quite often' in ans or 'moderate' in ans:
                    score += 2
                elif 'very often' in ans or 'always' in ans:
                    score += 3
                else:
                    score += 1
        elif tool == 'CDI':
            # Each question: 0-2, map keywords
            for r in responses:
                if not r or not r.strip():
                    continue
                ans = r.lower()
                if 'never' in ans or 'no' in ans:
                    score += 0
                elif 'sometimes' in ans or 'mild' in ans:
                    score += 1
                elif 'always' in ans or 'severe' in ans:
                    score += 2
                else:
                    score += 1
        # Severity mapping
        severity = None
        for threshold, sev in DEPRESSION_TOOLS[tool]['severity']:
            if score <= threshold:
                severity = sev
                break
        if not severity:
            severity = DEPRESSION_TOOLS[tool]['severity'][-1][1]
        report[tool] = {
            'score': score,
            'severity': severity,
            'responses': responses
        }
    session['depression_report'] = report
    return jsonify({'report': report})

# Load models on startup
print("=" * 70)
print(" " * 15 + "LOADING MODELS...")

print("=" * 70)

# Load ensemble model and scaler for depression prediction
from ensemble_predictor import predict_depression_score

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load report templates
with open('outputs/report_templates.json', 'r') as f:
    report_templates = json.load(f)

print("✓ Models loaded successfully")
print("=" * 70)

# Session storage
# Session storage
sessions = {}

# PTSD session storage (reuse sessions dict, but with type)
PTSD_SESSION_TYPE = 'ptsd'
# --- PTSD CHAT ENDPOINTS ---

@app.route('/ptsd_start_session', methods=['POST'])
def ptsd_start_session():
    data = request.json if request.is_json else {}
    name = data.get('name', '').strip()
    occupation = data.get('occupation', '').strip()
    trauma = data.get('trauma', '').strip() if 'trauma' in data else ''
    time_since = data.get('time_since', '').strip() if 'time_since' in data else ''
    symptom_timeframe = data.get('symptom_timeframe', '').strip() if 'symptom_timeframe' in data else ''
    session_id = str(uuid.uuid4())
    personalized_questions = generate_personalized_pcl5_questions(occupation, trauma, time_since, symptom_timeframe)
    sessions[session_id] = {
        'session_id': session_id,
        'started_at': datetime.now().isoformat(),
        'current_question': 0,
        'responses': [],
        'ptsd_version': 'PCL-5',
        'trauma': trauma,
        'time_since': time_since,
        'symptom_timeframe': symptom_timeframe,
        'type': PTSD_SESSION_TYPE,
        'name': name,
        'occupation': occupation,
        'personalized_questions': personalized_questions
    }
    return jsonify({'session_id': session_id, 'status': 'started'})

@app.route('/ptsd_set_version', methods=['POST'])
def ptsd_set_version():
    data = request.json
    session_id = data.get('session_id')
    version = data.get('version', 'PCL-5')
    trauma = data.get('trauma', '')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    sessions[session_id]['ptsd_version'] = version
    sessions[session_id]['trauma'] = trauma
    return jsonify({'status': 'ok'})

@app.route('/ptsd_get_question', methods=['GET'])
def ptsd_get_question():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    session = sessions[session_id]
    version = session.get('ptsd_version', 'PCL-5')
    q_idx = session['current_question']
    questions = session.get('personalized_questions', PCL5_QUESTIONS)
    options = PCL5_OPTIONS if version == 'PCL-5' else PCL_OPTIONS
    if q_idx >= len(questions):
        return jsonify({'complete': True})
    return jsonify({
        'complete': False,
        'question': questions[q_idx],
        'options': options,
        'progress': {
            'current': q_idx + 1,
            'total': len(questions)
        }
    })

@app.route('/ptsd_submit_response', methods=['POST'])
def ptsd_submit_response():
    data = request.json
    session_id = data.get('session_id')
    answer = data.get('answer')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    session = sessions[session_id]
    version = session.get('ptsd_version', 'PCL-5')
    questions = session.get('personalized_questions', PCL5_QUESTIONS)
    options = PCL5_OPTIONS if version == 'PCL-5' else PCL_OPTIONS
    q_idx = session['current_question']

    # Accept both index and free-text answers
    answer_int = None

    if isinstance(answer, int) or (isinstance(answer, str) and answer.isdigit()):
        answer_int = int(answer)
        if not (0 <= answer_int < len(options)):
            answer_int = None
    elif isinstance(answer, str) and answer.strip():
        # Free-text: map to option using simple sentiment/keyword logic
        text = answer.strip().lower()
        # Simple mapping: look for keywords or use sentiment
        keywords = [
            (['not at all', 'never', 'none', 'no'], 0),
            (['a little', 'slightly', 'rarely', 'sometimes', 'minor'], 1),
            (['moderate', 'somewhat', 'occasionally', 'sometimes'], 2),
            (['quite a bit', 'often', 'frequently', 'a lot', 'much'], 3),
            (['extreme', 'extremely', 'always', 'all the time', 'severe', 'worst'], 4)
        ]
        found = False
        for kwlist, idx in keywords:
            if any(kw in text for kw in kwlist):
                answer_int = idx
                found = True
                break
        if not found:
            # Fallback: use simple sentiment (negative = higher score)
            import re
            pos_words = ['not at all', 'no', 'never', 'none']
            neg_words = ['extreme', 'extremely', 'always', 'severe', 'worst']
            score = 0
            for w in pos_words:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    score -= 1
            for w in neg_words:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    score += 1
            # Map score to option
            if score <= -1:
                answer_int = 0
            elif score == 0:
                answer_int = 2
            else:
                answer_int = 4

        # Out-of-context detection (very basic: if no keyword or sentiment match, or answer is off-topic/irrelevant)
        # You can expand this logic as needed
        out_of_context = False
        if not found and score == 0 and len(text.split()) < 3:
            out_of_context = True
        # Add more advanced checks as needed
        if text in ['', 'idk', 'not sure', 'maybe', 'skip']:
            out_of_context = True

        # Pool of 15 varied guidance responses
        PTSD_GUIDANCE_RESPONSES = [
            "Let's focus on the question. Could you share how much this has affected you recently?",
            "I understand, but could you tell me how often you've experienced this in the past month?",
            "If possible, please relate your answer to the question above.",
            "Could you help me understand how much this has bothered you lately?",
            "Let's try to keep your answer about the question asked. How much has this been a problem for you?",
            "If you're unsure, just let me know how often this has happened.",
            "Please try to answer based on your experience with the specific symptom mentioned.",
            "If this doesn't apply, you can say 'not at all' or choose the closest option.",
            "Could you clarify your answer in relation to the question?",
            "If you need, you can skip, but a specific answer helps me understand better.",
            "Let's return to the question. How much has this affected you?",
            "If you can, please describe how often this has occurred for you.",
            "Try to answer about the symptom or feeling in the question above.",
            "If you're not sure, just give your best estimate for how much this has bothered you.",
            "Could you share more about how this relates to your recent experience?"
        ]
        import random
        if out_of_context:
            # Avoid repeating the same message for the same session/question
            last_guidance = session.get('last_guidance', None)
            available_guidance = [g for g in PTSD_GUIDANCE_RESPONSES if g != last_guidance]
            guidance = random.choice(available_guidance) if available_guidance else random.choice(PTSD_GUIDANCE_RESPONSES)
            session['last_guidance'] = guidance
            return jsonify({'complete': False, 'clarify': True, 'message': guidance}), 200
    else:
        answer_int = None

    session['responses'].append(answer_int)
    session['current_question'] += 1
    # If done, calculate score/severity
    if session['current_question'] >= len(questions):
        valid_scores = [a for a in session['responses'] if a is not None]
        score = sum(valid_scores)
        if version == 'PCL-5':
            severity = interpret_pcl5_score(score)
        else:
            severity = interpret_pcl_score(score)
        # Instead of immediately returning, provide a thank you and countdown message
        suggestions = [
            "Thank you for completing the PTSD screening. Your responses have been recorded.",
            "If you would like to talk to someone, consider reaching out to a mental health professional.",
            "If you are in distress, you can contact a helpline such as 9152987821 (India), 1-800-273-8255 (USA), or your local support.",
            "You can now download your personalized report with suggestions for next steps."
        ]
        return jsonify({
            'complete': True,
            'score': score,
            'severity': severity,
            'message': "Thank you for completing the screening! Please wait while your report is generated...",
            'countdown': 5,  # seconds, for frontend to show a countdown
            'suggestions': suggestions
        })
    return jsonify({'complete': False})

@app.route('/ptsd_generate_report', methods=['POST'])
def ptsd_generate_report():
    if request.method != 'POST':
        return jsonify({'error': 'Method Not Allowed'}), 405
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    session = sessions[session_id]
    version = session.get('ptsd_version', 'PCL-5')
    trauma = session.get('trauma', '')
    name = session.get('name', '')
    occupation = session.get('occupation', '')
    responses = session['responses']
    questions = session.get('personalized_questions', PCL5_QUESTIONS)
    options = PCL5_OPTIONS if version == 'PCL-5' else PCL_OPTIONS
    # Build transcript for report
    transcript = []
    n = min(len(responses), len(questions))
    for idx in range(n):
        a = responses[idx]
        if a is not None and 0 <= a < len(options):
            transcript.append(f"Q{idx+1}: {questions[idx]}\nA: {options[a]}")
        else:
            transcript.append(f"Q{idx+1}: {questions[idx]}\nA: Skipped")
    # For compatibility with report generator
    min_len = min(len(questions), len(responses), len(transcript))
    chat_df = pd.DataFrame({
        'question': questions[:min_len],
        'score': responses[:min_len],
        'text': transcript[:min_len]
    })
    pcl5_scores = [a if a is not None else 0 for a in responses]
    output_path = f'outputs/ptsd_report_{session_id[:8]}.pdf'
    generate_ptsd_report(session_id, chat_df, pcl5_scores, transcript, output_path, name=name, occupation=occupation)
    session['ptsd_report_path'] = output_path
    # Calculate total score and severity for display
    valid_scores = [a if a is not None else 0 for a in responses]
    total_score = sum(valid_scores)
    if version == 'PCL-5':
        severity = interpret_pcl5_score(total_score)
    else:
        severity = interpret_pcl_score(total_score)
    suggestions = [
        "Thank you for completing the PTSD screening. Your responses have been recorded.",
        "If you would like to talk to someone, consider reaching out to a mental health professional.",
        "If you are in distress, you can contact a helpline such as 9152987821 (India), 1-800-273-8255 (USA), or your local support.",
        "You can now download your personalized report with suggestions for next steps."
    ]
    return jsonify({'report_path': output_path, 'suggestions': suggestions, 'score': total_score, 'severity': severity})

@app.route('/ptsd_download_report/<session_id>', methods=['GET'])
def ptsd_download_report(session_id):
    if session_id not in sessions or 'ptsd_report_path' not in sessions[session_id]:
        return "Session/report not found", 404
    path = sessions[session_id]['ptsd_report_path']
    return send_file(path, as_attachment=True, download_name=f'ptsd_screening_{session_id[:8]}.pdf')

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
    
    # Predict current score using the new ensemble model (use all facial features)
    current_score = predict_depression_score(facial_features)
    
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
