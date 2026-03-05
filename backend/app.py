"""
Flask application for multimodal depression screening system
REST API for questionnaire, video analysis, and integrated assessment
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import traceback
import numpy as np

from backend.session_manager import session_manager
from backend.questionnaire import PHQ9Questionnaire, QuestionnaireSession
from backend.video_analyzer import VideoAnalyzer
from backend.services.feature_extractor import FeatureExtractor
from backend.services.hybrid_model import HybridDepressionModel
from backend.services.llm_service import LLMTextAnalyzer
from backend.services.microexpression_service import MicroExpressionDetector
from backend.services.report_generator import ReportGenerator
from backend.services.text_interpreter import TextInterpreter, identify_scenario
from backend.services.video_processor import VideoProcessor
from backend.services.assessment_data_loader import AssessmentDataLoader
from backend.fusion_engine import FusionEngine

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure max file upload size (50MB for video)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Initialize services
feature_extractor = FeatureExtractor(data_dir="data")
video_analyzer = VideoAnalyzer(data_dir="data")
video_processor = VideoProcessor()
hybrid_model = HybridDepressionModel()

text_analyzer = LLMTextAnalyzer()
microexpression_detector = MicroExpressionDetector()
report_generator = ReportGenerator()
fusion_engine = FusionEngine()
assessment_data_loader = AssessmentDataLoader(root_dir=".")

# Store questionnaire sessions
questionnaire_sessions = {}


def _personalize_question_text(question_text: str, assessment_type: str, scenario: str | None) -> str:
    """Make conversational questions feel specific to selected user scenario."""
    scenario = (scenario or '').strip().lower()
    if not scenario:
        return question_text

    depression_contexts = {
        'postpartum': 'in your postpartum experience',
        'work': 'around your work stress or burnout',
        'school': 'around your academic stress',
        'relationship': 'within your relationship situation',
        'loss': 'while coping with your loss or grief',
        'general': 'in your day-to-day life'
    }
    ptsd_contexts = {
        'accident': 'related to the accident or injury you went through',
        'violence': 'related to the violence or assault you experienced',
        'military': 'related to your military or combat experience',
        'loss': 'related to the sudden loss you experienced',
        'disaster': 'related to the disaster experience you went through',
        'other': 'related to the traumatic experience you identified'
    }

    context_map = depression_contexts if assessment_type == 'depression' else ptsd_contexts
    context = context_map.get(scenario)
    if not context:
        return question_text

    return f"{question_text} Thinking specifically about what you faced {context}, how often has this been true for you?"


# ============================================================================
# SESSION MANAGEMENT ROUTES
# ============================================================================

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create new screening session"""
    try:
        session_id = session_manager.create_session()
        questionnaire_sessions[session_id] = QuestionnaireSession(session_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session information"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'session': session
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/delete', methods=['POST'])
def delete_session(session_id):
    """Delete session after report generation"""
    try:
        success = session_manager.delete_session(session_id)
        if session_id in questionnaire_sessions:
            del questionnaire_sessions[session_id]
        
        return jsonify({'success': success}), 200 if success else 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# QUESTIONNAIRE ROUTES (PHQ-9)
# ============================================================================

@app.route('/api/questionnaire/questions', methods=['GET'])
def get_questionnaire_questions():
    """Get PHQ-9 questions in conversational format"""
    scenario = request.args.get('scenario', '')
    questions_list = []
    for i in range(len(PHQ9Questionnaire.QUESTIONS)):
        questions_list.append(PHQ9Questionnaire.get_conversational_question(i, scenario=scenario))
    
    return jsonify({
        'success': True,
        'questions': questions_list,
        'total_questions': len(questions_list),
        'response_scale': PHQ9Questionnaire.RESPONSE_LABELS,
        'note': 'Questions presented in conversational format for comfort'
    }), 200


@app.route('/api/questionnaire/<session_id>/submit', methods=['POST'])
def submit_questionnaire(session_id):
    """Submit questionnaire responses"""
    try:
        data = request.get_json()
        responses = data.get('responses', [])
        
        # Validate
        is_valid, msg = PHQ9Questionnaire.validate_responses(responses)
        if not is_valid:
            return jsonify({'error': msg}), 400
        
        # Get or create session
        if session_id not in questionnaire_sessions:
            questionnaire_sessions[session_id] = QuestionnaireSession(session_id)
        
        q_session = questionnaire_sessions[session_id]
        
        # Add responses
        for idx, response in enumerate(responses):
            q_session.add_response(idx, int(response))
        
        # Calculate score
        score_data = q_session.finalize()
        
        # Update main session
        session_manager.update_session(session_id, {
            'phq_total_score': score_data['total_score'],
            'phq_severity': score_data['severity'],
            'suicide_risk': score_data['suicide_risk'],
            'questionnaire_completed_at': datetime.utcnow().isoformat()
        })
        
        return jsonify({
            'success': True,
            'score': score_data['total_score'],
            'severity': score_data['severity'],
            'suicide_risk': score_data['suicide_risk'],
            'interpretation': PHQ9Questionnaire._interpret_phq_score(score_data['total_score'])
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# CONVERSATIONAL ASSESSMENT ROUTES
# ============================================================================

@app.route('/api/assessment/conversational/interpret', methods=['POST'])
def interpret_text_response():
    """Convert natural language response to assessment scale"""
    try:
        data = request.json
        text_response = data.get('response', '')
        question = data.get('question', None)
        
        if not text_response:
            return jsonify({'error': 'No response provided'}), 400
        
        # Interpret the response
        scale, interpretation = TextInterpreter.interpret_response(text_response, question)
        
        return jsonify({
            'success': True,
            'response': text_response,
            'scale_value': scale,
            'interpretation': interpretation,
            'scale_label': ['Not at all', 'A few days', 'More than half the days', 'Nearly every day'][scale]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assessment/conversational/phq9', methods=['GET'])
def get_phq9_conversational():
    """Get PHQ-9 questions in conversational format with scenario-specific follow-ups"""
    try:
        scenario = request.args.get('scenario', 'general')

        data_questions = assessment_data_loader.get_questions('phq9')
        followup_questions = assessment_data_loader.get_questions('phq9_scenario_followups', scenario=scenario)
        
        if data_questions:
            questions_list = []
            for i, q_data in enumerate(data_questions):
                formal_text = q_data.get('formal', q_data.get('question_formal', ''))
                conversational_text = q_data.get('conversational', q_data.get('question_conversational', ''))
                
                # Build question object with core question
                question_obj = {
                    'index': int(q_data.get('index', i)),
                    'formal': formal_text,
                    'conversational': _personalize_question_text(conversational_text, 'depression', scenario),
                    'scale': {
                        0: 'Not at all',
                        1: 'A few days',
                        2: 'More than half the days',
                        3: 'Nearly every day'
                    },
                    'followup': None  # Will be populated if scenario-specific follow-up exists
                }
                
                # Add scenario-specific follow-up if available
                if followup_questions:
                    for followup_data in followup_questions:
                        if followup_data.get('phq9_index') == i:
                            question_obj['followup'] = followup_data.get('followup', '')
                            break
                
                questions_list.append(question_obj)

            return jsonify({
                'success': True,
                'assessment_type': 'depression',
                'questions': questions_list,
                'total_questions': len(questions_list),
                'source': 'data/question_bank.json',
                'scenario': scenario,
                'has_followups': followup_questions is not None and len(followup_questions) > 0
            }), 200

        questions_list = []
        for i in range(len(PHQ9Questionnaire.QUESTIONS)):
            q_data = PHQ9Questionnaire.get_conversational_question(i, scenario=scenario)
            questions_list.append({
                'index': i,
                'formal': q_data['question_formal'],
                'conversational': q_data['question_conversational'],
                'scale': {
                    0: 'Not at all',
                    1: 'A few days',
                    2: 'More than half the days',
                    3: 'Nearly every day'
                },
                'followup': None
            })
        
        return jsonify({
            'success': True,
            'assessment_type': 'depression',
            'questions': questions_list,
            'total_questions': len(questions_list),
            'scenario': scenario,
            'has_followups': False
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assessment/conversational/pcl5', methods=['GET'])
def get_pcl5_conversational():
    """
    Get PCL-5 questionnaire - validated PTSD assessment  
    Based on: PCL-5 (PTSD Checklist for DSM-5)
    Covers all DSM-5 criteria: Intrusion, Avoidance, Negative Mood/Cognitions, Arousal
    """
    try:
        scenario = request.args.get('scenario', '')
        data_questions = assessment_data_loader.get_questions('pcl5')
        if data_questions:
            for question in data_questions:
                question['conversational'] = _personalize_question_text(
                    question.get('conversational', ''),
                    assessment_type='ptsd',
                    scenario=scenario
                )

            return jsonify({
                'success': True,
                'assessment_type': 'ptsd',
                'questions': data_questions,
                'total_questions': len(data_questions),
                'scale': {
                    0: 'Not at all',
                    1: 'A little bit',
                    2: 'Moderately',
                    3: 'Quite a bit',
                    4: 'Extremely'
                },
                'instrument': 'PCL-5 (PTSD Checklist for DSM-5)',
                'dsm5_criteria': ['Intrusion (B)', 'Avoidance (C)', 'Negative Cognitions & Mood (D)', 'Hyperarousal (E)'],
                'cutoff_score': {
                    'minimal': {'min': 0, 'max': 19},
                    'mild': {'min': 20, 'max': 35},
                    'moderate': {'min': 36, 'max': 51},
                    'severe': {'min': 52, 'max': 80}
                },
                'source': 'data/question_bank.json'
            }), 200

        pcl5_questions = [
            {
                'index': 0,
                'dsm5_criterion': 'Intrusion B1',
                'category': 'Intrusive Memories',
                'conversational': 'Have you been experiencing unwanted memories or flashbacks of the traumatic event that come back to you even when you don\'t want them to?',
                'formal': 'Repeated, involuntary, and intrusive distressing memories',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'A few times',
                    2: 'Several times a week',
                    3: 'Daily or almost daily'
                }
            },
            {
                'index': 1,
                'dsm5_criterion': 'Intrusion B2',
                'category': 'Nightmares',
                'conversational': 'Have you had nightmares related to the traumatic event? How often have you been waking up distressed?',
                'formal': 'Repeated disturbing dreams related to the traumatic event',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'A few times',
                    2: 'Several times a week',
                    3: 'Daily or most nights'
                }
            },
            {
                'index': 2,
                'dsm5_criterion': 'Intrusion B3',
                'category': 'Dissociative Reactions',
                'conversational': 'Do you ever feel like the traumatic event is happening again right now - like you\'re back in that situation? These are called dissociative or flashback episodes.',
                'formal': 'Dissociative reactions (flashbacks) where you feel the event is occurring again',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'A few times',
                    2: 'Several times',
                    3: 'Very often'
                }
            },
            {
                'index': 3,
                'dsm5_criterion': 'Intrusion B4',
                'category': 'Emotional Distress with Reminders',
                'conversational': 'When you encounter reminders of the traumatic event - like certain places, people, or things - how distressed do you become emotionally?',
                'formal': 'Intense or prolonged psychological distress at reminder cues',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'Slightly distressed',
                    2: 'Moderately distressed',
                    3: 'Extremely distressed'
                }
            },
            {
                'index': 4,
                'dsm5_criterion': 'Intrusion B5',
                'category': 'Physical Reactions to Reminders',
                'conversational': 'Do you experience physical reactions like sweating, heart racing, or shakiness when something reminds you of the trauma?',
                'formal': 'Marked physiological reactivity to trauma reminders',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'Mild physical reaction',
                    2: 'Moderate physical reaction',
                    3: 'Severe physical reaction'
                }
            },
            {
                'index': 5,
                'dsm5_criterion': 'Avoidance C1',
                'category': 'Avoidance of Thoughts/Feelings',
                'conversational': 'Do you try to push away thoughts or feelings related to the trauma? Do you try not to think about or feel the emotions?',
                'formal': 'Avoidance of distressing trauma-related thoughts or feelings',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'A little',
                    2: 'Quite a bit',
                    3: 'Extremely - constant effort to avoid'
                }
            },
            {
                'index': 6,
                'dsm5_criterion': 'Avoidance C2',
                'category': 'Avoidance of Reminders',
                'conversational': 'Do you avoid places, people, activities, or situations that remind you of the trauma? How much does this avoidance affect your life?',
                'formal': 'Avoidance of external reminders of the trauma',
                'severity_indicators': {
                    0: 'Nothing',
                    1: 'Mild avoidance',
                    2: 'Moderate avoidance limiting activities',
                    3: 'Extreme avoidance that significantly limits life'
                }
            },
            {
                'index': 7,
                'dsm5_criterion': 'Negative Mood D1',
                'category': 'Inability to Remember Key Trauma Details',
                'conversational': 'Can you remember important details about what happened? Or do you have a hard time remembering or gaps in your memory?',
                'formal': 'Inability to recall important aspects of the traumatic event',
                'severity_indicators': {
                    0: 'Remember clearly',
                    1: 'Slight memory gaps',
                    2: 'Moderate difficulty remembering',
                    3: 'Unable to remember key details'
                }
            },
            {
                'index': 8,
                'dsm5_criterion': 'Negative Mood D2',
                'category': 'Negative Beliefs About Self/World',
                'conversational': 'Since the trauma, have you developed negative beliefs about yourself or the world? Like thinking the world is completely dangerous, or that you are fundamentally flawed?',
                'formal': 'Persistent negative beliefs or expectations about self or the world',
                'severity_indicators': {
                    0: 'Not at all',
                    1: 'A little - occasional negative thoughts',
                    2: 'Quite a bit - frequent negative beliefs',
                    3: 'Extremely - pervasive negative worldview'
                }
            },
            {
                'index': 9,
                'dsm5_criterion': 'Negative Mood D3',
                'category': 'Blame of Self or Others',
                'conversational': 'Do you blame yourself for what happened or what you had to do to survive? Or do you blame others for the trauma?',
                'formal': 'Persistent, exaggerated self-blame or blame of others',
                'severity_indicators': {
                    0: 'Don\'t blame myself or others',
                    1: 'Slight self or other-blame',
                    2: 'Moderate blame affecting emotions',
                    3: 'Intense blame - preoccupied with fault'
                }
            },
            {
                'index': 10,
                'dsm5_criterion': 'Negative Mood D4',
                'category': 'Negative Emotions (Fear/Anger/Guilt)',
                'conversational': 'What emotions come up most? Are you experiencing a lot of fear, anger, guilt, shame, or other intense emotions related to the trauma?',
                'formal': 'Persistent negative emotional state (fear, anger, guilt, shame)',
                'severity_indicators': {
                    0: 'Not intensely',
                    1: 'Mild negative emotions',
                    2: 'Moderate emotional disturbance',
                    3: 'Intense emotions dominating your experience'
                }
            },
            {
                'index': 11,
                'dsm5_criterion': 'Negative Mood D5',
                'category': 'Loss of Interest in Activities',
                'conversational': 'Have you lost interest in activities you used to enjoy? Are there things you don\'t want to do anymore because of the trauma?',
                'formal': 'Markedly diminished interest in significant activities',
                'severity_indicators': {
                    0: 'Still interested in activities',
                    1: 'Slight loss of interest',
                    2: 'Moderate loss of interest',
                    3: 'No interest in most activities'
                }
            },
            {
                'index': 12,
                'dsm5_criterion': 'Negative Mood D6',
                'category': 'Detachment from Others',
                'conversational': 'Do you feel distant or cut off from people? Have you withdrawn from friends or family? Do you feel emotionally numb around others?',
                'formal': 'Persistent feeling of detachment or estrangement from others',
                'severity_indicators': {
                    0: 'Feel connected to others',
                    1: 'Slight detachment',
                    2: 'Noticeable distance from relationships',
                    3: 'Severe isolation and detachment'
                }
            },
            {
                'index': 13,
                'dsm5_criterion': 'Negative Mood D7',
                'category': 'Inability to Experience Positive Emotions',
                'conversational': 'Can you feel happiness, love, or other positive emotions? Or do you feel emotionally numb and unable to feel good things?',
                'formal': 'Persistent inability to experience positive emotions',
                'severity_indicators': {
                    0: 'Can feel positive emotions',
                    1: 'Reduced positive emotions',
                    2: 'Significant emotional numbness',
                    3: 'Unable to feel happiness or joy'
                }
            },
            {
                'index': 14,
                'dsm5_criterion': 'Hyperarousal E1',
                'category': 'Hypervigilance',
                'conversational': 'Do you find yourself constantly on guard or looking out for danger? Are you very aware of potential threats in your environment?',
                'formal': 'Heightened sense of current threat (hypervigilance)',
                'severity_indicators': {
                    0: 'Not vigilant',
                    1: 'Slightly more aware than before',
                    2: 'Noticeably more on guard',
                    3: 'Constantly scanning for danger'
                }
            },
            {
                'index': 15,
                'dsm5_criterion': 'Hyperarousal E2',
                'category': 'Exaggerated Startle Response',
                'conversational': 'Are you jumpy or easily startled by sudden noises or movements? Do you have strong startle reactions?',
                'formal': 'Exaggerated startle response',
                'severity_indicators': {
                    0: 'Normal startle',
                    1: 'Slightly more jumpy',
                    2: 'Noticeably easily startled',
                    3: 'Severely startled by minor stimuli'
                }
            },
            {
                'index': 16,
                'dsm5_criterion': 'Hyperarousal E3',
                'category': 'Irritability or Aggressive Behavior',
                'conversational': 'Have you been more irritable or aggressive than before? Do you have angry outbursts or react aggressively to others?',
                'formal': 'Irritability or aggressive behavior',
                'severity_indicators': {
                    0: 'Not irritable',
                    1: 'Slightly more irritable',
                    2: 'Noticeably irritable with occasional outbursts',
                    3: 'Frequent anger and aggressive reactions'
                }
            },
            {
                'index': 17,
                'dsm5_criterion': 'Hyperarousal E4',
                'category': 'Reckless or Self-Destructive Behavior',
                'conversational': 'Since the trauma, have you engaged in reckless activities, substance use, or anything self-destructive? Have your risk behaviors changed?',
                'formal': 'Reckless or self-destructive behavior',
                'severity_indicators': {
                    0: 'No reckless behavior',
                    1: 'Occasional risky behavior',
                    2: 'Noticeable increase in risk-taking',
                    3: 'Frequent dangerous or self-harm behaviors'
                }
            },
            {
                'index': 18,
                'dsm5_criterion': 'Hyperarousal E5',
                'category': 'Hyperarousal/Concentration Problems',
                'conversational': 'Are you having difficulty concentrating or focusing? Is it hard to stay on task? Do you have reduced attention span?',
                'formal': 'Problems with concentration',
                'severity_indicators': {
                    0: 'No concentration problems',
                    1: 'Occasional difficulty',
                    2: 'Noticeable concentration difficulty',
                    3: 'Severe concentration impairment'
                }
            },
            {
                'index': 19,
                'dsm5_criterion': 'Hyperarousal E6',
                'category': 'Sleep Disturbance',
                'conversational': 'How is your sleep affected? Do you have trouble falling asleep, staying asleep, or having restful sleep?',
                'formal': 'Sleep disturbance',
                'severity_indicators': {
                    0: 'No sleep problems',
                    1: 'Occasional sleep issues',
                    2: 'Noticeable sleep disturbance',
                    3: 'Severe sleep problems every night'
                }
            }
        ]

        for question in pcl5_questions:
            question['conversational'] = _personalize_question_text(
                question['conversational'],
                assessment_type='ptsd',
                scenario=scenario
            )
        
        return jsonify({
            'success': True,
            'assessment_type': 'ptsd',
            'questions': pcl5_questions,
            'total_questions': len(pcl5_questions),
            'scale': {
                0: 'Not at all',
                1: 'A little bit',
                2: 'Moderately',
                3: 'Quite a bit',
                4: 'Extremely'
            },
            'instrument': 'PCL-5 (PTSD Checklist for DSM-5)',
            'dsm5_criteria': ['Intrusion (B)', 'Avoidance (C)', 'Negative Cognitions & Mood (D)', 'Hyperarousal (E)'],
            'cutoff_score': {
                'minimal': {'min': 0, 'max': 19},
                'mild': {'min': 20, 'max': 35},
                'moderate': {'min': 36, 'max': 51},
                'severe': {'min': 52, 'max': 80}
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assessment/conversational/depression-detailed', methods=['GET'])
def get_depression_detailed():
    """
    Get detailed depression assessment using validated instruments
    Based on: BDI-II, CES-D, HAM-D (Hamilton Depression Rating Scale)
    Focuses on cognitive, behavioral, and somatic symptoms
    """
    try:
        scenario = request.args.get('scenario', '')
        data_questions = assessment_data_loader.get_questions('depression_detailed')
        if data_questions:
            for question in data_questions:
                question['conversational'] = _personalize_question_text(
                    question.get('conversational', ''),
                    assessment_type='depression',
                    scenario=scenario
                )

            return jsonify({
                'success': True,
                'assessment_type': 'depression_detailed',
                'questions': data_questions,
                'total_questions': len(data_questions),
                'scale': {
                    0: 'Not at all',
                    1: 'A little',
                    2: 'Quite a bit',
                    3: 'Very much'
                },
                'instruments_used': ['BDI-II', 'CES-D', 'HAM-D'],
                'note': 'Questions loaded from data folder and derived from validated depression instruments',
                'source': 'data/question_bank.json'
            }), 200

        depression_detailed_questions = [
            {
                'index': 0,
                'category': 'Cognitive Symptoms - Future Outlook',
                'conversational': 'When you think about your future, what comes to mind? Do you see yourself being able to overcome challenges, or does the future feel hopeless and uncertain?',
                'formal': 'Hopelessness and negative expectations',
                'instrument': 'BDI-II',
                'reasons': ['Hopelessness about future', 'Pessimistic thinking', 'Helplessness', 'Worthlessness'],
                'severity_indicators': {
                    0: 'Hopeful about future',
                    1: 'Some doubts but generally hopeful',
                    2: 'Alternating between hope and hopelessness',
                    3: 'Completely hopeless, sees no future'
                }
            },
            {
                'index': 1,
                'category': 'Shame & Self-Blame',
                'conversational': 'Do you find yourself being very critical of yourself? Do you blame yourself for things that have happened, even when it might not be entirely your fault?',
                'formal': 'Shame, self-blame, and excessive guilt',
                'instrument': 'HAM-D/BDI-II',
                'reasons': ['Excessive guilt', 'Self-blame', 'Shame and self-criticism', 'Feeling worthless'],
                'severity_indicators': {
                    0: 'Normal self-assessment',
                    1: 'Occasional self-blame',
                    2: 'Frequent self-blame and guilt',
                    3: 'Pervasive guilt and self-loathing'
                }
            },
            {
                'index': 2,
                'category': 'Concentration & Decision Making',
                'conversational': 'How is your concentration these days? Can you focus on tasks, or do you find your mind wandering? Are decisions harder to make?',
                'formal': 'Concentration difficulty and reduced decisiveness',
                'instrument': 'HAM-D/PHQ-9',
                'reasons': ['Poor concentration', 'Difficulty making decisions', 'Cognitive fog', 'Mental slowness'],
                'severity_indicators': {
                    0: 'Normal concentration and decision-making',
                    1: 'Occasional difficulty concentrating',
                    2: 'Obvious concentration difficulty',
                    3: 'Cannot concentrate or make decisions'
                }
            },
            {
                'index': 3,
                'category': 'Physical Activity & Motivation',
                'conversational': 'How has your motivation been? Do you have the drive to do things, or does everything feel exhausting and pointless? How is your activity level?',
                'formal': 'Loss of motivation and reduced physical activity',
                'instrument': 'CES-D/HAM-D',
                'reasons': ['Loss of motivation', 'Fatigue and low energy', 'Inactivity', 'Psychomotor retardation'],
                'severity_indicators': {
                    0: 'Normal motivation and activity',
                    1: 'Mild decrease in activity',
                    2: 'Noticeable decrease in motivation',
                    3: 'Unable to complete daily activities'
                }
            },
            {
                'index': 4,
                'category': 'Sleep Disturbance',
                'conversational': 'How has your sleep been? Are you waking up too early, having trouble falling asleep, or sleeping too much? How rested do you feel?',
                'formal': 'Sleep disturbance patterns (insomnia or hypersomnia)',
                'instrument': 'HAM-D/BDI-II',
                'reasons': ['Early morning awakening', 'Sleep onset insomnia', 'Hypersomnia', 'Unrefreshing sleep'],
                'severity_indicators': {
                    0: 'Normal sleep patterns',
                    1: 'Occasional sleep difficulties',
                    2: 'Definite sleep disturbance',
                    3: 'Severe sleep disturbance affecting function'
                }
            },
            {
                'index': 5,
                'category': 'Appetite & Weight Changes',
                'conversational': 'Have you noticed changes in your appetite? Are you eating more or less? Has your weight changed noticeably?',
                'formal': 'Appetite and weight changes',
                'instrument': 'HAM-D/PHQ-9',
                'reasons': ['Decreased appetite and weight loss', 'Increased appetite and weight gain', 'Anhedonia affecting eating', 'No energy for self-care'],
                'severity_indicators': {
                    0: 'No appetite changes',
                    1: 'Slight appetite decrease',
                    2: 'Definite appetite change',
                    3: 'Severe appetite change with weight fluctuation'
                }
            },
            {
                'index': 6,
                'category': 'Anhedonia - Loss of Pleasure',
                'conversational': 'What activities brought you joy before? Do those things still interest you, or have you lost interest in them? Do you experience pleasure anymore?',
                'formal': 'Anhedonia and loss of interest in activities',
                'instrument': 'BDI-II/CES-D',
                'reasons': ['Loss of interest in hobbies', 'Can\'t enjoy things anymore', 'Social withdrawal', 'Emotional numbness'],
                'severity_indicators': {
                    0: 'Normal interest in activities',
                    1: 'Slight loss of pleasure',
                    2: 'Marked loss of interest',
                    3: 'Complete anhedonia'
                }
            },
            {
                'index': 7,
                'category': 'Irritability & Anger',
                'conversational': 'Have you been feeling more irritable or angry than usual? Do small things bother you more? Have others commented on changes in your mood?',
                'formal': 'Irritability and increased anger/frustration',
                'instrument': 'BDI-II/CES-D',
                'reasons': ['Increased irritability', 'Excessive anger', 'Impatience with others', 'Difficulty with frustration'],
                'severity_indicators': {
                    0: 'Normal mood stability',
                    1: 'Slightly more irritable',
                    2: 'Noticeably irritable',
                    3: 'Severely irritable or angry'
                }
            },
            {
                'index': 8,
                'category': 'Suicidal Ideation - Critical Safety',
                'conversational': 'I need to ask directly: Have you had thoughts that life isn\'t worth living, or that others would be better off without you? Any thoughts about harming yourself?',
                'formal': 'Suicidal ideation and self-harm thoughts',
                'instrument': 'PHQ-9/HAM-D',
                'reasons': ['Passive death wish', 'Active suicidal ideation', 'Specific plan', 'Suicide attempt'],
                'suicide_risk': True,
                'severity_indicators': {
                    0: 'No suicidal thoughts',
                    1: 'Occasional passive thoughts',
                    2: 'Persistent passive or vague active thoughts',
                    3: 'Active suicidal ideation with plan or intent'
                }
            }
        ]

        for question in depression_detailed_questions:
            question['conversational'] = _personalize_question_text(
                question['conversational'],
                assessment_type='depression',
                scenario=scenario
            )
        
        return jsonify({
            'success': True,
            'assessment_type': 'depression_detailed',
            'questions': depression_detailed_questions,
            'total_questions': len(depression_detailed_questions),
            'scale': {
                0: 'Not at all',
                1: 'A little',
                2: 'Quite a bit',
                3: 'Very much'
            },
            'instruments_used': ['BDI-II', 'CES-D', 'HAM-D'],
            'note': 'Questions derived from validated depression assessment instruments'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assessment/divisions', methods=['GET'])
def get_assessment_divisions():
    """Return participant divisions from train/dev/test/full split CSV files."""
    try:
        include_ids = request.args.get('include_ids', 'false').lower() == 'true'
        divisions = assessment_data_loader.get_divisions(include_ids=include_ids)

        return jsonify({
            'success': True,
            'source': [
                'train_split_Depression_AVEC2017.csv',
                'dev_split_Depression_AVEC2017.csv',
                'test_split_Depression_AVEC2017.csv',
                'full_test_split.csv'
            ],
            'divisions': divisions
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assessment/conversational/scenario-identify', methods=['POST'])
def identify_assessment_scenario():
    """Identify the scenario based on user's description"""
    try:
        data = request.json
        user_description = data.get('description', '')
        assessment_type = data.get('assessment_type', 'depression')
        
        if not user_description:
            return jsonify({'error': 'No description provided'}), 400
        
        is_ptsd = assessment_type.lower() == 'ptsd'
        scenario = identify_scenario(user_description, is_ptsd=is_ptsd)
        
        return jsonify({
            'success': True,
            'assessment_type': assessment_type,
            'scenario': scenario,
            'description': user_description
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/assessment/conversational/submit', methods=['POST'])
def submit_conversational_responses():
    """Submit multiple text responses from conversational assessment"""
    try:
        data = request.json
        responses_text = data.get('responses', [])
        questions = data.get('questions', None)
        assessment_type = data.get('assessment_type', 'depression')
        session_id = data.get('session_id')
        
        if not responses_text:
            return jsonify({'error': 'No responses provided'}), 400
        
        # Convert text responses to scales
        scales = TextInterpreter.batch_interpret_responses(responses_text, questions)
        
        # Calculate assessment
        total_score = sum(scales)
        
        # Determine severity
        if assessment_type.lower() == 'depression':
            if total_score < 5:
                severity = 'Minimal'
            elif total_score < 10:
                severity = 'Mild'
            elif total_score < 15:
                severity = 'Moderate'
            elif total_score < 20:
                severity = 'Moderately Severe'
            else:
                severity = 'Severe'
        else:  # PTSD
            if total_score < 10:
                severity = 'Minimal'
            elif total_score < 20:
                severity = 'Mild'
            elif total_score < 35:
                severity = 'Moderate'
            elif total_score < 50:
                severity = 'Severe'
            else:
                severity = 'Severe PTSD'
        
        # Update session if provided
        if session_id and session_id in questionnaire_sessions:
            session = questionnaire_sessions[session_id]
            session.responses = scales
            session.completed = True
            session.score_data = {
                'total_score': total_score,
                'severity': severity,
                'assessment_type': assessment_type
            }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'assessment_type': assessment_type,
            'total_score': total_score,
            'severity': severity,
            'scales': scales,
            'responses_analyzed': len(scales)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# VIDEO/FACIAL EXPRESSION ANALYSIS ROUTES
# ============================================================================

@app.route('/api/video/analyze/<int:participant_id>', methods=['POST'])
def analyze_video(participant_id):
    """Analyze video for participant"""
    try:
        video_results = video_analyzer.analyze_video(participant_id)
        
        # Extract features for hybrid model
        features = feature_extractor.extract_participant_features(participant_id)
        
        return jsonify({
            'success': True,
            'participant_id': participant_id,
            'video_analysis': video_results,
            'features_extracted': 'action_units' in features
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/microexpression/detect/<int:participant_id>', methods=['POST'])
def detect_microexpressions(participant_id):
    """Detect micro-expressions for participant"""
    try:
        import numpy as np
        
        # Load AU data
        from pathlib import Path
        participant_dir = Path("data") / str(participant_id)
        au_file = participant_dir / f"{participant_id}_CLNF_AUs.txt"
        
        if not au_file.exists():
            return jsonify({'error': 'No audio data found'}), 404
        
        au_data = np.loadtxt(au_file, skiprows=0)
        if len(au_data.shape) == 1:
            au_data = au_data.reshape(-1, 1)
        
        # Create placeholder frame times
        n_frames = au_data.shape[0]
        fps = 30  # Assume 30 fps
        frame_times = np.arange(n_frames) / fps
        
        # Detect micro-expressions
        detections = microexpression_detector.detect_microexpressions(au_data, frame_times)
        
        summary = microexpression_detector.get_micro_expression_summary()
        
        return jsonify({
            'success': True,
            'participant_id': participant_id,
            'detections': detections,
            'summary': summary
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/video/upload-and-analyze', methods=['POST'])
def upload_and_analyze_video():
    """
    Upload video blob and analyze for micro-expressions
    Processes recorded video from assessment and detects real facial behaviors
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'unknown')
        assessment_type = data.get('assessment_type', 'depression')
        responses = data.get('responses', [])
        
        print(f"\n{'='*60}")
        print(f"[{session_id}] VIDEO ANALYSIS REQUEST")
        print(f"{'='*60}")
        print(f"Session ID: {session_id}")
        print(f"Assessment Type: {assessment_type}")
        print(f"Number of responses: {len(responses)}")
        
        # Handle video data (can be base64 encoded or raw bytes)
        video_data = data.get('video_data')
        if not video_data:
            print(f"[{session_id}] ERROR: No video data in request")
            return jsonify({
                'success': False,
                'error': 'No video data provided',
                'fallback': True
            }), 400
        
        print(f"[{session_id}] Video data size: {len(str(video_data))} chars")
        
        if isinstance(video_data, str):
            # If it's base64, decode it
            import base64
            try:
                video_blob = base64.b64decode(video_data)
                print(f"[{session_id}] ✓ Base64 decoded: {len(video_blob)} bytes ({len(video_blob)/1024:.2f} KB)")
            except Exception as e:
                print(f"[{session_id}] ERROR decoding base64: {e}")
                video_blob = video_data.encode()
        else:
            video_blob = video_data
            print(f"[{session_id}] Video blob received: {len(video_blob)} bytes")
        
        # Validate video blob size
        if len(video_blob) < 100:
            print(f"[{session_id}] ERROR: Video blob too small ({len(video_blob)} bytes)")
            return jsonify({
                'success': False,
                'error': 'Video data too small',
                'fallback': True
            }), 400
        
        # Process video to extract facial features
        print(f"[{session_id}] Starting video processing...")
        video_analysis = video_processor.process_video_blob(
            video_blob, 
            session_id, 
            responses
        )
        
        print(f"[{session_id}] Video processing complete")
        print(f"  - Success: {video_analysis.get('success')}")
        print(f"  - Frames: {video_analysis.get('total_frames', 0)}")
        print(f"  - Duration: {video_analysis.get('duration_seconds', 0):.2f}s")
        print(f"  - FPS: {video_analysis.get('fps', 0):.1f}")
        print(f"  - Faces detected: {video_analysis.get('facial_features_detected', False)}")
        print(f"  - Face detection rate: {video_analysis.get('face_detection_rate', 0):.1%}")
        
        if not video_analysis.get('success'):
            print(f"[{session_id}] WARNING: Video processing failed - using fallback estimates")
            print(f"  Error: {video_analysis.get('error')}")
            return jsonify({
                'success': False,
                'error': video_analysis.get('error'),
                'fallback': True,  # Frontend will use fallback values
                'video_analysis': {
                    'total_frames': video_analysis.get('total_frames', 0),
                    'duration_seconds': video_analysis.get('duration_seconds', 0),
                    'fps': video_analysis.get('fps', 0),
                    'facial_features_detected': False
                }
            }), 200
        
        # Extract AU patterns from video
        au_patterns = video_analysis.get('au_patterns', [])
        
        if isinstance(au_patterns, list):
            au_patterns = np.array(au_patterns)
        
        print(f"[{session_id}] AU patterns shape: {au_patterns.shape if hasattr(au_patterns, 'shape') else len(au_patterns)}")

        # Predict depression scale from AU patterns using trained video model (.pkl)
        video_model_prediction = None
        if assessment_type == 'depression' and len(au_patterns) > 0:
            video_model_prediction = hybrid_model.predict_video_scale_from_au_patterns(au_patterns)
            print(
                f"[{session_id}] Video model prediction: used={video_model_prediction.get('used_model')}, "
                f"scale_0_27={video_model_prediction.get('scale_0_27')}, "
                f"confidence={video_model_prediction.get('confidence', 0):.2f}"
            )
        
        response_microexpressions = {}
        
        # Analyze micro-expressions for each response if we have AU data
        if len(au_patterns) > 0:
            try:
                # Create frame times
                n_frames = len(au_patterns)
                fps = video_analysis.get('fps', 30)
                if fps == 0:
                    fps = 30
                frame_times = np.arange(n_frames) / fps
                
                print(f"[{session_id}] Analyzing {len(responses)} responses with {n_frames} frames at {fps} FPS")
                
                # Detect micro-expressions per response
                for idx, response in enumerate(responses):
                    question_id = f"q_{idx + 1}"
                    
                    # For multi-question assessment, divide the AU data proportionally
                    # Use a 2-second window for each question
                    window_size = int(2 * fps)  # 2-second window
                    start_idx = min(idx * window_size, len(au_patterns) - 1)
                    end_idx = min((idx + 1) * window_size, len(au_patterns))
                    
                    if start_idx < len(au_patterns):
                        au_segment = au_patterns[start_idx:end_idx]
                        frame_times_segment = frame_times[start_idx:end_idx] - frame_times[start_idx]
                        
                        if len(au_segment) > 0 and len(frame_times_segment) > 0:
                            print(f"[{session_id}]   Response {idx+1}: Analyzing frames {start_idx}-{end_idx} ({len(au_segment)} frames)")
                            
                            detections = microexpression_detector.detect_microexpressions(
                                au_segment,
                                frame_times_segment,
                                question_id=question_id,
                                time_window=2.0
                            )
                            
                            print(f"[{session_id}]   Response {idx+1}: Found {len(detections)} micro-expressions with probabilities")
                            if len(detections) > 0:
                                for det in detections[:3]:  # Show first 3
                                    print(f"[{session_id}]     - {det.get('expression', 'unknown')}: {det.get('confidence', 0):.1%}")
                            
                            # Format detections for response
                            response_microexpressions[f"response_{idx+1}"] = {
                                'response_text': response.get('text', ''),
                                'response_scale': response.get('scale', 0),
                                'detections': detections,
                                'detection_count': len(detections),
                                'analysis_type': 'video_real',
                                'probabilities': [d.get('confidence', 0.0) for d in detections]
                            }
            
            except Exception as e:
                print(f"[{session_id}] Micro-expression detection error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[{session_id}] ✓ Analysis complete - {len(response_microexpressions)} responses with real video data")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'video_analysis': {
                'total_frames': video_analysis.get('total_frames', 0),
                'duration_seconds': video_analysis.get('duration_seconds', 0),
                'fps': video_analysis.get('fps', 0),
                'facial_features_detected': video_analysis.get('facial_features_detected', False),
                'face_detection_rate': video_analysis.get('face_detection_rate', 0.0)
            },
            'response_microexpressions': response_microexpressions,
            'video_model_prediction': video_model_prediction,
            'analysis_type': 'video_real',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"[session_id] Video upload error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback': True
        }), 500


# ============================================================================
# TEXT ANALYSIS ROUTES
# ============================================================================

@app.route('/api/text/analyze', methods=['POST'])
def analyze_text():
    """Analyze transcript text"""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript:
            return jsonify({'error': 'No transcript provided'}), 400
        
        text_analysis = text_analyzer.analyze_transcript(transcript)
        
        return jsonify({
            'success': True,
            'analysis': text_analysis
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DEPRESSION PREDICTION ROUTES
# ============================================================================

@app.route('/api/predict/depression/<int:participant_id>', methods=['POST'])
def predict_depression(participant_id):
    """Predict depression severity from extracted features"""
    try:
        # Extract participant features
        features = feature_extractor.extract_participant_features(participant_id)
        
        if 'error' in features:
            return jsonify(features), 404
        
        # Get hybrid model prediction
        prediction = hybrid_model.predict_depression_severity(features)
        
        return jsonify({
            'success': True,
            'participant_id': participant_id,
            'prediction': prediction,
            'components': prediction['component_scores']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# INTEGRATED ASSESSMENT ROUTES
# ============================================================================

@app.route('/api/assessment/fused/<session_id>', methods=['POST'])
def fused_assessment(session_id):
    """Generate fused multimodal assessment"""
    try:
        data = request.get_json()
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get PHQ score
        phq_score = {
            'total_score': session.get('phq_total_score', 0),
            'suicide_risk': session.get('suicide_risk', False)
        }
        
        # Get video analysis
        video_analysis = data.get('video_analysis', {})
        
        # Get text analysis
        text_analysis = data.get('text_analysis', {})
        
        # Perform fusion
        fused_result = fusion_engine.fuse_assessment(
            phq_score=phq_score,
            video_analysis=video_analysis,
            text_analysis=text_analysis
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'assessment': fused_result
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# REPORT GENERATION ROUTES
# ============================================================================

@app.route('/api/report/generate/<session_id>', methods=['POST'])
def generate_report(session_id):
    """Generate comprehensive assessment report"""
    try:
        data = request.get_json()
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Prepare session data for report
        report_data = {
            'session_id': session_id,
            'participant_id': data.get('participant_id'),
            'phq_data': {
                'total_score': session.get('phq_total_score', 0),
                'severity': session.get('phq_severity', 'unknown'),
                'suicide_risk': session.get('suicide_risk', False)
            },
            'video_analysis': data.get('video_analysis', {}),
            'text_analysis': data.get('text_analysis', {}),
            'integrated_assessment': data.get('integrated_assessment', {})
        }
        
        # Generate report
        report = report_generator.generate_full_report(report_data)
        
        # Save report if path provided
        if 'export_path' in data:
            report_generator.export_report_json(report, data['export_path'])
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'report_id': report['report_id'],
            'report': report
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# HEALTH CHECK ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'running',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'active_sessions': session_manager.get_all_sessions_count()
    }), 200


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'title': 'Multimodal Depression Screening System',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/session/create': 'Create new session',
            '/api/questionnaire/questions': 'Get PHQ-9 questions',
            '/api/questionnaire/<session_id>/submit': 'Submit questionnaire responses',
            '/api/assessment/conversational/interpret': 'Interpret text response to scale',
            '/api/assessment/conversational/phq9': 'Get PHQ-9 conversational format',
            '/api/assessment/conversational/pcl5': 'Get PCL-5 questions',
            '/api/assessment/conversational/scenario-identify': 'Identify assessment scenario',
            '/api/assessment/conversational/submit': 'Submit conversational responses',
            '/api/video/analyze/<participant_id>': 'Analyze video',
            '/api/microexpression/detect/<participant_id>': 'Detect micro-expressions',
            '/api/text/analyze': 'Analyze text transcript',
            '/api/predict/depression/<participant_id>': 'Predict depression',
            '/api/assessment/fused/<session_id>': 'Generate fused assessment',
            '/api/report/generate/<session_id>': 'Generate final report'
        }
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run Flask development server
    print("Starting Multimodal Depression Screening System...")
    print("API Documentation available at http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)
