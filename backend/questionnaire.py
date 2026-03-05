"""
PHQ-9 Depression Questionnaire Management
Handles scoring and interpretation of depression severity
"""

class PHQ9Questionnaire:
    """PHQ-9 Depression Severity Scale"""

    SCENARIO_CONTEXTS = {
        'postpartum': 'after your recent childbirth experience',
        'work': 'in relation to your work stress or burnout',
        'school': 'in relation to your academic pressure',
        'relationship': 'in relation to your relationship experience',
        'loss': 'in relation to your loss or grief experience',
        'general': 'in your recent day-to-day life'
    }
    
    # Original PHQ-9 questions (for scoring/documentation)
    QUESTIONS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking slowly, or being restless",
        "Thoughts that you would be better off dead"
    ]
    
    # Conversational versions of questions (what users actually see)
    CONVERSATIONAL_QUESTIONS = [
        "I noticed you haven't mentioned things you enjoy doing lately. Tell me - what kinds of activities used to make you happy, and are you still interested in them?",
        "How would you describe your overall mood these past two weeks? Has there been a shift in how you generally feel day to day?",
        "Sleep is so important. How have you been sleeping lately? Any changes in your sleep patterns?",
        "A lot of people mention feeling drained or running on empty. How's your energy level been? Do you feel like yourself?",
        "Sometimes mood changes can affect our appetite. How's your eating been? Any changes you've noticed?",
        "I'm curious - how do you typically view yourself? Have you been harder on yourself lately, or noticed any changes in how you see yourself?",
        "Do you find yourself able to focus and concentrate on things the way you normally do? Or has that been a bit more challenging?",
        "Some people notice changes in how they move or speak when things are tough. Have you felt different physically - maybe slower, or more restless?",
        "I want to ask directly - have there been any moments over the last two weeks when you wished things were different, or thought life might be better if you weren't here?"
    ]
    
    # Response scale labels (more friendly)
    RESPONSE_LABELS = {
        0: "Not at all",
        1: "A few days",
        2: "More than half the days",
        3: "Nearly every day"
    }
    
    SEVERITY_LEVELS = {
        (0, 4): "Minimal",
        (5, 9): "Mild",
        (10, 14): "Moderate",
        (15, 19): "Moderately Severe",
        (20, 27): "Severe"
    }
    
    @staticmethod
    def get_conversational_question(index: int, scenario: str | None = None) -> dict:
        """Get conversational version of a question"""
        if index < 0 or index >= len(PHQ9Questionnaire.CONVERSATIONAL_QUESTIONS):
            return {"error": "Invalid question index"}

        base_question = PHQ9Questionnaire.CONVERSATIONAL_QUESTIONS[index]
        scenario_context = PHQ9Questionnaire.SCENARIO_CONTEXTS.get((scenario or '').strip().lower())
        if scenario_context:
            personalized_question = f"{base_question} Thinking specifically {scenario_context}, how often has this happened?"
        else:
            personalized_question = base_question
        
        return {
            "index": index,
            "question_formal": PHQ9Questionnaire.QUESTIONS[index],
            "question_conversational": personalized_question,
            "response_options": PHQ9Questionnaire.RESPONSE_LABELS
        }
    
    @staticmethod
    def calculate_phq9_score(responses: list[int]) -> dict:
        """
        Calculate PHQ-9 score from responses (0-3 scale per question)
        
        Args:
            responses: List of 9 integers (0-3) representing answers
            
        Returns:
            Dict with score and severity level
        """
        if len(responses) != 9 or not all(0 <= r <= 3 for r in responses):
            raise ValueError("PHQ-9 requires 9 responses, each 0-3")
        
        total_score = sum(responses)
        severity = PHQ9Questionnaire._get_severity(total_score)
        
        return {
            "total_score": total_score,
            "severity": severity,
            "suicide_risk": responses[8] > 0  # Question 9 indicates suicide risk
        }
    
    @staticmethod
    def _get_severity(score: int) -> str:
        """Get severity level based on score"""
        for (min_score, max_score), level in PHQ9Questionnaire.SEVERITY_LEVELS.items():
            if min_score <= score <= max_score:
                return level
        return "Severe"
    
    @staticmethod
    def _interpret_phq_score(score: int) -> str:
        """Generate human-friendly interpretation of PHQ-9 score"""
        if score < 5:
            return "Your responses show minimal depressive symptoms. You're doing well - keep it up!"
        elif score < 10:
            return "You're experiencing some mild depressive symptoms. It might help to talk to someone or focus on self-care."
        elif score < 15:
            return "Your symptoms suggest moderate depression. Speaking with a professional could really help."
        elif score < 20:
            return "You're dealing with moderately severe depression. Professional support is recommended."
        else:
            return "Your responses indicate severe depression. Please reach out to a healthcare professional as soon as possible."
    
    @staticmethod
    def validate_responses(responses: list) -> tuple[bool, str]:
        """Validate questionnaire responses"""
        if not isinstance(responses, list):
            return False, "Responses must be a list"
        if len(responses) != 9:
            return False, f"Expected 9 responses, got {len(responses)}"
        if not all(isinstance(r, (int, float)) and 0 <= r <= 3 for r in responses):
            return False, "Each response must be 0-3"
        return True, "Valid"


class QuestionnaireSession:
    """Manages questionnaire state during a session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.responses = []
        self.completed = False
        self.score_data = None
    
    def add_response(self, question_idx: int, answer: int) -> bool:
        """Add single response"""
        if question_idx >= len(PHQ9Questionnaire.QUESTIONS):
            return False
        if not 0 <= answer <= 3:
            return False
        
        # Ensure list is large enough
        while len(self.responses) <= question_idx:
            self.responses.append(None)
        
        self.responses[question_idx] = answer
        return True
    
    def is_complete(self) -> bool:
        """Check if all responses received"""
        return len(self.responses) == 9 and all(r is not None for r in self.responses)
    
    def finalize(self) -> dict:
        """Calculate final score and mark complete"""
        if not self.is_complete():
            return {"error": "Questionnaire incomplete"}
        
        self.score_data = PHQ9Questionnaire.calculate_phq9_score(self.responses)
        self.completed = True
        return self.score_data
    
    def get_summary(self) -> dict:
        """Get questionnaire summary"""
        return {
            "session_id": self.session_id,
            "responses": self.responses,
            "completed": self.completed,
            "score_data": self.score_data
        }
