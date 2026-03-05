"""
Text Interpreter Service
Converts natural language responses to assessment scales (0-3)
Uses pattern matching and sentiment analysis
"""
from typing import Tuple

class TextInterpreter:
    """Convert text responses to PHQ-9/PCL-5 scale responses"""
    
    # Intensity mapping
    INTENSITY_PATTERNS = {
        3: {  # Nearly every day / Very severe
            'keywords': [
                'always', 'constantly', 'absolutely', 'definitely', 'all the time',
                'every day', 'extremely', 'very much', 'completely', 'totally',
                'intensely', 'severely', 'badly', 'very badly', 'awful', 'terrible',
                'non-stop', 'continuously', 'without stop', 'can\'t stop'
            ],
            'negations': ['not', 'never', 'hardly']
        },
        2: {  # More than half the days / Moderate
            'keywords': [
                'often', 'frequently', 'regularly', 'quite a bit', 'most days',
                'usually', 'most of the time', 'a lot', 'significant', 'considerable',
                'substantial', 'marked', 'noticeable', 'pretty much', 'quite often'
            ],
            'negations': ['not really', 'not much']
        },
        1: {  # Several days / Mild
            'keywords': [
                'sometimes', 'occasionally', 'few', 'bit', 'little', 'some',
                'once in a while', 'here and there', 'now and then', 'at times',
                'from time to time', 'less often', 'somewhat', 'slightly'
            ],
            'negations': ['barely', 'hardly']
        },
        0: {  # Not at all / Not present
            'keywords': [
                'no', 'not', 'none', 'nope', 'nothing', 'not really',
                'not at all', 'none of that', 'no issues', 'fine', 'okay', 'alright',
                'doing well', 'doing good', 'normal', 'good', 'better', 'fine'
            ],
            'negations': []
        }
    }
    
    CONTEXT_MODIFIERS = {
        'depression_markers': {
            'sad': 1, 'depressed': 1.5, 'hopeless': 2, 'suicidal': 3,
            'empty': 1, 'numb': 1.5, 'worthless': 2, 'ashamed': 1.5,
            'guilty': 1, 'tired': 0.5, 'exhausted': 1.5, 'drained': 1,
            'unmotivated': 1, 'anxious': 0.5, 'worried': 0.5
        },
        'positive_indicators': {
            'good': -0.5, 'better': -1, 'great': -1, 'excellent': -1,
            'happy': -1.5, 'joyful': -1.5, 'fine': -0.5, 'okay': 0
        },
        'negation_words': [
            'not', 'no', 'never', 'hardly', 'barely', 'don\'t', 'doesn\'t', 'didn\'t'
        ]
    }
    
    @staticmethod
    def interpret_response(text: str, question_context: str = None) -> Tuple[int, str]:
        """
        Convert user text response to scale value (0-3)
        
        Args:
            text: User's natural language response
            question_context: The question asked (optional, for context)
            
        Returns:
            Tuple of (scale_value 0-3, interpretation_reason)
        """
        if not text or not isinstance(text, str):
            return 0, "No response provided"
        
        text_lower = text.lower().strip()
        
        # Check for direct affirmations/negations
        for scale, patterns in TextInterpreter.INTENSITY_PATTERNS.items():
            score = TextInterpreter._match_patterns(text_lower, patterns, scale)
            if score is not None:
                reason = TextInterpreter._get_interpretation(scale)
                return scale, reason
        
        # If no direct match, analyze sentiment
        return TextInterpreter._analyze_sentiment(text, question_context)
    
    @staticmethod
    def _match_patterns(text: str, patterns: dict, expected_scale: int) -> int:
        """Match intensity patterns in text"""
        keywords = patterns.get('keywords', [])
        negations = patterns.get('negations', [])
        
        # Check if any keyword is present
        keyword_found = any(kw in text for kw in keywords)
        
        if keyword_found:
            # Check if negated
            negation_found = any(neg in text for neg in negations)
            
            if negation_found:
                # If negated, return lower scale
                return max(0, expected_scale - 2)
            else:
                return expected_scale
        
        return None
    
    @staticmethod
    def _analyze_sentiment(text: str, context: str = None) -> Tuple[int, str]:
        """Analyze sentiment and context to determine scale"""
        text_lower = text.lower()
        
        score = 1  # Default to mild/moderate
        reasons = []
        
        # Look for depression-specific markers
        depression_words = TextInterpreter.CONTEXT_MODIFIERS['depression_markers']
        for word, modifier in depression_words.items():
            if word in text_lower:
                score += modifier
                reasons.append(f"Contains depression indicator: {word}")
        
        # Look for positive indicators
        positive_words = TextInterpreter.CONTEXT_MODIFIERS['positive_indicators']
        for word, modifier in positive_words.items():
            if word in text_lower:
                score += modifier
                reasons.append(f"Contains positive indicator: {word}")
        
        # Check for negations
        negation_words = TextInterpreter.CONTEXT_MODIFIERS['negation_words']
        for word in negation_words:
            if word in text_lower:
                score -= 0.5
                reasons.append(f"Contains negation: {word}")
        
        # Clamp to 0-3 range
        final_score = max(0, min(3, int(round(score))))
        reason = " | ".join(reasons) if reasons else "Sentiment-based assessment"
        
        return final_score, reason
    
    @staticmethod
    def _get_interpretation(scale: int) -> str:
        """Get human-friendly interpretation of scale"""
        interpretations = {
            0: "Not at all - no symptoms indicated",
            1: "A few days - mild symptoms",
            2: "More than half the days - moderate symptoms",
            3: "Nearly every day - severe symptoms"
        }
        return interpretations.get(scale, "Unable to assess")
    
    @staticmethod
    def batch_interpret_responses(texts: list[str], questions: list[str] = None) -> list[int]:
        """
        Convert multiple text responses to scales
        
        Args:
            texts: List of user responses
            questions: Optional list of questions for context
            
        Returns:
            List of scale values (0-3)
        """
        scales = []
        for i, text in enumerate(texts):
            context = questions[i] if questions and i < len(questions) else None
            scale, _ = TextInterpreter.interpret_response(text, context)
            scales.append(scale)
        return scales
    
    @staticmethod
    def validate_and_correct(scale: int) -> int:
        """Ensure scale is valid (0-3)"""
        return max(0, min(3, int(scale)))


# PCL-5 and Major Depression Scenario Keywords
DEPRESSION_SCENARIOS = {
    'postpartum': ['baby', 'pregnant', 'pregnancy', 'mother', 'newborn', 'postpartum'],
    'work': ['work', 'job', 'career', 'boss', 'stress', 'burnout', 'deadline'],
    'school': ['school', 'exam', 'test', 'grade', 'college', 'university', 'student'],
    'relationship': ['relationship', 'partner', 'breakup', 'divorce', 'spouse', 'family'],
    'loss': ['death', 'died', 'lost', 'loss', 'passed', 'miss', 'grief', 'funeral'],
    'general': ['sad', 'depressed', 'blue', 'down', 'no reason', 'just feel']
}

PTSD_SCENARIOS = {
    'accident': ['accident', 'car', 'crash', 'injury', 'injured', 'hospital'],
    'violence': ['assault', 'violence', 'attack', 'hit', 'abuse', 'domestic'],
    'military': ['military', 'combat', 'war', 'soldier', 'battle', 'deployed'],
    'loss': ['death', 'died', 'lost', 'loss', 'passed', 'suicide'],
    'disaster': ['disaster', 'earthquake', 'flood', 'fire', 'tornado', 'hurricane'],
    'other': ['trauma', 'traumatic', 'event', 'incident', 'experience']
}

def identify_scenario(text: str, is_ptsd: bool = False) -> str:
    """Identify the scenario based on user's text"""
    text_lower = text.lower()
    scenarios = PTSD_SCENARIOS if is_ptsd else DEPRESSION_SCENARIOS
    
    best_match = 'general' if not is_ptsd else 'other'
    best_score = 0
    
    for scenario, keywords in scenarios.items():
        match_count = sum(1 for keyword in keywords if keyword in text_lower)
        if match_count > best_score:
            best_score = match_count
            best_match = scenario
    
    return best_match
