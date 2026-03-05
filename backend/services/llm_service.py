"""
LLM-based text analysis service
Performs sentiment, semantic, and linguistic analysis of transcripts
"""
from typing import Dict, List, Optional
import re

class LLMTextAnalyzer:
    """Analyzes linguistic and semantic features from interview transcripts"""
    
    def __init__(self):
        self.depression_keywords = {
            'negative': [
                'sad', 'depressed', 'hopeless', 'worthless', 'guilty', 'failure',
                'lonely', 'empty', 'pain', 'suffer', 'hate', 'death', 'suicide',
                'bad', 'terrible', 'awful', 'horrible'
            ],
            'fatigue': [
                'tired', 'exhausted', 'fatigue', 'energy', 'weak', 'lazy',
                'sluggish', 'drain', 'burn out'
            ],
            'cognitive': [
                'concentrate', 'focus', 'memory', 'confused', 'concentrate',
                'think', 'clear', 'mind'
            ]
        }
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Comprehensive linguistic analysis of interview transcript
        
        Args:
            transcript: Full interview transcript text
            
        Returns:
            Dictionary with linguistic and semantic features
        """
        if not transcript or len(transcript.strip()) == 0:
            return {"error": "Empty transcript"}
        
        # Clean and tokenize
        tokens = self._tokenize(transcript)
        sentences = self._split_sentences(transcript)
        
        analysis = {
            'length': len(transcript),
            'token_count': len(tokens),
            'sentence_count': len(sentences),
            'first_person_ratio': self._calculate_first_person_ratio(tokens),
            'negative_word_ratio': self._calculate_negative_word_ratio(tokens),
            'sentiment_score': self._calculate_sentiment(tokens),
            'semantic_diversity': self._calculate_semantic_diversity(tokens),
            'speech_rate_features': self._analyze_speech_patterns(transcript),
            'pronoun_analysis': self._analyze_pronouns(tokens),
            'emotion_indicators': self._detect_emotion_language(tokens),
            'depression_risk_features': self._extract_depression_features(tokens, sentences)
        }
        
        return analysis
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        # Convert to lowercase and split
        text = text.lower()
        # Remove punctuation but keep contractions
        words = re.findall(r"\b[\w']+\b", text)
        return words
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_first_person_ratio(self, tokens: List[str]) -> float:
        """Calculate proportion of first-person pronouns (I, me, my, we, us)"""
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        count = sum(1 for token in tokens if token in first_person)
        
        if len(tokens) == 0:
            return 0.0
        
        return count / len(tokens)
    
    def _calculate_negative_word_ratio(self, tokens: List[str]) -> float:
        """Calculate proportion of negative sentiment words"""
        negative_words = self.depression_keywords['negative']
        count = sum(1 for token in tokens if token in negative_words)
        
        if len(tokens) == 0:
            return 0.0
        
        return count / len(tokens)
    
    def _calculate_sentiment(self, tokens: List[str]) -> float:
        """
        Calculate overall sentiment score (-1 to 1)
        -1 = very negative, 0 = neutral, 1 = very positive
        """
        positive_words = [
            'good', 'great', 'happy', 'love', 'beautiful', 'wonderful',
            'excellent', 'amazing', 'fantastic', 'best', 'perfect'
        ]
        
        negative_words = self.depression_keywords['negative']
        
        pos_count = sum(1 for token in tokens if token in positive_words)
        neg_count = sum(1 for token in tokens if token in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _calculate_semantic_diversity(self, tokens: List[str]) -> float:
        """
        Calculate type-token ratio (diversity of vocabulary)
        Higher = more diverse, Lower = repetitive (depression-related)
        """
        if len(tokens) == 0:
            return 0.0
        
        unique_tokens = len(set(tokens))
        return unique_tokens / len(tokens)
    
    def _analyze_speech_patterns(self, text: str) -> Dict:
        """Analyze speech rate and patterns"""
        return {
            'avg_word_length': self._calculate_avg_word_length(text),
            'has_long_pauses': self._detect_hesitation(text),
            'repetition_ratio': self._detect_repetition(text)
        }
    
    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length"""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        return sum(len(t) for t in tokens) / len(tokens)
    
    def _detect_hesitation(self, text: str) -> bool:
        """Detect hesitation patterns (um, uh, pauses)"""
        hesitation_patterns = [r'\bum\b', r'\buh\b', r'\bahhh\b', r'\.\.\.', r'\.\.\s']
        return any(re.search(pattern, text.lower()) for pattern in hesitation_patterns)
    
    def _detect_repetition(self, text: str) -> float:
        """Detect word repetition (sign of rumination)"""
        tokens = self._tokenize(text)
        if len(tokens) < 10:
            return 0.0
        
        # Count repeated words
        from collections import Counter
        word_counts = Counter(tokens)
        repeated = sum(1 for count in word_counts.values() if count > 3)
        
        return repeated / len(word_counts) if word_counts else 0.0
    
    def _analyze_pronouns(self, tokens: List[str]) -> Dict:
        """Analyze pronoun usage patterns"""
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        second_person = ['you', 'your', 'yours']
        third_person = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'their', 'theirs']
        
        first_count = sum(1 for t in tokens if t in first_person)
        second_count = sum(1 for t in tokens if t in second_person)
        third_count = sum(1 for t in tokens if t in third_person)
        
        total_pronouns = first_count + second_count + third_count
        
        return {
            'first_person_count': first_count,
            'second_person_count': second_count,
            'third_person_count': third_count,
            'total_pronouns': total_pronouns
        }
    
    def _detect_emotion_language(self, tokens: List[str]) -> Dict:
        """Detect emotionally charged language"""
        emotion_words = {
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'sorrowful'],
            'anxiety': ['anxiety', 'anxious', 'worried', 'nervous', 'scared', 'afraid'],
            'anger': ['angry', 'furious', 'rage', 'mad', 'irritated'],
            'hopelessness': ['hopeless', 'helpless', 'worthless', 'useless', 'pointless']
        }
        
        results = {}
        for emotion, words in emotion_words.items():
            count = sum(1 for token in tokens if token in words)
            results[emotion] = count
        
        return results
    
    def _extract_depression_features(self, tokens: List[str], sentences: List[str]) -> Dict:
        """Extract depression-specific linguistic features"""
        
        # Cognitive distortions
        absolute_words = ['always', 'never', 'nobody', 'nothing', 'everything', 'worst']
        absolute_count = sum(1 for t in tokens if t in absolute_words)
        
        # Rumination (many subordinate clauses)
        rumination_words = ['because', 'when', 'if', 'while', 'before', 'after']
        rumination_count = sum(1 for t in tokens if t in rumination_words)
        
        # Existential references
        existential_words = ['life', 'death', 'meaning', 'purpose', 'exist', 'live']
        existential_count = sum(1 for t in tokens if t in existential_words)
        
        return {
            'absolute_thinking_score': absolute_count / max(len(tokens), 1),
            'rumination_score': rumination_count / max(len(tokens), 1),
            'existential_concerns_score': existential_count / max(len(tokens), 1)
        }
