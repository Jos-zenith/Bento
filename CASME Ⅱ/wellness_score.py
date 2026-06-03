# ============================================================================
# Wellness Engagement Scoring - Panjaayathu Integration
# ============================================================================

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MicroExpressionResult:
    """Result from micro-expression classification"""
    emotion_class: str
    confidence: float
    emotion_probabilities: Dict[str, float]
    affective_score: float


@dataclass
class SentimentResult:
    """Result from sentiment analysis"""
    text: str
    sentiment_score: float
    confidence: float


@dataclass
class WellnessScore:
    """Composite wellness engagement score"""
    affective_score: float
    sentiment_score: float
    cultural_alignment_score: float
    overall_wellness_score: float
    breakdown: Dict[str, float]


class WellnessScoreCalculator:
    """
    Calculates Wellness Engagement Score: W(u) = alpha*A(u) + beta*S(u) + gamma*C(u)
    
    Where:
    - A(u): Affective/Micro-expression score from CASME II model
    - S(u): Sentiment score from LLM/VLM text analysis
    - C(u): Cultural alignment/context score
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        emotion_valence: Optional[Dict[str, float]] = None,
        dharma_keywords: Optional[List[str]] = None,
        seva_keywords: Optional[List[str]] = None,
    ):
        """
        Args:
            alpha: Weight for affective score
            beta: Weight for sentiment score
            gamma: Weight for cultural alignment score
            emotion_valence: Mapping of emotion to valence [-1, 1]
            dharma_keywords: Keywords indicating Dharma alignment
            seva_keywords: Keywords indicating Seva (service) alignment
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Emotion valence mapping (default)
        self.emotion_valence = emotion_valence or {
            'Happiness': 1.0,
            'Surprise': 0.0,
            'Disgust': -0.8,
            'Repression': -0.5,
            'Others': 0.0
        }
        
        # Cultural alignment keywords
        self.dharma_keywords = dharma_keywords or [
            'purpose', 'duty', 'principle', 'virtue', 'integrity',
            'righteousness', 'dharma', 'ethics', 'truth', 'law'
        ]
        
        self.seva_keywords = seva_keywords or [
            'serve', 'help', 'support', 'care', 'compassion',
            'seva', 'community', 'together', 'collective', 'volunteer'
        ]
    
    def compute_affective_score(
        self,
        micro_expr_result: MicroExpressionResult
    ) -> float:
        """
        Compute affective score from micro-expression classification.
        
        Uses emotion valence mapping:
        - Positive emotions: +1.0
        - Neutral emotions: 0.0
        - Negative emotions: -0.5 to -0.8
        
        Weighted by confidence for reliability.
        
        Args:
            micro_expr_result: Result from CASME II model
            
        Returns:
            affective_score: Score in range [-1, 1]
        """
        emotion = micro_expr_result.emotion_class
        confidence = micro_expr_result.confidence
        
        # Get valence for detected emotion
        valence = self.emotion_valence.get(emotion, 0.0)
        
        # Weight by confidence
        affective_score = valence * confidence
        
        return np.clip(affective_score, -1.0, 1.0)
    
    def compute_sentiment_score(self, sentiment_result: SentimentResult) -> float:
        """
        Normalize sentiment score to [-1, 1] range.
        
        Args:
            sentiment_result: Result from sentiment analysis
            
        Returns:
            sentiment_score: Score in range [-1, 1]
        """
        # Typically already in [-1, 1] range
        return np.clip(sentiment_result.sentiment_score, -1.0, 1.0)
    
    def compute_cultural_alignment(
        self,
        text: str,
        dharma_weight: float = 0.6,
        seva_weight: float = 0.4,
    ) -> float:
        """
        Compute cultural alignment score based on Dharma and Seva principles.
        
        Dharma: Adherence to principles, duty, righteousness
        Seva: Community service, compassion, collective benefit
        
        Args:
            text: Text to analyze (e.g., user statement, coaching feedback)
            dharma_weight: Weight for Dharma alignment [0, 1]
            seva_weight: Weight for Seva alignment [0, 1]
            
        Returns:
            cultural_alignment_score: Score in range [0, 1]
        """
        text_lower = text.lower()
        
        # Count keyword matches
        dharma_matches = sum(1 for kw in self.dharma_keywords if kw in text_lower)
        seva_matches = sum(1 for kw in self.seva_keywords if kw in text_lower)
        
        # Normalize by keyword counts
        total_dharma = len(self.dharma_keywords)
        total_seva = len(self.seva_keywords)
        
        dharma_score = min(dharma_matches / max(total_dharma, 1), 1.0)
        seva_score = min(seva_matches / max(total_seva, 1), 1.0)
        
        # Weighted combination
        alignment_score = (dharma_weight * dharma_score + seva_weight * seva_score)
        
        return np.clip(alignment_score, 0.0, 1.0)
    
    def calculate_wellness_score(
        self,
        micro_expr_result: MicroExpressionResult,
        sentiment_result: Optional[SentimentResult] = None,
        cultural_text: Optional[str] = None,
    ) -> WellnessScore:
        """
        Calculate overall wellness engagement score.
        
        W(u) = alpha*A(u) + beta*S(u) + gamma*C(u)
        
        Args:
            micro_expr_result: Micro-expression classification result
            sentiment_result: (Optional) Sentiment analysis result
            cultural_text: (Optional) Text for cultural alignment analysis
            
        Returns:
            WellnessScore object with breakdown
        """
        # Affective score (always computed)
        affective_score = self.compute_affective_score(micro_expr_result)
        
        # Sentiment score
        if sentiment_result is not None:
            sentiment_score = self.compute_sentiment_score(sentiment_result)
        else:
            # Fallback to affective score if sentiment not provided
            sentiment_score = affective_score
        
        # Cultural alignment score
        if cultural_text is not None:
            cultural_alignment_score = self.compute_cultural_alignment(cultural_text)
        else:
            # Default neutral if not provided
            cultural_alignment_score = 0.5
        
        # Composite wellness score (normalized to [0, 1])
        # Convert [-1, 1] scores to [0, 1]
        affective_normalized = (affective_score + 1.0) / 2.0
        sentiment_normalized = (sentiment_score + 1.0) / 2.0
        
        overall_wellness_score = (
            self.alpha * affective_normalized +
            self.beta * sentiment_normalized +
            self.gamma * cultural_alignment_score
        )
        
        return WellnessScore(
            affective_score=affective_score,
            sentiment_score=sentiment_score,
            cultural_alignment_score=cultural_alignment_score,
            overall_wellness_score=overall_wellness_score,
            breakdown={
                'affective_component': self.alpha * affective_normalized,
                'sentiment_component': self.beta * sentiment_normalized,
                'cultural_component': self.gamma * cultural_alignment_score,
            }
        )
    
    def interpret_wellness_score(self, score: float) -> str:
        """
        Provide interpretation of wellness score.
        
        Args:
            score: Wellness score in [0, 1]
            
        Returns:
            Interpretation string
        """
        if score >= 0.8:
            return "Excellent wellness engagement - Strong positive emotions and alignment"
        elif score >= 0.6:
            return "Good wellness engagement - Generally positive outlook"
        elif score >= 0.4:
            return "Moderate wellness engagement - Mixed emotions present"
        elif score >= 0.2:
            return "Low wellness engagement - Concerns detected, coaching recommended"
        else:
            return "Critical wellness concern - Immediate intervention may be needed"
    
    def generate_coaching_insights(self, micro_expr_result: MicroExpressionResult) -> Dict[str, str]:
        """
        Generate coaching insights based on detected micro-expressions.
        
        Args:
            micro_expr_result: Micro-expression classification result
            
        Returns:
            Dictionary with coaching recommendations
        """
        emotion = micro_expr_result.emotion_class
        
        coaching_insights = {
            'Happiness': {
                'insight': 'Positive emotion detected',
                'recommendation': 'Encourage continuation of current behavior or thought pattern',
                'coaching_script': 'I notice genuine positive feelings. What specifically is contributing to this happiness?'
            },
            'Surprise': {
                'insight': 'Unexpected reaction or new information processing',
                'recommendation': 'Explore the cause of surprise, check for unmet expectations',
                'coaching_script': 'Something seems unexpected. Help me understand what surprised you just now?'
            },
            'Disgust': {
                'insight': 'Negative judgment or rejection response',
                'recommendation': 'Explore source of aversion, validate feelings, reframe if appropriate',
                'coaching_script': 'I sense some resistance or rejection. What specifically is creating that feeling?'
            },
            'Repression': {
                'insight': 'Suppressed or hidden emotions - potential incongruence',
                'recommendation': 'Carefully explore gap between stated and felt emotions, build psychological safety',
                'coaching_script': "Your words and micro-expressions might not align. What's really going on beneath the surface?"
            },
            'Others': {
                'insight': 'Complex or ambiguous emotional state',
                'recommendation': 'Use open-ended questions to clarify emotional landscape',
                'coaching_script': "Your expression is complex. Help me understand the full picture of how you're feeling."
            }
        }
        
        return coaching_insights.get(emotion, {})


# ============================================================================
# Batch Wellness Score Calculation
# ============================================================================

def batch_wellness_scores(
    micro_expr_results: List[MicroExpressionResult],
    sentiment_results: Optional[List[SentimentResult]] = None,
    cultural_texts: Optional[List[str]] = None,
    config: Optional[Dict] = None,
) -> List[WellnessScore]:
    """
    Calculate wellness scores for a batch of results.
    
    Args:
        micro_expr_results: List of micro-expression results
        sentiment_results: (Optional) List of sentiment results
        cultural_texts: (Optional) List of cultural context texts
        config: (Optional) Configuration dict with alpha, beta, gamma
        
    Returns:
        List of WellnessScore objects
    """
    if config is None:
        config = {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
    
    calculator = WellnessScoreCalculator(
        alpha=config.get('alpha', 0.5),
        beta=config.get('beta', 0.3),
        gamma=config.get('gamma', 0.2),
    )
    
    wellness_scores = []
    for i, micro_expr_result in enumerate(micro_expr_results):
        sentiment_result = sentiment_results[i] if sentiment_results else None
        cultural_text = cultural_texts[i] if cultural_texts else None
        
        score = calculator.calculate_wellness_score(
            micro_expr_result,
            sentiment_result,
            cultural_text,
        )
        wellness_scores.append(score)
    
    return wellness_scores


if __name__ == "__main__":
    # Example usage
    from config import Config
    
    calculator = WellnessScoreCalculator(
        alpha=Config.ALPHA,
        beta=Config.BETA,
        gamma=Config.GAMMA,
        emotion_valence=Config.EMOTION_VALENCE,
    )
    
    # Sample micro-expression result
    micro_expr_result = MicroExpressionResult(
        emotion_class='Happiness',
        confidence=0.92,
        emotion_probabilities={
            'Happiness': 0.92,
            'Surprise': 0.05,
            'Disgust': 0.01,
            'Repression': 0.01,
            'Others': 0.01
        },
        affective_score=0.92
    )
    
    # Sample sentiment result
    sentiment_result = SentimentResult(
        text="I'm feeling great about the progress we've made together!",
        sentiment_score=0.85,
        confidence=0.89
    )
    
    # Sample cultural text
    cultural_text = "This aligns with our duty to serve others and live with integrity in our community."
    
    # Calculate wellness score
    wellness_score = calculator.calculate_wellness_score(
        micro_expr_result,
        sentiment_result,
        cultural_text
    )
    
    print(f"Wellness Score: {wellness_score.overall_wellness_score:.3f}")
    print(f"Interpretation: {calculator.interpret_wellness_score(wellness_score.overall_wellness_score)}")
    print(f"\nBreakdown: {wellness_score.breakdown}")
    
    # Get coaching insights
    insights = calculator.generate_coaching_insights(micro_expr_result)
    print(f"\nCoaching Insight: {insights.get('insight')}")
    print(f"Coaching Script: {insights.get('coaching_script')}")
