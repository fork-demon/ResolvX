"""
Quality assessment system for the Golden Agent Framework.

Provides comprehensive quality evaluation of agent responses including
relevance, accuracy, completeness, and clarity metrics.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from core.exceptions import QualityError
from core.observability import get_logger


class QualityDimension(str, Enum):
    """Quality assessment dimensions."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    COHERENCE = "coherence"
    USEFULNESS = "usefulness"


class QualityResult(BaseModel):
    """Result of quality assessment."""

    overall_score: float
    dimension_scores: Dict[QualityDimension, float] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class QualityAssessor:
    """
    Quality assessment system for agent responses.
    
    Evaluates responses across multiple quality dimensions
    and provides actionable feedback.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality assessor.
        
        Args:
            config: Quality assessment configuration
        """
        self.config = config or {}
        self.logger = get_logger("evaluation.quality")
        
        # Quality criteria
        self._relevance_criteria = self._load_relevance_criteria()
        self._accuracy_criteria = self._load_accuracy_criteria()
        self._completeness_criteria = self._load_completeness_criteria()
        self._clarity_criteria = self._load_clarity_criteria()
        
        # Configuration
        self.min_score_threshold = self.config.get("min_score_threshold", 0.6)
        self.enable_detailed_analysis = self.config.get("enable_detailed_analysis", True)

    async def assess_response(
        self,
        response: str,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        expected_format: Optional[str] = None
    ) -> QualityResult:
        """
        Assess quality of agent response.
        
        Args:
            response: Agent response to assess
            query: Original query (if available)
            context: Additional context for assessment
            expected_format: Expected response format
            
        Returns:
            Quality assessment result
        """
        try:
            dimension_scores = {}
            strengths = []
            weaknesses = []
            suggestions = []
            
            # Assess each quality dimension
            relevance_score = await self._assess_relevance(response, query, context)
            dimension_scores[QualityDimension.RELEVANCE] = relevance_score
            
            accuracy_score = await self._assess_accuracy(response, context)
            dimension_scores[QualityDimension.ACCURACY] = accuracy_score
            
            completeness_score = await self._assess_completeness(response, query, context)
            dimension_scores[QualityDimension.COMPLETENESS] = completeness_score
            
            clarity_score = await self._assess_clarity(response, expected_format)
            dimension_scores[QualityDimension.CLARITY] = clarity_score
            
            coherence_score = await self._assess_coherence(response)
            dimension_scores[QualityDimension.COHERENCE] = coherence_score
            
            usefulness_score = await self._assess_usefulness(response, query, context)
            dimension_scores[QualityDimension.USEFULNESS] = usefulness_score
            
            # Calculate overall score
            overall_score = sum(dimension_scores.values()) / len(dimension_scores)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(dimension_scores)
            
            # Generate suggestions
            suggestions = await self._generate_quality_suggestions(
                dimension_scores, weaknesses, response, query
            )
            
            return QualityResult(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                suggestions=suggestions,
                metadata={
                    "response_length": len(response),
                    "query_provided": query is not None,
                    "context_provided": context is not None,
                    "expected_format": expected_format,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return QualityResult(
                overall_score=0.0,
                dimension_scores={},
                weaknesses=[f"Quality assessment failed: {e}"],
            )

    async def _assess_relevance(
        self, response: str, query: Optional[str], context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess relevance of response to query."""
        if not query:
            return 0.5  # Neutral score if no query provided
        
        score = 0.0
        criteria_met = 0
        total_criteria = len(self._relevance_criteria)
        
        # Check keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        if overlap > 0:
            score += 0.3
            criteria_met += 1
        
        # Check for direct question answering
        if any(word in response.lower() for word in ["answer", "response", "solution", "explanation"]):
            score += 0.2
            criteria_met += 1
        
        # Check for topic consistency
        if self._topics_consistent(query, response):
            score += 0.3
            criteria_met += 1
        
        # Check for context awareness
        if context and self._uses_context(response, context):
            score += 0.2
            criteria_met += 1
        
        return min(score, 1.0)

    async def _assess_accuracy(self, response: str, context: Optional[Dict[str, Any]]) -> float:
        """Assess accuracy of response."""
        score = 0.0
        criteria_met = 0
        total_criteria = len(self._accuracy_criteria)
        
        # Check for factual consistency
        if self._factually_consistent(response):
            score += 0.3
            criteria_met += 1
        
        # Check for logical consistency
        if self._logically_consistent(response):
            score += 0.2
            criteria_met += 1
        
        # Check for proper citations
        if self._has_proper_citations(response):
            score += 0.2
            criteria_met += 1
        
        # Check for uncertainty markers
        if self._acknowledges_uncertainty(response):
            score += 0.1
            criteria_met += 1
        
        # Check against context if available
        if context and self._consistent_with_context(response, context):
            score += 0.2
            criteria_met += 1
        
        return min(score, 1.0)

    async def _assess_completeness(
        self, response: str, query: Optional[str], context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess completeness of response."""
        score = 0.0
        criteria_met = 0
        total_criteria = len(self._completeness_criteria)
        
        # Check response length adequacy
        if len(response) >= 50:  # Minimum meaningful response
            score += 0.2
            criteria_met += 1
        
        # Check for comprehensive coverage
        if self._covers_multiple_aspects(response):
            score += 0.3
            criteria_met += 1
        
        # Check for examples or details
        if self._includes_examples(response):
            score += 0.2
            criteria_met += 1
        
        # Check for addressing all parts of query
        if query and self._addresses_all_query_parts(response, query):
            score += 0.3
            criteria_met += 1
        
        return min(score, 1.0)

    async def _assess_clarity(self, response: str, expected_format: Optional[str]) -> float:
        """Assess clarity of response."""
        score = 0.0
        criteria_met = 0
        total_criteria = len(self._clarity_criteria)
        
        # Check for clear structure
        if self._has_clear_structure(response):
            score += 0.3
            criteria_met += 1
        
        # Check for appropriate language
        if self._uses_appropriate_language(response):
            score += 0.2
            criteria_met += 1
        
        # Check for formatting
        if self._has_good_formatting(response):
            score += 0.2
            criteria_met += 1
        
        # Check for expected format compliance
        if expected_format and self._matches_expected_format(response, expected_format):
            score += 0.3
            criteria_met += 1
        
        return min(score, 1.0)

    async def _assess_coherence(self, response: str) -> float:
        """Assess coherence of response."""
        score = 0.0
        
        # Check for logical flow
        if self._has_logical_flow(response):
            score += 0.4
        
        # Check for consistent tone
        if self._has_consistent_tone(response):
            score += 0.3
        
        # Check for proper transitions
        if self._has_good_transitions(response):
            score += 0.3
        
        return min(score, 1.0)

    async def _assess_usefulness(
        self, response: str, query: Optional[str], context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess usefulness of response."""
        score = 0.0
        
        # Check for actionable information
        if self._provides_actionable_info(response):
            score += 0.3
        
        # Check for practical value
        if self._has_practical_value(response):
            score += 0.3
        
        # Check for depth of insight
        if self._provides_insights(response):
            score += 0.2
        
        # Check for completeness of answer
        if query and self._fully_answers_query(response, query):
            score += 0.2
        
        return min(score, 1.0)

    def _topics_consistent(self, query: str, response: str) -> bool:
        """Check if response topics are consistent with query."""
        # Simple keyword-based consistency check
        query_topics = set(query.lower().split())
        response_topics = set(response.lower().split())
        overlap = len(query_topics.intersection(response_topics))
        return overlap >= len(query_topics) * 0.3

    def _uses_context(self, response: str, context: Dict[str, Any]) -> bool:
        """Check if response uses provided context."""
        context_keys = list(context.keys())
        for key in context_keys:
            if key.lower() in response.lower():
                return True
        return False

    def _factually_consistent(self, response: str) -> bool:
        """Check for factual consistency."""
        # Check for contradictory statements
        contradictions = [
            (r'\b(always|never)\b', r'\b(sometimes|occasionally)\b'),
            (r'\b(all|every)\b', r'\b(some|few)\b'),
        ]
        
        for pos_pattern, neg_pattern in contradictions:
            if re.search(pos_pattern, response, re.IGNORECASE) and \
               re.search(neg_pattern, response, re.IGNORECASE):
                return False
        
        return True

    def _logically_consistent(self, response: str) -> bool:
        """Check for logical consistency."""
        # Check for logical fallacies or inconsistencies
        logical_issues = [
            r'\b(if.*then.*but.*not)\b',
            r'\b(because.*therefore.*but)\b',
        ]
        
        for pattern in logical_issues:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        
        return True

    def _has_proper_citations(self, response: str) -> bool:
        """Check for proper citations."""
        citation_patterns = [
            r'\[.*?\]',
            r'\(.*?\d{4}.*?\)',
            r'according to',
            r'as stated in',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in citation_patterns)

    def _acknowledges_uncertainty(self, response: str) -> bool:
        """Check if response acknowledges uncertainty."""
        uncertainty_markers = [
            r'\b(may|might|could|possibly|perhaps|likely|probably)\b',
            r'\b(uncertain|unclear|unknown|debatable)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in uncertainty_markers)

    def _consistent_with_context(self, response: str, context: Dict[str, Any]) -> bool:
        """Check if response is consistent with context."""
        # This would need more sophisticated context checking
        return True

    def _covers_multiple_aspects(self, response: str) -> bool:
        """Check if response covers multiple aspects."""
        # Look for transition words that indicate multiple aspects
        transition_words = [
            r'\b(first|second|third|additionally|furthermore|moreover)\b',
            r'\b(also|besides|in addition|on the other hand)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in transition_words)

    def _includes_examples(self, response: str) -> bool:
        """Check if response includes examples."""
        example_indicators = [
            r'\b(for example|for instance|such as|like)\b',
            r'\b(example|instance|case)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in example_indicators)

    def _addresses_all_query_parts(self, response: str, query: str) -> bool:
        """Check if response addresses all parts of query."""
        # Simple check for question words
        question_words = ["what", "how", "why", "when", "where", "who"]
        query_questions = [word for word in question_words if word in query.lower()]
        
        if not query_questions:
            return True
        
        # Check if response addresses these questions
        response_lower = response.lower()
        return all(word in response_lower for word in query_questions)

    def _has_clear_structure(self, response: str) -> bool:
        """Check if response has clear structure."""
        structure_indicators = [
            r'\b(introduction|conclusion|summary)\b',
            r'\b(first|second|third|finally)\b',
            r'\b(overview|details|examples)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in structure_indicators)

    def _uses_appropriate_language(self, response: str) -> bool:
        """Check if response uses appropriate language."""
        # Check for professional tone
        inappropriate_patterns = [
            r'\b(awesome|cool|amazing|incredible)\b',
            r'\b(like|you know|um|uh)\b',
        ]
        
        return not any(re.search(pattern, response, re.IGNORECASE) for pattern in inappropriate_patterns)

    def _has_good_formatting(self, response: str) -> bool:
        """Check if response has good formatting."""
        # Check for proper capitalization
        sentences = response.split('.')
        properly_capitalized = all(s.strip()[0].isupper() for s in sentences if s.strip())
        
        # Check for reasonable paragraph breaks
        has_paragraphs = '\n\n' in response or len(response) < 200
        
        return properly_capitalized and has_paragraphs

    def _matches_expected_format(self, response: str, expected_format: str) -> bool:
        """Check if response matches expected format."""
        if expected_format == "json":
            return response.strip().startswith('{') and response.strip().endswith('}')
        elif expected_format == "list":
            return response.strip().startswith(('-', '*', '1.', '2.'))
        elif expected_format == "code":
            return '```' in response or response.strip().startswith(('def ', 'class ', 'import '))
        
        return True

    def _has_logical_flow(self, response: str) -> bool:
        """Check if response has logical flow."""
        flow_indicators = [
            r'\b(first|initially|to begin)\b',
            r'\b(then|next|subsequently)\b',
            r'\b(finally|in conclusion|to summarize)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in flow_indicators)

    def _has_consistent_tone(self, response: str) -> bool:
        """Check if response has consistent tone."""
        # This would need more sophisticated tone analysis
        return True

    def _has_good_transitions(self, response: str) -> bool:
        """Check if response has good transitions."""
        transition_words = [
            r'\b(however|therefore|moreover|furthermore|additionally)\b',
            r'\b(on the other hand|in contrast|similarly)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in transition_words)

    def _provides_actionable_info(self, response: str) -> bool:
        """Check if response provides actionable information."""
        action_indicators = [
            r'\b(you should|you can|you need to|steps|instructions)\b',
            r'\b(how to|process|procedure|method)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in action_indicators)

    def _has_practical_value(self, response: str) -> bool:
        """Check if response has practical value."""
        practical_indicators = [
            r'\b(useful|helpful|practical|applicable)\b',
            r'\b(benefit|advantage|solution|recommendation)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in practical_indicators)

    def _provides_insights(self, response: str) -> bool:
        """Check if response provides insights."""
        insight_indicators = [
            r'\b(insight|analysis|understanding|perspective)\b',
            r'\b(important|significant|key|critical)\b',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in insight_indicators)

    def _fully_answers_query(self, response: str, query: str) -> bool:
        """Check if response fully answers the query."""
        # This would need more sophisticated query-answer matching
        return len(response) > len(query) * 2

    def _identify_strengths_weaknesses(
        self, dimension_scores: Dict[QualityDimension, float]
    ) -> tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on dimension scores."""
        strengths = []
        weaknesses = []
        
        for dimension, score in dimension_scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {dimension.value}")
            elif score <= 0.4:
                weaknesses.append(f"Weak {dimension.value}")
        
        return strengths, weaknesses

    async def _generate_quality_suggestions(
        self,
        dimension_scores: Dict[QualityDimension, float],
        weaknesses: List[str],
        response: str,
        query: Optional[str]
    ) -> List[str]:
        """Generate quality improvement suggestions."""
        suggestions = []
        
        for dimension, score in dimension_scores.items():
            if score < 0.6:
                if dimension == QualityDimension.RELEVANCE:
                    suggestions.append("Ensure response directly addresses the query")
                elif dimension == QualityDimension.ACCURACY:
                    suggestions.append("Verify facts and add citations where appropriate")
                elif dimension == QualityDimension.COMPLETENESS:
                    suggestions.append("Provide more comprehensive coverage of the topic")
                elif dimension == QualityDimension.CLARITY:
                    suggestions.append("Improve structure and use clearer language")
                elif dimension == QualityDimension.COHERENCE:
                    suggestions.append("Ensure logical flow and consistent tone")
                elif dimension == QualityDimension.USEFULNESS:
                    suggestions.append("Focus on providing actionable and practical information")
        
        return suggestions

    def _load_relevance_criteria(self) -> List[str]:
        """Load relevance assessment criteria."""
        return [
            "keyword_overlap",
            "direct_question_answering",
            "topic_consistency",
            "context_awareness",
        ]

    def _load_accuracy_criteria(self) -> List[str]:
        """Load accuracy assessment criteria."""
        return [
            "factual_consistency",
            "logical_consistency",
            "proper_citations",
            "uncertainty_acknowledgment",
        ]

    def _load_completeness_criteria(self) -> List[str]:
        """Load completeness assessment criteria."""
        return [
            "response_length_adequacy",
            "comprehensive_coverage",
            "examples_and_details",
            "query_parts_addressed",
        ]

    def _load_clarity_criteria(self) -> List[str]:
        """Load clarity assessment criteria."""
        return [
            "clear_structure",
            "appropriate_language",
            "good_formatting",
            "format_compliance",
        ]
