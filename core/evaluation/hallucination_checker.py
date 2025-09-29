"""
Hallucination detection system for the Golden Agent Framework.

Provides detection and mitigation of AI hallucinations in agent responses
using various validation techniques.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from core.exceptions import HallucinationError
from core.observability import get_logger


class HallucinationType(str, Enum):
    """Types of hallucinations."""
    FACTUAL_ERROR = "factual_error"
    FABRICATED_CITATION = "fabricated_citation"
    INCONSISTENT_CLAIM = "inconsistent_claim"
    IMPOSSIBLE_CLAIM = "impossible_claim"
    OUTDATED_INFO = "outdated_info"
    SPECULATIVE_CLAIM = "speculative_claim"


class HallucinationResult(BaseModel):
    """Result of hallucination detection."""

    has_hallucination: bool
    confidence: float
    hallucination_types: List[HallucinationType] = Field(default_factory=list)
    detected_claims: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class HallucinationChecker:
    """
    Hallucination detection and mitigation system.
    
    Uses multiple techniques to detect and flag potential hallucinations
    in agent responses.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hallucination checker.
        
        Args:
            config: Hallucination checker configuration
        """
        self.config = config or {}
        self.logger = get_logger("evaluation.hallucination")
        
        # Detection patterns
        self._factual_patterns = self._load_factual_patterns()
        self._citation_patterns = self._load_citation_patterns()
        self._impossible_patterns = self._load_impossible_patterns()
        self._speculative_patterns = self._load_speculative_patterns()
        
        # Configuration
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_factual_check = self.config.get("enable_factual_check", True)
        self.enable_citation_check = self.config.get("enable_citation_check", True)
        self.enable_consistency_check = self.config.get("enable_consistency_check", True)

    async def check_response(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> HallucinationResult:
        """
        Check response for hallucinations.
        
        Args:
            response: Agent response to check
            context: Additional context for checking
            sources: Source documents used for generation
            
        Returns:
            Hallucination detection result
        """
        try:
            hallucination_types = []
            detected_claims = []
            suggestions = []
            confidence_scores = []
            
            # Check for factual errors
            if self.enable_factual_check:
                factual_result = await self._check_factual_errors(response)
                if factual_result["has_hallucination"]:
                    hallucination_types.append(HallucinationType.FACTUAL_ERROR)
                    detected_claims.extend(factual_result["claims"])
                    confidence_scores.append(factual_result["confidence"])
            
            # Check for fabricated citations
            if self.enable_citation_check:
                citation_result = await self._check_fabricated_citations(response, sources)
                if citation_result["has_hallucination"]:
                    hallucination_types.append(HallucinationType.FABRICATED_CITATION)
                    detected_claims.extend(citation_result["claims"])
                    confidence_scores.append(citation_result["confidence"])
            
            # Check for impossible claims
            impossible_result = await self._check_impossible_claims(response)
            if impossible_result["has_hallucination"]:
                hallucination_types.append(HallucinationType.IMPOSSIBLE_CLAIM)
                detected_claims.extend(impossible_result["claims"])
                confidence_scores.append(impossible_result["confidence"])
            
            # Check for speculative claims
            speculative_result = await self._check_speculative_claims(response)
            if speculative_result["has_hallucination"]:
                hallucination_types.append(HallucinationType.SPECULATIVE_CLAIM)
                detected_claims.extend(speculative_result["claims"])
                confidence_scores.append(speculative_result["confidence"])
            
            # Check for inconsistencies
            if self.enable_consistency_check:
                consistency_result = await self._check_inconsistencies(response)
                if consistency_result["has_hallucination"]:
                    hallucination_types.append(HallucinationType.INCONSISTENT_CLAIM)
                    detected_claims.extend(consistency_result["claims"])
                    confidence_scores.append(consistency_result["confidence"])
            
            # Calculate overall confidence
            overall_confidence = max(confidence_scores) if confidence_scores else 0.0
            
            # Generate suggestions
            suggestions = await self._generate_hallucination_suggestions(hallucination_types, detected_claims)
            
            # Determine if hallucination detected
            has_hallucination = len(hallucination_types) > 0 and overall_confidence >= self.confidence_threshold
            
            return HallucinationResult(
                has_hallucination=has_hallucination,
                confidence=overall_confidence,
                hallucination_types=hallucination_types,
                detected_claims=detected_claims,
                suggestions=suggestions,
                metadata={
                    "response_length": len(response),
                    "sources_count": len(sources) if sources else 0,
                    "context": context,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Hallucination check failed: {e}")
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                suggestions=[f"Hallucination check failed: {e}"],
            )

    async def _check_factual_errors(self, response: str) -> Dict[str, Any]:
        """Check for factual errors in response."""
        claims = []
        confidence = 0.0
        
        # Check for common factual error patterns
        for pattern_name, pattern in self._factual_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claim = {
                    "text": match.group(),
                    "type": "factual_error",
                    "pattern": pattern_name,
                    "position": match.start(),
                }
                claims.append(claim)
                confidence = max(confidence, 0.8)  # High confidence for pattern matches
        
        return {
            "has_hallucination": len(claims) > 0,
            "claims": claims,
            "confidence": confidence,
        }

    async def _check_fabricated_citations(self, response: str, sources: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Check for fabricated citations."""
        claims = []
        confidence = 0.0
        
        if not sources:
            return {"has_hallucination": False, "claims": [], "confidence": 0.0}
        
        # Extract citations from response
        citation_matches = re.finditer(r'\[(\d+)\]|\([^)]*\)', response)
        cited_sources = set()
        
        for match in citation_matches:
            citation_text = match.group()
            # Check if citation references actual sources
            if not self._citation_exists(citation_text, sources):
                claim = {
                    "text": citation_text,
                    "type": "fabricated_citation",
                    "position": match.start(),
                }
                claims.append(claim)
                cited_sources.add(citation_text)
                confidence = max(confidence, 0.9)  # Very high confidence for fabricated citations
        
        return {
            "has_hallucination": len(claims) > 0,
            "claims": claims,
            "confidence": confidence,
        }

    async def _check_impossible_claims(self, response: str) -> Dict[str, Any]:
        """Check for impossible or contradictory claims."""
        claims = []
        confidence = 0.0
        
        # Check for impossible patterns
        for pattern_name, pattern in self._impossible_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claim = {
                    "text": match.group(),
                    "type": "impossible_claim",
                    "pattern": pattern_name,
                    "position": match.start(),
                }
                claims.append(claim)
                confidence = max(confidence, 0.7)
        
        return {
            "has_hallucination": len(claims) > 0,
            "claims": claims,
            "confidence": confidence,
        }

    async def _check_speculative_claims(self, response: str) -> Dict[str, Any]:
        """Check for speculative or uncertain claims."""
        claims = []
        confidence = 0.0
        
        # Check for speculative language
        for pattern_name, pattern in self._speculative_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claim = {
                    "text": match.group(),
                    "type": "speculative_claim",
                    "pattern": pattern_name,
                    "position": match.start(),
                }
                claims.append(claim)
                confidence = max(confidence, 0.6)  # Lower confidence for speculative claims
        
        return {
            "has_hallucination": len(claims) > 0,
            "claims": claims,
            "confidence": confidence,
        }

    async def _check_inconsistencies(self, response: str) -> Dict[str, Any]:
        """Check for internal inconsistencies in response."""
        claims = []
        confidence = 0.0
        
        # Check for contradictory statements
        contradictions = [
            (r'\b(always|never)\b', r'\b(sometimes|occasionally)\b'),
            (r'\b(all|every)\b', r'\b(some|few)\b'),
            (r'\b(impossible|cannot)\b', r'\b(possible|can)\b'),
        ]
        
        for pos_pattern, neg_pattern in contradictions:
            pos_matches = list(re.finditer(pos_pattern, response, re.IGNORECASE))
            neg_matches = list(re.finditer(neg_pattern, response, re.IGNORECASE))
            
            if pos_matches and neg_matches:
                # Check if they're close enough to be contradictory
                for pos_match in pos_matches:
                    for neg_match in neg_matches:
                        if abs(pos_match.start() - neg_match.start()) < 200:  # Within 200 characters
                            claim = {
                                "text": f"{pos_match.group()} vs {neg_match.group()}",
                                "type": "inconsistent_claim",
                                "position": min(pos_match.start(), neg_match.start()),
                            }
                            claims.append(claim)
                            confidence = max(confidence, 0.8)
        
        return {
            "has_hallucination": len(claims) > 0,
            "claims": claims,
            "confidence": confidence,
        }

    def _citation_exists(self, citation: str, sources: List[Dict[str, Any]]) -> bool:
        """Check if citation references actual sources."""
        # Extract citation number or text
        citation_match = re.search(r'\[(\d+)\]', citation)
        if citation_match:
            citation_num = int(citation_match.group(1))
            return 1 <= citation_num <= len(sources)
        
        # Check if citation text matches source content
        citation_text = citation.strip('[]()')
        for source in sources:
            if citation_text.lower() in source.get("content", "").lower():
                return True
        
        return False

    async def _generate_hallucination_suggestions(
        self,
        hallucination_types: List[HallucinationType],
        detected_claims: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate suggestions for addressing hallucinations."""
        suggestions = []
        
        if HallucinationType.FACTUAL_ERROR in hallucination_types:
            suggestions.append("Verify factual claims against reliable sources")
        
        if HallucinationType.FABRICATED_CITATION in hallucination_types:
            suggestions.append("Ensure all citations reference actual sources")
        
        if HallucinationType.IMPOSSIBLE_CLAIM in hallucination_types:
            suggestions.append("Review claims for logical consistency and feasibility")
        
        if HallucinationType.SPECULATIVE_CLAIM in hallucination_types:
            suggestions.append("Clearly mark speculative content as such")
        
        if HallucinationType.INCONSISTENT_CLAIM in hallucination_types:
            suggestions.append("Review response for internal contradictions")
        
        if detected_claims:
            suggestions.append(f"Review {len(detected_claims)} flagged claims for accuracy")
        
        return suggestions

    def _load_factual_patterns(self) -> Dict[str, str]:
        """Load patterns for detecting factual errors."""
        return {
            "specific_dates": r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            "specific_numbers": r'\b\d+(\.\d+)?\s*(million|billion|trillion|percent|%)\b',
            "specific_names": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "definitive_claims": r'\b(always|never|all|every|none|completely|entirely)\b',
        }

    def _load_citation_patterns(self) -> Dict[str, str]:
        """Load patterns for detecting citations."""
        return {
            "numbered_citation": r'\[\d+\]',
            "parenthetical_citation": r'\([^)]*\)',
            "author_year": r'\([A-Za-z]+\s+\d{4}\)',
        }

    def _load_impossible_patterns(self) -> Dict[str, str]:
        """Load patterns for detecting impossible claims."""
        return {
            "impossible_time": r'\b(before\s+time\s+began|after\s+the\s+end\s+of\s+time)\b',
            "impossible_physics": r'\b(faster\s+than\s+light|time\s+travel)\b',
            "impossible_math": r'\b(divide\s+by\s+zero|square\s+root\s+of\s+negative)\b',
        }

    def _load_speculative_patterns(self) -> Dict[str, str]:
        """Load patterns for detecting speculative claims."""
        return {
            "uncertainty": r'\b(might|may|could|possibly|perhaps|maybe)\b',
            "speculation": r'\b(I\s+think|I\s+believe|I\s+assume|I\s+guess)\b',
            "conditional": r'\b(if|unless|provided\s+that|assuming)\b',
        }
