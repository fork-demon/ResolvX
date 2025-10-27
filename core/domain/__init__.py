"""
Domain Knowledge Module

Provides structured access to domain-specific knowledge including:
- Entity definitions (GTIN, TPNB, LocationCluster)
- Tool mappings and requirements
- Incident type classifications
- Escalation procedures
"""

from core.domain.knowledge_loader import (
    DomainKnowledge,
    KnowledgeLoader,
    load_domain_knowledge,
)

__all__ = [
    "DomainKnowledge",
    "KnowledgeLoader",
    "load_domain_knowledge",
]

