"""
Memory Agent for historical ticket lookup and conditional insertion.

Behavior:
- Search memory for historical tickets similar to the current ticket.
- If related tickets are found, forward them to the Supervisor (or return payload).
- If none are found, insert the current ticket into memory for future reference.
- Maintain a simple daily cache for search results that resets at midnight.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.memory.base import BaseMemory
from core.observability import get_logger, get_tracer, get_metrics_client


class MemoryAgentState(AgentState):
    """State for the Memory Agent."""

    ticket: Dict[str, Any] = Field(default_factory=dict, description="Current ticket payload")
    team: str = Field(default="operations", description="Team for namespacing")
    related_tickets: List[Dict[str, Any]] = Field(default_factory=list, description="Found related tickets")
    result: Dict[str, Any] = Field(default_factory=dict, description="Final result")


class MemoryAgent(BaseAgent):
    """
    Memory Agent that performs historical lookup and conditional insertion.
    """

    def __init__(self, config: Dict[str, Any], memory: Optional[BaseMemory] = None, **kwargs: Any):
        super().__init__("memory", config, **kwargs)
        self.logger = get_logger("agent.memory")
        self.tracer = get_tracer("agent.memory")
        self.metrics = get_metrics_client()
        self.memory = memory

        # Config
        self.namespace_prefix: str = config.get("namespace_prefix", "tickets")
        self.search_limit: int = config.get("search_limit", 10)
        self.search_threshold: float = config.get("search_threshold", 0.7)
        self.duplicate_threshold: float = config.get("duplicate_threshold", 0.85)
        self.forward_mode: str = config.get("forward_mode", "return")  # or "invoke"
        self.embedding_model_name: Optional[str] = config.get("embedding_model")

        # Daily cache
        self._cache_date: Optional[datetime.date] = None
        self._search_cache: Dict[str, List[Dict[str, Any]]] = {}

        # Lazy embedder
        self._embedder = None

    def build_graph(self) -> StateGraph:
        graph = StateGraph(MemoryAgentState)
        graph.add_node("search_related", self._search_related)
        graph.add_node("forward_or_store", self._forward_or_store)
        graph.add_edge("search_related", "forward_or_store")
        graph.set_entry_point("search_related")
        return graph.compile()

    async def process(self, state: MemoryAgentState) -> Dict[str, Any]:
        with self.tracer.start_as_current_span("memory_agent_process") as span:
            span.set_attribute("ticket_id", state.input_data.get("ticket", {}).get("id", "unknown"))
            result = await self.graph.ainvoke(state.dict())
            span.set_attribute("related_count", len(result.get("related_tickets", [])))
            return result

    def _reset_cache_if_needed(self):
        today = datetime.date.today()
        if self._cache_date != today:
            self._search_cache.clear()
            self._cache_date = today
            self.logger.info("MemoryAgent daily cache reset")

    def _cache_key_for_ticket(self, ticket: Dict[str, Any]) -> str:
        subject = (ticket.get("subject") or "").strip().lower()
        description = (ticket.get("description") or "").strip().lower()
        team = (ticket.get("team") or "").strip().lower()
        return f"{team}|{subject}|{description}"

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        if not self.embedding_model_name:
            return None
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            vec = self._embedder.encode([text], normalize_embeddings=True)
            return vec[0].tolist()
        except Exception as e:
            self.logger.warning(f"Embedding unavailable; falling back to text similarity: {e}")
            return None

    async def _search_related(self, state: MemoryAgentState) -> Dict[str, Any]:
        try:
            self._reset_cache_if_needed()
            ticket = state.input_data.get("ticket", {})
            team = state.input_data.get("team") or ticket.get("team") or "operations"
            namespace = f"{self.namespace_prefix}:{team}"

            # Cache check
            key = self._cache_key_for_ticket(ticket)
            if key in self._search_cache:
                state.related_tickets = self._search_cache[key]
                return state.dict()

            # Build query text
            subject = ticket.get("subject", "")
            description = ticket.get("description", "")
            tags = ticket.get("tags", [])
            query_text = f"{subject}\n{description}\n{' '.join(tags)}".strip()

            # Optional embedding
            embedding = await self._get_embedding(query_text)

            # Perform search
            related: List[Dict[str, Any]] = []
            if self.memory:
                results = await self.memory.search(
                    query=query_text,
                    namespace=namespace,
                    limit=self.search_limit,
                    threshold=self.search_threshold,
                    query_embedding=embedding,
                    metadata_filter={"type": "ticket"},
                )
                for r in results:
                    entry = r.entry
                    related.append({
                        "id": entry.id,
                        "score": r.score,
                        "content": entry.content,
                        "metadata": entry.metadata,
                        "created_at": getattr(entry, "created_at", None),
                        "updated_at": getattr(entry, "updated_at", None),
                    })

            state.related_tickets = related
            self._search_cache[key] = related
            return state.dict()

        except Exception as e:
            self.logger.error(f"Related search failed: {e}")
            state.related_tickets = []
            return state.dict()

    async def _forward_or_store(self, state: MemoryAgentState) -> Dict[str, Any]:
        try:
            ticket = state.input_data.get("ticket", {})
            team = state.input_data.get("team") or ticket.get("team") or "operations"
            namespace = f"{self.namespace_prefix}:{team}"

            # If related tickets exist, evaluate duplicate logic against top match
            if state.related_tickets:
                top = state.related_tickets[0]
                is_duplicate = top.get("score", 0.0) >= self.duplicate_threshold
                action = "forward_to_supervisor"
                decision = {}

                if is_duplicate:
                    prev_meta = (top.get("metadata") or {})
                    prev_status = (prev_meta.get("status") or "").lower()
                    prev_resolution = prev_meta.get("resolution")

                    if prev_status in ("resolved", "solved", "closed") and prev_resolution:
                        action = "duplicate_resolved_return_resolution"
                        decision = {"resolution": prev_resolution, "duplicate_of": top.get("id")}
                    elif prev_status in ("in_progress", "open", "pending"):
                        action = "duplicate_in_progress_merge"
                        decision = {"merge_into": top.get("id")}
                    elif prev_status in ("closed",) and not prev_resolution:
                        action = "duplicate_closed_unresolved_escalate"
                        decision = {"escalate": True, "duplicate_of": top.get("id")}

                payload = {
                    "success": True,
                    "action": action,
                    "team": team,
                    "ticket": ticket,
                    "related_tickets": state.related_tickets,
                    "decision": decision,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                state.result = payload
                return payload

            # No related: store current ticket
            if self.memory and (ticket.get("subject") or ticket.get("description")):
                content = f"{ticket.get('subject','').strip()}\n\n{ticket.get('description','').strip()}"
                metadata = {
                    "type": "ticket",
                    "ticket_id": ticket.get("id"),
                    "status": ticket.get("status"),
                    "priority": ticket.get("priority"),
                    "tags": ticket.get("tags", []),
                    "requester_id": ticket.get("requester_id"),
                    "assignee_id": ticket.get("assignee_id"),
                    "interactions": ticket.get("interactions", []),
                    "resolution": ticket.get("resolution"),
                }
                embedding = await self._get_embedding(content)
                await self.memory.store(
                    content=content,
                    embedding=embedding,
                    metadata=metadata,
                    namespace=namespace,
                    category="ticket",
                    source="zendesk",
                    source_id=str(ticket.get("id")),
                    importance_score=0.6,
                )

            payload = {
                "success": True,
                "action": "stored_current_ticket",
                "team": team,
                "ticket": ticket,
                "related_tickets": [],
                "timestamp": datetime.datetime.now().isoformat(),
            }
            state.result = payload
            return payload

        except Exception as e:
            self.logger.error(f"Forward-or-store failed: {e}")
            return {"success": False, "error": str(e)}


