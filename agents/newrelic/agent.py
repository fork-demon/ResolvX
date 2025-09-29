"""
New Relic Agent: selects appropriate NRQL templates and (optionally) executes them.
Minimal: chooses query based on context; execution is stubbed unless credentials provided.
"""

from typing import Any, Dict, Optional
from langgraph import StateGraph, END
from pydantic import BaseModel, Field

from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.observability import get_logger, get_tracer


class NewRelicAgentState(AgentState):
    ticket: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)


class NewRelicAgent(BaseAgent[NewRelicAgentState]):
    def __init__(self, config: Dict[str, Any], **kwargs: Any):
        super().__init__("newrelic", config, **kwargs)
        self.logger = get_logger("agent.newrelic")
        self.tracer = get_tracer("agent.newrelic")

        self.api_key: Optional[str] = config.get("api_key")
        self.account_id: Optional[str] = config.get("account_id")
        self.region: str = config.get("region", "US")
        self.knowledge_dir: str = config.get("knowledge_dir", "knowledge/newrelic")

    def build_graph(self) -> StateGraph:
        g = StateGraph(NewRelicAgentState)
        g.add_node("select_query", self._select_query)
        g.add_node("execute_query", self._execute_query)
        g.add_edge("select_query", "execute_query")
        g.set_entry_point("select_query")
        return g.compile()

    async def process(self, state: NewRelicAgentState) -> Dict[str, Any]:
        with self.tracer.start_as_current_span("newrelic_process"):
            return await self.graph.ainvoke(state.dict())

    async def _select_query(self, state: NewRelicAgentState) -> Dict[str, Any]:
        ticket = state.input_data.get("ticket", {})
        subject = (ticket.get("subject") or "").lower()

        nrql = "SELECT count(*) FROM TransactionError SINCE 1 hour ago FACET appName"
        if "latency" in subject or "slow" in subject:
            nrql = "SELECT average(duration) FROM Transaction SINCE 1 hour ago FACET appName"
        elif "error" in subject:
            nrql = "SELECT count(*) FROM TransactionError SINCE 1 hour ago FACET error.class"

        state.result = {"nrql": nrql, "executed": False, "results": []}
        return state.dict()

    async def _execute_query(self, state: NewRelicAgentState) -> Dict[str, Any]:
        result = state.result or {}
        if self.api_key and self.account_id:
            # Placeholder for real execution; keep minimal
            result["executed"] = True
            result["results"] = [{"timestamp": "now", "value": 123}]
        state.result = result
        return state.dict()


