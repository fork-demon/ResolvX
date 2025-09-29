"""
Triage agent for intelligent incident routing and prioritization.

This agent analyzes incoming incidents, alerts, and requests to determine
severity levels and route tasks to appropriate teams or agents.
"""

from agents.triage.agent import TriageAgent

__all__ = ["TriageAgent"]