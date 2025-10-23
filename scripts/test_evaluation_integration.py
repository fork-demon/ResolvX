#!/usr/bin/env python3
"""
Test script for evaluation and guardrails integration.

This script demonstrates how the evaluation system (guardrails, hallucination detection,
and quality assessment) is integrated into the agent workflow.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import load_config
from core.evaluation import Guardrails, HallucinationChecker, QualityAssessor
from core.observability import get_logger


async def test_guardrails():
    """Test guardrails functionality."""
    print("🔒 Testing Guardrails...")
    
    # Initialize guardrails with configuration
    guardrails = Guardrails({
        "enable_pii_detection": True,
        "enable_safety_checks": True,
        "enable_policy_enforcement": True,
        "pii_patterns": ["email", "phone", "ssn"],
        "safety_patterns": ["malicious", "harmful"]
    })
    
    # Test cases
    test_cases = [
        {
            "content": '{"severity": "critical", "category": "infrastructure", "tools_to_use": ["splunk_search"]}',
            "description": "Clean JSON response"
        },
        {
            "content": '{"severity": "critical", "user_email": "john.doe@company.com", "tools_to_use": ["splunk_search"]}',
            "description": "Response with PII (email)"
        },
        {
            "content": '{"severity": "critical", "malicious_code": "rm -rf /", "tools_to_use": ["splunk_search"]}',
            "description": "Response with safety violation"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test_case['description']}")
        
        result = await guardrails.check_content(
            content=test_case["content"],
            content_type="json",
            context={"test_case": i}
        )
        
        print(f"    Passed: {result.passed}")
        print(f"    Severity: {result.severity}")
        print(f"    Message: {result.message}")
        if result.violations:
            print(f"    Violations: {result.violations}")
        if result.suggestions:
            print(f"    Suggestions: {result.suggestions}")


async def test_hallucination_checker():
    """Test hallucination detection."""
    print("\n🧠 Testing Hallucination Checker...")
    
    # Initialize hallucination checker
    hallucination_checker = HallucinationChecker({
        "enable_fact_checking": True,
        "enable_consistency_checks": True,
        "knowledge_base_threshold": 0.8
    })
    
    # Test cases
    test_cases = [
        {
            "response": '{"severity": "critical", "category": "infrastructure", "tools_to_use": ["splunk_search"]}',
            "context": {"ticket": {"subject": "Server down", "description": "Production server is not responding"}},
            "description": "Realistic response"
        },
        {
            "response": '{"severity": "critical", "category": "infrastructure", "tools_to_use": ["splunk_search"], "solution": "Restart the quantum computer"}',
            "context": {"ticket": {"subject": "Server down", "description": "Production server is not responding"}},
            "description": "Response with unrealistic solution"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test_case['description']}")
        
        result = await hallucination_checker.check_response(
            response=test_case["response"],
            context=test_case["context"],
            sources=None  # No sources for this test
        )
        
        print(f"    Has Hallucination: {result.has_hallucination}")
        print(f"    Confidence: {result.confidence}")
        if result.hallucination_types:
            print(f"    Types: {result.hallucination_types}")
        if result.suggestions:
            print(f"    Suggestions: {result.suggestions}")


async def test_quality_assessor():
    """Test quality assessment."""
    print("\n📊 Testing Quality Assessor...")
    
    # Initialize quality assessor
    quality_assessor = QualityAssessor({
        "enable_quality_assessment": True,
        "expected_quality": "high",
        "performance_tracking": True
    })
    
    # Test cases
    test_cases = [
        {
            "response": '{"severity": "critical", "category": "infrastructure", "tools_to_use": ["splunk_search"], "reasoning": "Server is down, need to check logs"}',
            "context": {"ticket": {"subject": "Server down", "description": "Production server is not responding"}},
            "description": "High quality response"
        },
        {
            "response": '{"severity": "unknown", "category": "unknown", "tools_to_use": []}',
            "context": {"ticket": {"subject": "Server down", "description": "Production server is not responding"}},
            "description": "Low quality response"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test_case['description']}")
        
        result = await quality_assessor.assess_response(
            response=test_case["response"],
            context=test_case["context"],
            expected_format="json"
        )
        
        print(f"    Overall Score: {result.overall_score}")
        print(f"    Dimension Scores: {result.dimension_scores}")
        if result.strengths:
            print(f"    Strengths: {result.strengths}")
        if result.weaknesses:
            print(f"    Weaknesses: {result.weaknesses}")
        if result.suggestions:
            print(f"    Suggestions: {result.suggestions}")


async def test_integrated_evaluation():
    """Test integrated evaluation workflow."""
    print("\n🔄 Testing Integrated Evaluation...")
    
    # Load configuration
    config = load_config()
    
    # Initialize all evaluation components
    triage_config = getattr(config.agents, "triage", {})
    guardrails = Guardrails(getattr(triage_config, "guardrails", {}) if hasattr(triage_config, "guardrails") else {})
    hallucination_checker = HallucinationChecker(getattr(triage_config, "hallucination", {}) if hasattr(triage_config, "hallucination") else {})
    quality_assessor = QualityAssessor(getattr(triage_config, "quality", {}) if hasattr(triage_config, "quality") else {})
    
    # Simulate a ticket analysis scenario
    ticket = {
        "id": "TEST-001",
        "subject": "Production server down",
        "description": "The main production server is not responding to requests. Users are reporting 500 errors.",
        "source": "monitoring"
    }
    
    # Simulate LLM analysis response
    llm_response = '''{
        "severity": "critical",
        "category": "infrastructure", 
        "tools_to_use": ["splunk_search", "newrelic_metrics"],
        "reasoning": "Server is down, need to check logs and metrics to diagnose the issue"
    }'''
    
    print(f"  Ticket: {ticket['subject']}")
    print(f"  LLM Response: {llm_response[:100]}...")
    
    # Run all evaluations
    print("\n  Running evaluations...")
    
    # Guardrails check
    guardrail_result = await guardrails.check_content(
        content=llm_response,
        content_type="json",
        context={"ticket": ticket}
    )
    print(f"    Guardrails: {'✅ PASS' if guardrail_result.passed else '❌ FAIL'} ({guardrail_result.severity})")
    
    # Hallucination check
    hallucination_result = await hallucination_checker.check_response(
        response=llm_response,
        context={"ticket": ticket},
        sources=None
    )
    print(f"    Hallucination: {'✅ PASS' if not hallucination_result.has_hallucination else '❌ FAIL'}")
    
    # Quality assessment
    quality_result = await quality_assessor.assess_response(
        response=llm_response,
        context={"ticket": ticket, "tools_recommended": ["splunk_search", "newrelic_metrics"]},
        expected_format="json"
    )
    print(f"    Quality: {quality_result.overall_score}/10")
    
    # Summary
    print(f"\n  📊 Evaluation Summary:")
    print(f"    Guardrails: {'✅' if guardrail_result.passed else '❌'} {guardrail_result.message}")
    print(f"    Hallucination: {'✅' if not hallucination_result.has_hallucination else '❌'} (confidence: {hallucination_result.confidence})")
    print(f"    Quality: {quality_result.overall_score}/10")
    
    # Overall assessment
    overall_passed = guardrail_result.passed and not hallucination_result.has_hallucination and quality_result.overall_score >= 0.7
    print(f"\n  🎯 Overall Assessment: {'✅ APPROVED' if overall_passed else '❌ NEEDS REVIEW'}")


async def main():
    """Run all evaluation tests."""
    print("🧪 EVALUATION INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test individual components
        await test_guardrails()
        await test_hallucination_checker()
        await test_quality_assessor()
        
        # Test integrated workflow
        await test_integrated_evaluation()
        
        print("\n" + "=" * 50)
        print("✅ All evaluation tests completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Run the full workflow test to see evaluation in action")
        print("   2. Check LangFuse for evaluation metrics and traces")
        print("   3. Configure evaluation settings in config/agent.yaml")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
