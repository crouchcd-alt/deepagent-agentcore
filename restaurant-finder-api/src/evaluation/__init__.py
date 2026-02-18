"""
AgentCore Evaluations module for the Restaurant Finder Agent.

This module provides comprehensive evaluation capabilities using AWS Bedrock AgentCore
Evaluations, including:
- Built-in evaluators (Correctness, GoalSuccessRate, ToolSelectionAccuracy, etc.)
- Custom evaluators for restaurant-specific metrics
- On-demand evaluation for development and testing
- Online evaluation for production monitoring

Usage:
    from src.evaluation import EvaluationClient, run_on_demand_evaluation

    # Run on-demand evaluation
    results = await run_on_demand_evaluation(session_id="your-session-id")

    # Setup online evaluation for production
    from src.evaluation import setup_online_evaluation
    config = await setup_online_evaluation()

    # Run comprehensive evaluation with test cases
    from src.evaluation import EvaluationRunner
    runner = EvaluationRunner(agent_id="my-agent", agent_arn="arn:...")
    results = await runner.run_full_evaluation()
"""

from src.evaluation.client import EvaluationClient
from src.evaluation.on_demand import run_on_demand_evaluation, evaluate_session
from src.evaluation.online import setup_online_evaluation, OnlineEvaluationManager
from src.evaluation.test_cases import RESTAURANT_EVAL_CASES, EvalTestCase, TestCategory
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationClient",
    "EvaluationRunner",
    "run_on_demand_evaluation",
    "evaluate_session",
    "setup_online_evaluation",
    "OnlineEvaluationManager",
    "RESTAURANT_EVAL_CASES",
    "EvalTestCase",
    "TestCategory",
]
