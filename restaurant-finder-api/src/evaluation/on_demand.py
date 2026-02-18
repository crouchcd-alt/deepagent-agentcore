"""
On-demand evaluation runner for the Restaurant Finder Agent.

Provides functionality to:
- Run evaluations on specific sessions
- Evaluate against test cases
- Generate evaluation reports
- Save results for analysis

Usage:
    # Evaluate a specific session
    python -m src.evaluation.on_demand --session-id <session_id> --agent-id <agent_id>

    # Run evaluation with custom evaluators
    python -m src.evaluation.on_demand --session-id <session_id> --agent-id <agent_id> --create-custom
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import settings
from src.evaluation.client import (
    EvaluationClient,
    EvaluationResult,
    AggregatedMetrics,
    BUILTIN_EVALUATORS,
)


# Default evaluators for restaurant finder
DEFAULT_EVALUATORS = [
    "Builtin.Correctness",
    "Builtin.GoalSuccessRate",
    "Builtin.Helpfulness",
    "Builtin.ToolSelectionAccuracy",
    "Builtin.ToolParameterAccuracy",
    "Builtin.Harmfulness",
]


async def evaluate_session(
    agent_id: str,
    session_id: str,
    evaluators: Optional[list[str]] = None,
    create_custom_evaluators: bool = False,
    output_dir: Optional[str] = None,
) -> tuple[list[EvaluationResult], AggregatedMetrics]:
    """
    Run on-demand evaluation on a specific session.

    Args:
        agent_id: The AgentCore agent ID.
        session_id: The session ID to evaluate.
        evaluators: Optional list of evaluator IDs. Defaults to DEFAULT_EVALUATORS.
        create_custom_evaluators: If True, create and include custom evaluators.
        output_dir: Optional directory to save results.

    Returns:
        Tuple of (list of EvaluationResult, AggregatedMetrics).
    """
    client = EvaluationClient()

    # Determine evaluators to use
    eval_list = evaluators or DEFAULT_EVALUATORS.copy()

    # Create custom evaluators if requested
    if create_custom_evaluators:
        logger.info("Creating custom evaluators...")
        custom_ids = client.create_all_custom_evaluators()
        eval_list.extend(custom_ids.values())
        logger.info(f"Added {len(custom_ids)} custom evaluators")

    # Setup output path
    output_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"eval_{session_id}_{timestamp}.json")

    # Run evaluation
    logger.info(f"Running evaluation on session: {session_id}")
    logger.info(f"Using {len(eval_list)} evaluators")

    results = client.run_evaluation(
        agent_id=agent_id,
        session_id=session_id,
        evaluators=eval_list,
        output_path=output_path,
    )

    # Aggregate results
    metrics = client.aggregate_results(
        results=results,
        session_id=session_id,
        agent_id=agent_id,
    )

    return results, metrics


async def run_on_demand_evaluation(
    session_id: str,
    agent_id: Optional[str] = None,
    evaluators: Optional[list[str]] = None,
    include_custom: bool = True,
    output_dir: str = "evaluation_results",
) -> dict:
    """
    Run a comprehensive on-demand evaluation.

    Args:
        session_id: The session ID to evaluate.
        agent_id: Optional agent ID. Defaults to settings.RUNTIME_ID.
        evaluators: Optional custom list of evaluators.
        include_custom: Whether to include custom evaluators.
        output_dir: Directory for saving results.

    Returns:
        dict with evaluation results and metrics.
    """
    agent_id = agent_id or settings.RUNTIME_ID

    if not agent_id:
        raise ValueError("Agent ID is required. Set RUNTIME_ID in settings or pass agent_id.")

    results, metrics = await evaluate_session(
        agent_id=agent_id,
        session_id=session_id,
        evaluators=evaluators,
        create_custom_evaluators=include_custom,
        output_dir=output_dir,
    )

    # Format results for return
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_evaluations": metrics.total_evaluations,
            "average_scores": metrics.average_scores,
            "pass_rates": metrics.pass_rates,
        },
        "results": [
            {
                "evaluator": r.evaluator_name,
                "value": r.value,
                "label": r.label,
                "explanation": r.explanation,
            }
            for r in results
        ],
    }


def print_evaluation_report(
    results: list[EvaluationResult],
    metrics: AggregatedMetrics,
) -> None:
    """Print a formatted evaluation report to console."""
    print("\n" + "=" * 70)
    print("🔬 RESTAURANT FINDER AGENT EVALUATION REPORT")
    print("=" * 70)

    if metrics.session_id:
        print(f"Session ID: {metrics.session_id}")
    if metrics.agent_id:
        print(f"Agent ID:   {metrics.agent_id}")

    print(f"\nTotal Evaluations: {metrics.total_evaluations}")
    print("-" * 70)

    # Print scores by evaluator
    print("\n📊 SCORES BY EVALUATOR:")
    print("-" * 70)
    print(f"{'Evaluator':<40} {'Avg Score':>10} {'Pass Rate':>12}")
    print("-" * 70)

    for name, avg_score in sorted(metrics.average_scores.items()):
        pass_rate = metrics.pass_rates.get(name, 0)
        status = "✅" if pass_rate >= 0.7 else "⚠️" if pass_rate >= 0.5 else "❌"
        print(f"{name:<40} {avg_score:>10.2f} {pass_rate:>10.1%} {status}")

    # Print detailed results
    print("\n" + "-" * 70)
    print("📋 DETAILED RESULTS:")
    print("-" * 70)

    for result in results:
        status = "✅" if result.value >= 0.7 else "⚠️" if result.value >= 0.5 else "❌"
        print(f"\n{status} {result.evaluator_name}")
        print(f"   Score: {result.value:.2f} ({result.label})")
        if result.explanation:
            # Truncate long explanations
            explanation = result.explanation[:200] + "..." if len(result.explanation) > 200 else result.explanation
            print(f"   Explanation: {explanation}")

    # Overall summary
    print("\n" + "=" * 70)
    overall_avg = sum(metrics.average_scores.values()) / len(metrics.average_scores) if metrics.average_scores else 0
    overall_pass = sum(metrics.pass_rates.values()) / len(metrics.pass_rates) if metrics.pass_rates else 0

    print(f"📈 OVERALL SUMMARY:")
    print(f"   Average Score: {overall_avg:.2f}")
    print(f"   Overall Pass Rate: {overall_pass:.1%}")

    if overall_pass >= 0.8:
        print("   Status: ✅ EXCELLENT - Agent performing well")
    elif overall_pass >= 0.6:
        print("   Status: ⚠️ ACCEPTABLE - Some improvements needed")
    else:
        print("   Status: ❌ NEEDS IMPROVEMENT - Review agent configuration")

    print("=" * 70 + "\n")

# In real production, you should save these results into an S3 Bucket
def save_results_json(
    results: list[EvaluationResult],
    metrics: AggregatedMetrics,
    output_path: str,
) -> None:
    """Save evaluation results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": metrics.session_id,
        "agent_id": metrics.agent_id,
        "summary": {
            "total_evaluations": metrics.total_evaluations,
            "average_scores": metrics.average_scores,
            "pass_rates": metrics.pass_rates,
        },
        "results": [
            {
                "evaluator_id": r.evaluator_id,
                "evaluator_name": r.evaluator_name,
                "value": r.value,
                "label": r.label,
                "explanation": r.explanation,
                "context": r.context,
                "token_usage": r.token_usage,
                "trace_id": r.trace_id,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


async def main():
    """CLI entry point for on-demand evaluation."""
    parser = argparse.ArgumentParser(
        description="Run on-demand evaluation for Restaurant Finder Agent"
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID to evaluate",
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Agent ID (defaults to RUNTIME_ID from settings)",
    )
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=None,
        help="Specific evaluators to use (space-separated)",
    )
    parser.add_argument(
        "--create-custom",
        action="store_true",
        help="Create and use custom evaluators",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)",
    )
    parser.add_argument(
        "--list-evaluators",
        action="store_true",
        help="List available evaluators and exit",
    )

    args = parser.parse_args()

    # List evaluators if requested
    if args.list_evaluators:
        client = EvaluationClient()
        evaluators = client.list_evaluators()
        print("\n📋 Available Evaluators:")
        print("-" * 50)
        for ev in evaluators.get("evaluators", []):
            print(f"  • {ev.get('evaluatorId')}: {ev.get('description', 'No description')}")
        return

    # Run evaluation
    agent_id = args.agent_id or settings.RUNTIME_ID
    if not agent_id:
        print("❌ Error: Agent ID required. Set RUNTIME_ID in .env or use --agent-id")
        return

    print(f"\n🔬 Running On-Demand Evaluation")
    print(f"   Session: {args.session_id}")
    print(f"   Agent:   {agent_id}")
    print(f"   Custom:  {'Yes' if args.create_custom else 'No'}")

    results, metrics = await evaluate_session(
        agent_id=agent_id,
        session_id=args.session_id,
        evaluators=args.evaluators,
        create_custom_evaluators=args.create_custom,
        output_dir=args.output_dir,
    )

    # Print report
    print_evaluation_report(results, metrics)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_report_{args.session_id}_{timestamp}.json"
    save_results_json(results, metrics, str(output_path))


if __name__ == "__main__":
    asyncio.run(main())
