"""
Comprehensive evaluation runner for the Restaurant Finder Agent.

This module provides a complete evaluation workflow including:
- Agent invocation with test cases
- On-demand evaluation execution
- Results collection and reporting
- Integration with AgentCore Observability

Usage:
    # Run full evaluation suite
    python -m src.evaluation.runner --agent-id <agent_id>

    # Run specific test categories
    python -m src.evaluation.runner --agent-id <agent_id> --categories basic_search dietary_search

    # Run with custom evaluators
    python -m src.evaluation.runner --agent-id <agent_id> --create-custom
"""

import argparse
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from loguru import logger

from src.config import settings
from src.evaluation.client import EvaluationClient, AggregatedMetrics
from src.evaluation.test_cases import (
    RESTAURANT_EVAL_CASES,
    EvalTestCase,
    TestCategory,
    get_test_cases_by_category,
)
from src.evaluation.on_demand import (
    evaluate_session,
    print_evaluation_report,
    save_results_json,
    DEFAULT_EVALUATORS,
)


class EvaluationRunner:
    """
    End-to-end evaluation runner for the Restaurant Finder Agent.

    Orchestrates the full evaluation workflow:
    1. Invokes the agent with test cases
    2. Captures session/trace IDs
    3. Runs AgentCore Evaluations
    4. Aggregates and reports results

    Example:
        runner = EvaluationRunner(agent_id="my-agent", agent_arn="arn:...")

        # Run full suite
        results = await runner.run_full_evaluation()

        # Run specific categories
        results = await runner.run_evaluation(
            categories=[TestCategory.BASIC_SEARCH, TestCategory.DIETARY_SEARCH]
        )
    """

    def __init__(
        self,
        agent_id: str,
        agent_arn: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize the evaluation runner.

        Args:
            agent_id: The AgentCore agent ID.
            agent_arn: Optional agent ARN for invocation.
            region: AWS region. Defaults to settings.AWS_REGION.
        """
        self.agent_id = agent_id
        self.agent_arn = agent_arn
        self.region = region or settings.AWS_REGION

        self._agentcore_client = boto3.client(
            "bedrock-agentcore",
            region_name=self.region,
        )
        self._eval_client = EvaluationClient(region=self.region)

    async def invoke_agent(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Invoke the agent with a prompt.

        Args:
            prompt: The user prompt to send.
            session_id: Optional session ID for continuity.

        Returns:
            dict with response and session metadata.
        """
        if not self.agent_arn:
            raise ValueError("Agent ARN required for invocation")

        session_id = session_id or str(uuid.uuid4())

        try:
            response = self._agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=self.agent_arn,
                qualifier="DEFAULT",
                runtimeSessionId=session_id,
                payload=json.dumps({
                    "prompt": prompt,
                    "conversation_id": session_id,
                }),
            )

            # Collect response content
            content = []
            content_type = response.get("contentType", "")

            if "text/event-stream" in content_type:
                for line in response["response"].iter_lines(chunk_size=1):
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            content.append(line[6:])
            else:
                try:
                    for event in response.get("response", []):
                        content.append(event.decode("utf-8"))
                except Exception:
                    pass

            return {
                "session_id": session_id,
                "prompt": prompt,
                "response": "\n".join(content),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            return {
                "session_id": session_id,
                "prompt": prompt,
                "response": None,
                "success": False,
                "error": str(e),
            }

    async def run_test_cases(
        self,
        test_cases: list[EvalTestCase],
        session_id: Optional[str] = None,
    ) -> tuple[str, list[dict]]:
        """
        Run a set of test cases against the agent.

        Args:
            test_cases: List of test cases to run.
            session_id: Optional session ID. Auto-generated if not provided.

        Returns:
            Tuple of (session_id, list of invocation results).
        """
        session_id = session_id or f"eval-{uuid.uuid4()}"

        results = []
        total = len(test_cases)

        logger.info(f"Running {total} test cases in session: {session_id}")

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"[{i}/{total}] Running test: {test_case.id}")
            logger.debug(f"  Prompt: {test_case.prompt[:50]}...")

            result = await self.invoke_agent(
                prompt=test_case.prompt,
                session_id=session_id,
            )
            result["test_case"] = {
                "id": test_case.id,
                "category": test_case.category.value,
                "expected_behavior": test_case.expected_behavior,
                "expected_tools": test_case.expected_tools,
            }
            results.append(result)

            # Small delay between invocations
            await asyncio.sleep(0.5)

        logger.info(f"Completed {total} test cases")
        return session_id, results

    async def run_evaluation(
        self,
        categories: Optional[list[TestCategory]] = None,
        test_cases: Optional[list[EvalTestCase]] = None,
        evaluators: Optional[list[str]] = None,
        create_custom_evaluators: bool = False,
        output_dir: str = "evaluation_results",
    ) -> dict:
        """
        Run a complete evaluation workflow.

        Args:
            categories: Optional list of test categories to run.
            test_cases: Optional explicit list of test cases (overrides categories).
            evaluators: Optional list of evaluator IDs.
            create_custom_evaluators: If True, create custom evaluators.
            output_dir: Directory for saving results.

        Returns:
            dict with complete evaluation results.
        """
        # Determine test cases
        if test_cases is None:
            if categories:
                test_cases = []
                for cat in categories:
                    test_cases.extend(get_test_cases_by_category(cat))
            else:
                test_cases = RESTAURANT_EVAL_CASES

        logger.info(f"Starting evaluation with {len(test_cases)} test cases")

        # Step 1: Run test cases to generate traces
        session_id, invocation_results = await self.run_test_cases(test_cases)

        # Give CloudWatch time to ingest traces (typically 30-60s)
        logger.info("Waiting for trace ingestion (45s)...")
        await asyncio.sleep(45)

        # Step 2: Run AgentCore Evaluations
        eval_results, metrics = await evaluate_session(
            agent_id=self.agent_id,
            session_id=session_id,
            evaluators=evaluators,
            create_custom_evaluators=create_custom_evaluators,
            output_dir=output_dir,
        )

        # Step 3: Compile comprehensive results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comprehensive_results = {
            "evaluation_id": f"eval_{timestamp}",
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "session_id": session_id,
            "test_summary": {
                "total_test_cases": len(test_cases),
                "successful_invocations": sum(1 for r in invocation_results if r["success"]),
                "failed_invocations": sum(1 for r in invocation_results if not r["success"]),
                "categories": list(set(tc.category.value for tc in test_cases)),
            },
            "evaluation_summary": {
                "total_evaluations": metrics.total_evaluations,
                "average_scores": metrics.average_scores,
                "pass_rates": metrics.pass_rates,
            },
            "invocation_results": invocation_results,
            "evaluation_results": [
                {
                    "evaluator": r.evaluator_name,
                    "value": r.value,
                    "label": r.label,
                    "explanation": r.explanation,
                }
                for r in eval_results
            ],
        }

        # Save comprehensive results
        results_path = output_dir / f"comprehensive_eval_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(comprehensive_results, f, indent=2)

        logger.info(f"Comprehensive results saved to: {results_path}")

        return comprehensive_results

    async def run_full_evaluation(
        self,
        create_custom_evaluators: bool = True,
        output_dir: str = "evaluation_results",
    ) -> dict:
        """
        Run the full evaluation suite with all test cases.

        Args:
            create_custom_evaluators: If True, create custom evaluators.
            output_dir: Directory for saving results.

        Returns:
            dict with complete evaluation results.
        """
        return await self.run_evaluation(
            categories=None,  # All categories
            create_custom_evaluators=create_custom_evaluators,
            output_dir=output_dir,
        )


def print_comprehensive_report(results: dict) -> None:
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 70)
    print("🔬 RESTAURANT FINDER AGENT - COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)

    print(f"\nEvaluation ID: {results.get('evaluation_id')}")
    print(f"Timestamp:     {results.get('timestamp')}")
    print(f"Agent ID:      {results.get('agent_id')}")
    print(f"Session ID:    {results.get('session_id')}")

    # Test Summary
    test_summary = results.get("test_summary", {})
    print("\n" + "-" * 70)
    print("📋 TEST EXECUTION SUMMARY")
    print("-" * 70)
    print(f"Total Test Cases:       {test_summary.get('total_test_cases', 0)}")
    print(f"Successful Invocations: {test_summary.get('successful_invocations', 0)}")
    print(f"Failed Invocations:     {test_summary.get('failed_invocations', 0)}")
    print(f"Categories Tested:      {', '.join(test_summary.get('categories', []))}")

    # Evaluation Summary
    eval_summary = results.get("evaluation_summary", {})
    print("\n" + "-" * 70)
    print("📊 EVALUATION METRICS")
    print("-" * 70)
    print(f"Total Evaluations: {eval_summary.get('total_evaluations', 0)}")

    avg_scores = eval_summary.get("average_scores", {})
    pass_rates = eval_summary.get("pass_rates", {})

    print(f"\n{'Evaluator':<40} {'Avg Score':>10} {'Pass Rate':>12}")
    print("-" * 62)

    for name, avg in sorted(avg_scores.items()):
        rate = pass_rates.get(name, 0)
        status = "✅" if rate >= 0.7 else "⚠️" if rate >= 0.5 else "❌"
        print(f"{name:<40} {avg:>10.2f} {rate:>10.1%} {status}")

    # Overall Score
    print("\n" + "-" * 70)
    print("📈 OVERALL ASSESSMENT")
    print("-" * 70)

    if avg_scores:
        overall_avg = sum(avg_scores.values()) / len(avg_scores)
        overall_pass = sum(pass_rates.values()) / len(pass_rates) if pass_rates else 0

        print(f"Overall Average Score: {overall_avg:.2f}")
        print(f"Overall Pass Rate:     {overall_pass:.1%}")

        if overall_pass >= 0.8:
            print("\n✅ EXCELLENT - Agent is performing well across all metrics")
        elif overall_pass >= 0.6:
            print("\n⚠️ ACCEPTABLE - Some areas need improvement")
        else:
            print("\n❌ NEEDS IMPROVEMENT - Review agent configuration and prompts")

    print("\n" + "=" * 70)


async def main():
    """CLI entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation for Restaurant Finder Agent"
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="AgentCore Agent ID",
    )
    parser.add_argument(
        "--agent-arn",
        default=None,
        help="Agent ARN for invocation (required for test case execution)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Existing session ID to evaluate (skips test case execution)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in TestCategory],
        help="Test categories to run",
    )
    parser.add_argument(
        "--create-custom",
        action="store_true",
        help="Create and use custom evaluators",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory for results (default: evaluation_results)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories",
    )
    parser.add_argument(
        "--list-test-cases",
        action="store_true",
        help="List all test cases",
    )

    args = parser.parse_args()

    # Handle list actions
    if args.list_categories:
        print("\n📋 Available Test Categories:")
        for cat in TestCategory:
            cases = get_test_cases_by_category(cat)
            print(f"  • {cat.value}: {len(cases)} test cases")
        return

    if args.list_test_cases:
        print("\n📋 All Test Cases:")
        for tc in RESTAURANT_EVAL_CASES:
            print(f"\n  [{tc.id}] ({tc.category.value})")
            print(f"    Prompt: {tc.prompt[:60]}...")
            print(f"    Expected Tools: {tc.expected_tools}")
        return

    # Evaluate existing session
    if args.session_id:
        print(f"\n🔬 Evaluating Existing Session: {args.session_id}")
        eval_results, metrics = await evaluate_session(
            agent_id=args.agent_id,
            session_id=args.session_id,
            create_custom_evaluators=args.create_custom,
            output_dir=args.output_dir,
        )
        print_evaluation_report(eval_results, metrics)
        return

    # Run full evaluation
    if not args.agent_arn:
        print("❌ Error: --agent-arn required for running test cases")
        print("   Use --session-id to evaluate an existing session instead")
        return

    runner = EvaluationRunner(
        agent_id=args.agent_id,
        agent_arn=args.agent_arn,
        region=settings.AWS_REGION,
    )

    # Determine categories
    categories = None
    if args.categories:
        categories = [TestCategory(c) for c in args.categories]

    print(f"\n🔬 Starting Comprehensive Evaluation")
    print(f"   Agent ID:  {args.agent_id}")
    print(f"   Agent ARN: {args.agent_arn}")
    if categories:
        print(f"   Categories: {[c.value for c in categories]}")
    else:
        print(f"   Categories: All ({len(RESTAURANT_EVAL_CASES)} test cases)")

    results = await runner.run_evaluation(
        categories=categories,
        create_custom_evaluators=args.create_custom,
        output_dir=args.output_dir,
    )

    print_comprehensive_report(results)


if __name__ == "__main__":
    asyncio.run(main())
