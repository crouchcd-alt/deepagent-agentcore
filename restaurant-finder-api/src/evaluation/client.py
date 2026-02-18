"""
Evaluation client wrapper for AWS Bedrock AgentCore Evaluations.

Provides a unified interface for:
- Listing and managing evaluators (built-in and custom)
- Creating custom evaluators from JSON configurations
- Running on-demand evaluations
- Managing online evaluation configurations
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from bedrock_agentcore_starter_toolkit import Evaluation
from bedrock_agentcore_starter_toolkit.operations.observability.query_builder import (
    CloudWatchQueryBuilder,
)
from loguru import logger

from src.config import settings


def _patch_span_query_builder() -> None:
    """Patch the SDK's span query to skip the cloud.resource_id filter.

    The SDK filters spans by parsing ``resource.attributes.cloud.resource_id``
    to extract the agent ID.  However, AgentCore runtimes that use the
    ``aws-opentelemetry-distro`` auto-instrumentation do not emit that
    attribute, so the filter drops every span and the evaluation fails
    with "No spans found".

    This patch replaces the session query so it only filters by
    ``attributes.session.id`` (which *is* present on every span).
    """

    @staticmethod
    def _build(session_id: str, agent_id: str = "") -> str:
        return f"""fields @timestamp,
               @message,
               traceId,
               spanId,
               name as spanName,
               kind,
               status.code as statusCode,
               status.message as statusMessage,
               durationNano/1000000 as durationMs,
               attributes.session.id as sessionId,
               startTimeUnixNano,
               endTimeUnixNano,
               parentSpanId,
               events,
               resource.attributes.service.name as serviceName,
               resource.attributes.cloud.resource_id as resourceId,
               attributes.aws.remote.service as serviceType
        | filter attributes.session.id = '{session_id}'
        | sort startTimeUnixNano asc"""

    CloudWatchQueryBuilder.build_spans_by_session_query = _build  # type: ignore[assignment]


# Apply the patch at import time so every evaluation run picks it up.
_patch_span_query_builder()


# Built-in evaluators available in AgentCore Evaluations
BUILTIN_EVALUATORS = [
    "Builtin.Correctness",
    "Builtin.GoalSuccessRate",
    "Builtin.Helpfulness",
    "Builtin.Faithfulness",
    "Builtin.ToolSelectionAccuracy",
    "Builtin.ToolParameterAccuracy",
    "Builtin.Harmfulness",
    "Builtin.Maliciousness",
    "Builtin.Toxicity",
    "Builtin.Jailbreak",
    "Builtin.PromptInjection",
    "Builtin.Refusal",
    "Builtin.ContextRelevance",
]

# Custom evaluator configurations for the restaurant finder
CUSTOM_EVALUATOR_CONFIGS = {
    "response_quality": "response_quality.json",
    "recommendation_quality": "restaurant_recommendation_quality.json",
    "safety_compliance": "safety_compliance.json",
}


@dataclass
class EvaluationResult:
    """Result from a single evaluation."""

    evaluator_id: str
    evaluator_name: str
    value: float
    label: str
    explanation: str
    context: Optional[dict] = None
    token_usage: Optional[dict] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from multiple evaluations."""

    total_evaluations: int
    evaluator_scores: dict[str, list[float]]
    average_scores: dict[str, float]
    pass_rates: dict[str, float]  # Percentage of scores >= 0.7
    session_id: Optional[str] = None
    agent_id: Optional[str] = None


class EvaluationClient:
    """
    Client for AWS Bedrock AgentCore Evaluations.

    Provides methods for:
    - Listing available evaluators
    - Creating custom evaluators
    - Running on-demand evaluations
    - Managing online evaluation configurations

    Example:
        client = EvaluationClient()

        # List available evaluators
        evaluators = client.list_evaluators()

        # Create custom evaluator
        evaluator_id = client.create_custom_evaluator("response_quality")

        # Run evaluation
        results = client.run_evaluation(
            agent_id="my-agent",
            session_id="session-123",
            evaluators=["Builtin.Correctness", evaluator_id]
        )
    """

    def __init__(self, region: Optional[str] = None):
        """
        Initialize the evaluation client.

        Args:
            region: AWS region. Defaults to settings.AWS_REGION.
        """
        self.region = region or settings.AWS_REGION
        self._client: Optional[Evaluation] = None
        self._custom_evaluator_ids: dict[str, str] = {}
        self._metrics_dir = Path(__file__).parent / "metrics"

    @property
    def client(self) -> Evaluation:
        """Lazy initialization of the AgentCore Evaluation client."""
        if self._client is None:
            self._client = Evaluation(region=self.region)
            logger.info(f"Initialized AgentCore Evaluation client in region: {self.region}")
        return self._client

    def list_evaluators(self) -> dict:
        """
        List all available evaluators (built-in and custom).

        Returns:
            dict with 'evaluators' key containing list of evaluator details.
        """
        try:
            evaluators = self.client.list_evaluators()
            logger.info(f"Found {len(evaluators.get('evaluators', []))} evaluators")
            return evaluators
        except Exception as e:
            logger.error(f"Failed to list evaluators: {e}")
            raise

    def get_evaluator(self, evaluator_id: str) -> dict:
        """
        Get details of a specific evaluator.

        Args:
            evaluator_id: The evaluator ID (e.g., "Builtin.Correctness" or custom ID)

        Returns:
            dict with evaluator details.
        """
        try:
            return self.client.get_evaluator(evaluator_id=evaluator_id)
        except Exception as e:
            logger.error(f"Failed to get evaluator {evaluator_id}: {e}")
            raise

    def create_custom_evaluator(
        self,
        name: str,
        description: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> str:
        """
        Create a custom evaluator from a JSON configuration.

        Args:
            name: Name of the evaluator (matches key in CUSTOM_EVALUATOR_CONFIGS)
            description: Optional description. Defaults to name-based description.
            config_path: Optional path to config JSON. Defaults to predefined path.

        Returns:
            The evaluator ID for the created evaluator.
        """
        # Check if already created
        if name in self._custom_evaluator_ids:
            logger.info(f"Custom evaluator '{name}' already exists: {self._custom_evaluator_ids[name]}")
            return self._custom_evaluator_ids[name]

        # Get config path
        if config_path is None:
            if name not in CUSTOM_EVALUATOR_CONFIGS:
                raise ValueError(
                    f"Unknown evaluator '{name}'. Available: {list(CUSTOM_EVALUATOR_CONFIGS.keys())}"
                )
            config_path = self._metrics_dir / CUSTOM_EVALUATOR_CONFIGS[name]
        else:
            config_path = Path(config_path)

        # Load config
        if not config_path.exists():
            raise FileNotFoundError(f"Evaluator config not found: {config_path}")

        with open(config_path) as f:
            eval_config = json.load(f)

        # Create evaluator
        description = description or f"Custom evaluator for restaurant finder: {name}"

        try:
            response = self.client.create_evaluator(
                name=f"restaurant_finder_{name}",
                level="TRACE",  # Evaluate at trace level
                description=description,
                config=eval_config,
            )

            evaluator_id = response.get("evaluatorId")
            self._custom_evaluator_ids[name] = evaluator_id

            logger.info(f"Created custom evaluator '{name}': {evaluator_id}")
            return evaluator_id

        except Exception as e:
            logger.error(f"Failed to create custom evaluator '{name}': {e}")
            raise

    def create_all_custom_evaluators(self) -> dict[str, str]:
        """
        Create all predefined custom evaluators.

        Returns:
            dict mapping evaluator names to their IDs.
        """
        created = {}
        for name in CUSTOM_EVALUATOR_CONFIGS:
            try:
                evaluator_id = self.create_custom_evaluator(name)
                created[name] = evaluator_id
            except Exception as e:
                logger.warning(f"Failed to create evaluator '{name}': {e}")

        return created

    def run_evaluation(
        self,
        agent_id: str,
        session_id: str,
        evaluators: list[str],
        output_path: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """
        Run on-demand evaluation on a session.

        Args:
            agent_id: The AgentCore agent ID.
            session_id: The session ID to evaluate.
            evaluators: List of evaluator IDs (built-in or custom).
            output_path: Optional path to save results JSON.

        Returns:
            List of EvaluationResult objects.
        """
        logger.info(
            f"Running evaluation - Agent: {agent_id}, Session: {session_id}, "
            f"Evaluators: {evaluators}"
        )

        try:
            # Run evaluation via AgentCore SDK
            kwargs = {
                "agent_id": agent_id,
                "session_id": session_id,
                "evaluators": evaluators,
            }
            if output_path:
                kwargs["output"] = output_path

            response = self.client.run(**kwargs)

            # Parse results
            results = []
            for result in response.results:
                eval_result = EvaluationResult(
                    evaluator_id=getattr(result, "evaluator_id", ""),
                    evaluator_name=getattr(result, "evaluator_name", ""),
                    value=getattr(result, "value", 0.0),
                    label=getattr(result, "label", ""),
                    explanation=getattr(result, "explanation", ""),
                    context=getattr(result, "context", None),
                    token_usage=getattr(result, "token_usage", None),
                    trace_id=getattr(result, "trace_id", None),
                    span_id=getattr(result, "span_id", None),
                )
                results.append(eval_result)

            logger.info(f"Evaluation complete: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def aggregate_results(
        self,
        results: list[EvaluationResult],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AggregatedMetrics:
        """
        Aggregate evaluation results into summary metrics.

        Args:
            results: List of evaluation results.
            session_id: Optional session ID for context.
            agent_id: Optional agent ID for context.

        Returns:
            AggregatedMetrics with scores and pass rates.
        """
        evaluator_scores: dict[str, list[float]] = {}

        for result in results:
            name = result.evaluator_name or result.evaluator_id
            if name not in evaluator_scores:
                evaluator_scores[name] = []
            evaluator_scores[name].append(result.value)

        # Compute averages
        average_scores = {
            name: sum(scores) / len(scores)
            for name, scores in evaluator_scores.items()
        }

        # Compute pass rates (score >= 0.7)
        pass_rates = {
            name: sum(1 for s in scores if s >= 0.7) / len(scores)
            for name, scores in evaluator_scores.items()
        }

        return AggregatedMetrics(
            total_evaluations=len(results),
            evaluator_scores=evaluator_scores,
            average_scores=average_scores,
            pass_rates=pass_rates,
            session_id=session_id,
            agent_id=agent_id,
        )

    def get_recommended_evaluators(self) -> list[str]:
        """
        Get the recommended set of evaluators for restaurant finder.

        Returns:
            List of evaluator IDs (built-in + custom).
        """
        # Built-in evaluators relevant for restaurant finder
        recommended = [
            "Builtin.Correctness",
            "Builtin.GoalSuccessRate",
            "Builtin.Helpfulness",
            "Builtin.ToolSelectionAccuracy",
            "Builtin.ToolParameterAccuracy",
            "Builtin.Harmfulness",
        ]

        # Add custom evaluators if created
        recommended.extend(self._custom_evaluator_ids.values())

        return recommended
