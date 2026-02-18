"""
Online evaluation configuration for the Restaurant Finder Agent.

Provides functionality to:
- Setup continuous production monitoring
- Configure sampling rates and evaluators
- Manage online evaluation configurations
- Integrate with CloudWatch dashboards

Usage:
    # Setup online evaluation
    python -m src.evaluation.online --agent-id <agent_id> --sampling-rate 10

    # List existing configurations
    python -m src.evaluation.online --list

    # Delete a configuration
    python -m src.evaluation.online --delete --config-id <config_id>
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from bedrock_agentcore_starter_toolkit import Evaluation
from loguru import logger

from src.config import settings
from src.evaluation.client import EvaluationClient, BUILTIN_EVALUATORS


# Default evaluators for online monitoring
ONLINE_EVALUATORS = [
    "Builtin.GoalSuccessRate",
    "Builtin.Correctness",
    "Builtin.Helpfulness",
    "Builtin.ToolSelectionAccuracy",
    "Builtin.ToolParameterAccuracy",
    "Builtin.Harmfulness",
]


@dataclass
class OnlineEvalConfig:
    """Configuration for online evaluation."""

    config_id: str
    config_name: str
    agent_id: str
    sampling_rate: int
    evaluators: list[str]
    status: str
    created_at: Optional[datetime] = None


class OnlineEvaluationManager:
    """
    Manager for online evaluation configurations.

    Handles creation, retrieval, and management of online evaluation
    configurations for continuous production monitoring.

    Example:
        manager = OnlineEvaluationManager()

        # Setup online evaluation
        config = await manager.setup_online_evaluation(
            agent_id="my-agent",
            sampling_rate=10,  # Evaluate 10% of sessions
        )

        # List configurations
        configs = manager.list_configurations()

        # Delete configuration
        manager.delete_configuration(config.config_id)
    """

    def __init__(self, region: Optional[str] = None):
        """
        Initialize the online evaluation manager.

        Args:
            region: AWS region. Defaults to settings.AWS_REGION.
        """
        self.region = region or settings.AWS_REGION
        self._client: Optional[Evaluation] = None
        self._eval_client = EvaluationClient(region=self.region)

    @property
    def client(self) -> Evaluation:
        """Lazy initialization of the AgentCore Evaluation client."""
        if self._client is None:
            self._client = Evaluation(region=self.region)
        return self._client

    async def setup_online_evaluation(
        self,
        agent_id: str,
        config_name: Optional[str] = None,
        sampling_rate: int = 10,
        evaluators: Optional[list[str]] = None,
        include_custom: bool = True,
        description: Optional[str] = None,
    ) -> OnlineEvalConfig:
        """
        Setup online evaluation for an agent.

        Args:
            agent_id: The AgentCore agent ID.
            config_name: Optional configuration name. Auto-generated if not provided.
            sampling_rate: Percentage of sessions to evaluate (1-100). Default 10%.
            evaluators: Optional list of evaluator IDs. Defaults to ONLINE_EVALUATORS.
            include_custom: If True, create and include custom evaluators.
            description: Optional description for the configuration.

        Returns:
            OnlineEvalConfig with the created configuration details.
        """
        # Generate config name if not provided
        if config_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"restaurant_finder_eval_{timestamp}"

        # Determine evaluators
        eval_list = evaluators or ONLINE_EVALUATORS.copy()

        # Create custom evaluators if requested
        if include_custom:
            logger.info("Creating custom evaluators for online monitoring...")
            custom_ids = self._eval_client.create_all_custom_evaluators()
            eval_list.extend(custom_ids.values())
            logger.info(f"Added {len(custom_ids)} custom evaluators")

        # Create description
        if description is None:
            description = (
                f"Restaurant Finder Agent online evaluation. "
                f"Sampling {sampling_rate}% of sessions with {len(eval_list)} evaluators."
            )

        logger.info(f"Creating online evaluation configuration: {config_name}")
        logger.info(f"Agent ID: {agent_id}")
        logger.info(f"Sampling rate: {sampling_rate}%")
        logger.info(f"Evaluators: {len(eval_list)}")

        try:
            response = self.client.create_online_config(
                agent_id=agent_id,
                config_name=config_name,
                sampling_rate=sampling_rate,
                evaluator_list=eval_list,
                config_description=description,
                auto_create_execution_role=True,
            )

            config_id = response.get("onlineEvaluationConfigId")

            logger.info(f"Online evaluation configured: {config_id}")

            return OnlineEvalConfig(
                config_id=config_id,
                config_name=config_name,
                agent_id=agent_id,
                sampling_rate=sampling_rate,
                evaluators=eval_list,
                status="ENABLED",
                created_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to create online evaluation config: {e}")
            raise

    def get_configuration(self, config_id: str) -> dict:
        """
        Get details of an online evaluation configuration.

        Args:
            config_id: The configuration ID.

        Returns:
            dict with configuration details.
        """
        try:
            return self.client.get_online_config(config_id=config_id)
        except Exception as e:
            logger.error(f"Failed to get configuration {config_id}: {e}")
            raise

    def list_configurations(self) -> list[dict]:
        """
        List all online evaluation configurations.

        Returns:
            List of configuration details.
        """
        try:
            response = self.client.list_online_configs()
            return response.get("onlineEvaluationConfigs", [])
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            raise

    def delete_configuration(self, config_id: str) -> bool:
        """
        Delete an online evaluation configuration.

        Args:
            config_id: The configuration ID to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            self.client.delete_online_config(config_id=config_id)
            logger.info(f"Deleted online evaluation config: {config_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete configuration {config_id}: {e}")
            raise

    def pause_configuration(self, config_id: str) -> bool:
        """
        Pause an online evaluation configuration.

        Args:
            config_id: The configuration ID to pause.

        Returns:
            True if paused successfully.
        """
        try:
            self.client.update_online_config(
                config_id=config_id,
                status="PAUSED",
            )
            logger.info(f"Paused online evaluation config: {config_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause configuration {config_id}: {e}")
            raise

    def resume_configuration(self, config_id: str) -> bool:
        """
        Resume a paused online evaluation configuration.

        Args:
            config_id: The configuration ID to resume.

        Returns:
            True if resumed successfully.
        """
        try:
            self.client.update_online_config(
                config_id=config_id,
                status="ENABLED",
            )
            logger.info(f"Resumed online evaluation config: {config_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume configuration {config_id}: {e}")
            raise


async def setup_online_evaluation(
    agent_id: Optional[str] = None,
    sampling_rate: int = 10,
    evaluators: Optional[list[str]] = None,
    include_custom: bool = True,
) -> OnlineEvalConfig:
    """
    Convenience function to setup online evaluation.

    Args:
        agent_id: Optional agent ID. Defaults to settings.RUNTIME_ID.
        sampling_rate: Percentage of sessions to evaluate (1-100).
        evaluators: Optional list of evaluator IDs.
        include_custom: Whether to include custom evaluators.

    Returns:
        OnlineEvalConfig with the created configuration details.
    """
    agent_id = agent_id or settings.RUNTIME_ID

    if not agent_id:
        raise ValueError("Agent ID is required. Set RUNTIME_ID in settings or pass agent_id.")

    manager = OnlineEvaluationManager()
    return await manager.setup_online_evaluation(
        agent_id=agent_id,
        sampling_rate=sampling_rate,
        evaluators=evaluators,
        include_custom=include_custom,
    )


def print_config_details(config: dict) -> None:
    """Print configuration details in a formatted way."""
    print("\n" + "-" * 50)
    print(f"Config ID:     {config.get('onlineEvaluationConfigId', 'N/A')}")
    print(f"Config Name:   {config.get('configName', 'N/A')}")
    print(f"Agent ID:      {config.get('agentId', 'N/A')}")
    print(f"Status:        {config.get('status', 'N/A')}")
    print(f"Sampling Rate: {config.get('samplingRate', 'N/A')}%")

    evaluators = config.get("evaluators", [])
    print(f"Evaluators:    {len(evaluators)}")
    for ev in evaluators[:5]:  # Show first 5
        print(f"  • {ev}")
    if len(evaluators) > 5:
        print(f"  ... and {len(evaluators) - 5} more")


async def main():
    """CLI entry point for online evaluation management."""
    parser = argparse.ArgumentParser(
        description="Manage online evaluation for Restaurant Finder Agent"
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Agent ID (defaults to RUNTIME_ID from settings)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=10,
        help="Percentage of sessions to evaluate (1-100, default: 10)",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Custom configuration name",
    )
    parser.add_argument(
        "--no-custom",
        action="store_true",
        help="Exclude custom evaluators (use built-in only)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing configurations",
    )
    parser.add_argument(
        "--get",
        metavar="CONFIG_ID",
        help="Get details of a specific configuration",
    )
    parser.add_argument(
        "--delete",
        metavar="CONFIG_ID",
        help="Delete a configuration",
    )
    parser.add_argument(
        "--pause",
        metavar="CONFIG_ID",
        help="Pause a configuration",
    )
    parser.add_argument(
        "--resume",
        metavar="CONFIG_ID",
        help="Resume a paused configuration",
    )

    args = parser.parse_args()
    manager = OnlineEvaluationManager()

    # Handle list action
    if args.list:
        print("\n📋 Online Evaluation Configurations:")
        print("=" * 50)
        configs = manager.list_configurations()
        if not configs:
            print("No configurations found.")
        else:
            for config in configs:
                print_config_details(config)
        return

    # Handle get action
    if args.get:
        config = manager.get_configuration(args.get)
        print("\n📋 Configuration Details:")
        print("=" * 50)
        print_config_details(config)
        return

    # Handle delete action
    if args.delete:
        confirm = input(f"Delete configuration {args.delete}? (y/N): ")
        if confirm.lower() == "y":
            manager.delete_configuration(args.delete)
            print(f"✅ Configuration {args.delete} deleted.")
        else:
            print("Cancelled.")
        return

    # Handle pause action
    if args.pause:
        manager.pause_configuration(args.pause)
        print(f"⏸️  Configuration {args.pause} paused.")
        return

    # Handle resume action
    if args.resume:
        manager.resume_configuration(args.resume)
        print(f"▶️  Configuration {args.resume} resumed.")
        return

    # Create new configuration
    agent_id = args.agent_id or settings.RUNTIME_ID
    if not agent_id:
        print("❌ Error: Agent ID required. Set RUNTIME_ID in .env or use --agent-id")
        return

    print(f"\n🔧 Setting up Online Evaluation")
    print(f"   Agent:         {agent_id}")
    print(f"   Sampling Rate: {args.sampling_rate}%")
    print(f"   Custom Evals:  {'No' if args.no_custom else 'Yes'}")

    config = await manager.setup_online_evaluation(
        agent_id=agent_id,
        config_name=args.config_name,
        sampling_rate=args.sampling_rate,
        include_custom=not args.no_custom,
    )

    print("\n✅ Online Evaluation Configured!")
    print("=" * 50)
    print(f"Config ID:     {config.config_id}")
    print(f"Config Name:   {config.config_name}")
    print(f"Sampling Rate: {config.sampling_rate}%")
    print(f"Evaluators:    {len(config.evaluators)}")
    print(f"Status:        {config.status}")
    print("\n📊 View results in CloudWatch GenAI Observability dashboard:")
    print(f"   https://console.aws.amazon.com/cloudwatch/home#gen-ai-observability/agent-core/agents")


if __name__ == "__main__":
    asyncio.run(main())
