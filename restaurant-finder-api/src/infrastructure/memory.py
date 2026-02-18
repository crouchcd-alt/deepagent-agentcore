"""
Short-term memory manager using AgentCore Memory.

Memory is created and configured via CDK infrastructure (agentcore-stack.ts).
The MEMORY_ID environment variable must be set from the CDK stack output.

Memory strategies are defined in the CDK stack:
- UserPreferenceStrategy: /users/{actorId}/preferences
- SemanticStrategy: /conversations/{actorId}/facts
- SummaryStrategy: /conversations/{sessionId}/summaries
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langgraph_checkpoint_aws import AgentCoreMemorySaver
from bedrock_agentcore.memory import MemoryClient
from loguru import logger

from src.config import settings


# Singleton instance for application-wide use
_memory_instance: "ShortTermMemory | None" = None


def get_memory_instance() -> "ShortTermMemory":
    """Get or create the shared ShortTermMemory singleton instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ShortTermMemory()
    return _memory_instance


class ShortTermMemory:
    """
    Short-term memory manager using AgentCore Memory.

    The MEMORY_ID must be provided via environment variable.
    Memory creation is handled by CDK infrastructure, not at runtime.
    """

    def __init__(self):
        if not settings.MEMORY_ID:
            raise RuntimeError(
                "MEMORY_ID environment variable is required. "
                "Deploy the CDK stack and set MEMORY_ID from the stack output."
            )

        self._memory_id = settings.MEMORY_ID
        self._client = MemoryClient(region_name=settings.AWS_REGION)
        logger.info(f"Using MEMORY_ID from environment: {self._memory_id}")

    @property
    def memory_id(self) -> str:
        """Return the configured memory ID."""
        return self._memory_id

    def get_memory(self) -> AgentCoreMemorySaver:
        """Get the LangGraph checkpointer for state persistence."""
        return AgentCoreMemorySaver(
            memory_id=self._memory_id,
            region_name=settings.AWS_REGION
        )

    def _retrieve_from_namespace(
        self,
        namespace: str,
        query: str,
        actor_id: str,
        top_k: int,
        category: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Helper to retrieve memories from a single namespace."""
        try:
            results = self._client.retrieve_memories(
                memory_id=self._memory_id,
                namespace=namespace,
                query=query,
                actor_id=actor_id,
                top_k=top_k,
            )
            logger.debug(f"Retrieved {len(results)} {category}")
            return category, results
        except Exception as e:
            logger.warning(f"Failed to retrieve {category}: {e}")
            return category, []

    def retrieve_memories(
        self,
        query: str,
        actor_id: str,
        session_id: str,
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve all memory types before processing user input.

        Retrieves in parallel from:
        - User preferences (personalization data)
        - Semantic facts (conversation history facts)
        - Conversation summaries (session-based)

        Args:
            query: The user's input message to search against
            actor_id: The user/actor identifier
            session_id: The conversation session identifier
            top_k: Number of results to retrieve per namespace

        Returns:
            Dictionary with retrieved memories by category
        """
        return self.retrieve_specific_memories(
            query=query,
            actor_id=actor_id,
            session_id=session_id,
            memory_types=["preferences", "facts", "summaries"],
            top_k=top_k,
        )

    def retrieve_specific_memories(
        self,
        query: str,
        actor_id: str,
        session_id: str,
        memory_types: list[str],
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve specific memory types in parallel.

        This method allows selective retrieval of memory types for efficiency.
        Memory types are retrieved in parallel for speed optimization.

        Available memory types:
        - "preferences": User preferences (dietary restrictions, favorite cuisines, etc.)
        - "facts": Semantic facts extracted from conversation history
        - "summaries": Conversation summaries from the current session

        Args:
            query: The user's input message to search against
            actor_id: The user/actor identifier
            session_id: The conversation session identifier
            memory_types: List of memory types to retrieve (e.g., ["preferences", "facts"])
            top_k: Number of results to retrieve per namespace

        Returns:
            Dictionary with retrieved memories by category
        """
        retrieved = {}

        # Map memory types to their namespaces
        type_to_namespace = {
            "preferences": f"/users/{actor_id}/preferences",
            "facts": f"/conversations/{actor_id}/facts",
            "summaries": f"/conversations/{session_id}/summaries",
        }

        # Filter to only requested memory types
        retrieval_tasks = [
            (type_to_namespace[mem_type], mem_type)
            for mem_type in memory_types
            if mem_type in type_to_namespace
        ]

        if not retrieval_tasks:
            logger.warning(f"No valid memory types specified: {memory_types}")
            return retrieved

        # Execute all retrievals in parallel
        with ThreadPoolExecutor(max_workers=len(retrieval_tasks)) as executor:
            futures = {
                executor.submit(
                    self._retrieve_from_namespace,
                    namespace,
                    query,
                    actor_id,
                    top_k,
                    category,
                ): category
                for namespace, category in retrieval_tasks
            }

            for future in as_completed(futures):
                try:
                    category, results = future.result()
                    retrieved[category] = results
                except Exception as e:
                    category = futures[future]
                    logger.warning(f"Parallel retrieval failed for {category}: {e}")
                    retrieved[category] = []

        return retrieved

    def process_turn(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
    ) -> dict[str, Any]:
        """
        Post-hook: Process and save the conversation turn to memory.

        This triggers all memory strategies configured in CDK:
        - User preference extraction and storage
        - Semantic fact extraction
        - Conversation summarization

        Args:
            actor_id: The user/actor identifier
            session_id: The conversation session identifier
            user_input: The user's message
            agent_response: The agent's response

        Returns:
            Dictionary with processing results
        """
        try:
            retrieved_memories, event_info = self._client.process_turn(
                memory_id=self._memory_id,
                actor_id=actor_id,
                session_id=session_id,
                user_input=user_input,
                agent_response=agent_response,
            )
            logger.info(f"Processed conversation turn for actor={actor_id}, session={session_id}")
            return {
                "success": True,
                "retrieved_memories": retrieved_memories,
                "event_info": event_info,
            }
        except Exception as e:
            logger.error(f"Failed to process conversation turn: {e}")
            return {
                "success": False,
                "error": str(e),
            }
