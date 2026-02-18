from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState
from src.application.orchestrator.workflow.nodes import (
    router_node,
    search_agent_node,
    simple_response_node,
    memory_post_hook,
)
from src.application.orchestrator.workflow.edges import (
    route_by_intent,
    should_continue_search_agent,
)
from src.application.orchestrator.workflow.tools import get_orchestrator_tools
from src.infrastructure.memory import ShortTermMemory


# Module-level graph instance (created lazily)
_graph_instance = None


def create_orchestrator_graph(force_recreate: bool = False):
    """
    Create the workflow graph with Router + Search Agent (ReAct pattern).

    Uses a module-level singleton pattern instead of @lru_cache to support
    dynamic tool loading based on configuration changes.

    Note: Guardrails are applied at the API layer (utils.py), not in the graph.
    This keeps the graph focused on orchestration logic.

    Args:
        force_recreate: If True, recreates the graph even if one exists.
                       Useful when configuration changes at runtime.

    Architecture (Router + Search Agent Pattern):

        START
          │
          ▼
    ┌───────────────┐
    │    Router     │  (Intent Classification)
    │   (Classify)  │
    └───────┬───────┘
            │
            ▼
    [route by intent]
       │              │
       │restaurant    │simple/off_topic
       │_search       │
       ▼              ▼
    ┌───────────────┐  ┌───────────────┐
    │ Search Agent  │  │Simple Response│
    │   (ReAct)     │  │  (No Tools)   │
    └───────┬───────┘  └───────┬───────┘
            │                  │
            ▼                  │
    [has tool calls?]          │
       │         │             │
    yes│         │no           │
       ▼         │             │
    ┌─────────┐  │             │
    │ToolNode │  │             │
    │ (Act)   │  │             │
    └────┬────┘  │             │
         │       │             │
         └───────┘             │
                 │             │
                 ▼             ▼
            ┌───────────────────┐
            │  Memory Post-Hook │
            └─────────┬─────────┘
                      │
                      ▼
                     END

    Router Node:
    - Classifies intent: restaurant_search, simple, off_topic
    - Routes to appropriate handler

    Search Agent (for restaurant_search):
    - Implements ReAct pattern (Reasoning + Acting)
    - Reasoning happens internally in the LLM (not exposed in output)
    - Acting is done via native tool calls (LangChain tool_calls)
    - Observation is the tool result returned to the agent
    - Loop continues until agent responds without tool calls

    Simple Response Node (for simple/off_topic):
    - Direct LLM response without tools
    - Handles greetings, thanks, off-topic redirections

    The search agent decides which tools to call:
    - restaurant_data_tool: MCP Gateway for structured restaurant data (always available)
    - restaurant_explorer_tool: Browser-based web search (if ENABLE_BROWSER_TOOLS=True)
    - restaurant_research_tool: Detailed restaurant research (if ENABLE_BROWSER_TOOLS=True)
    - memory_retrieval_tool: On-demand memory retrieval (always available)

    Memory:
    - Retrieval: On-demand via memory_retrieval_tool (agent calls when needed)
    - Post-hook: Saves the conversation turn to trigger memory strategies
    """
    global _graph_instance

    if _graph_instance is not None and not force_recreate:
        return _graph_instance

    logger.info("Creating workflow graph (Router + Search Agent pattern)...")

    graph_builder = StateGraph(OrchestratorState)

    # Add the router node (intent classification)
    graph_builder.add_node("router_node", router_node)

    # Add the search agent node (implements ReAct reasoning for restaurant searches)
    graph_builder.add_node("search_agent_node", search_agent_node)

    # Add the simple response node (handles non-restaurant queries)
    graph_builder.add_node("simple_response_node", simple_response_node)

    # Get tools dynamically based on current config (respects ENABLE_BROWSER_TOOLS)
    tools = get_orchestrator_tools()
    tool_node = ToolNode(tools)
    graph_builder.add_node("tool_node", tool_node)

    # Add the memory post-hook node (saves conversation turn after response)
    graph_builder.add_node("memory_post_hook", memory_post_hook)

    # Define edges
    # START -> Router (intent classification)
    graph_builder.add_edge(START, "router_node")

    # Conditional edge from router - route by intent
    # - restaurant_search: route to search_agent (full ReAct with tools)
    # - simple/off_topic: route to simple_response (no tools)
    graph_builder.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "search_agent": "search_agent_node",
            "simple_response": "simple_response_node",
        },
    )

    # Conditional edge from search agent - ReAct loop
    # - If tool calls: route to tools, then back to search_agent
    # - If no tool calls (Final Answer): route to memory hook, then END
    graph_builder.add_conditional_edges(
        "search_agent_node",
        should_continue_search_agent,
        {
            "tools": "tool_node",
            "end": "memory_post_hook",
        },
    )

    # After tools (Observation), return to search agent for next Thought/Action
    graph_builder.add_edge("tool_node", "search_agent_node")

    # Simple response -> Memory Post-Hook
    graph_builder.add_edge("simple_response_node", "memory_post_hook")

    # Memory Post-Hook -> END
    graph_builder.add_edge("memory_post_hook", END)

    # Setup Short-Term Memory (STM) checkpointer
    checkpointer = ShortTermMemory().get_memory()

    _graph_instance = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Workflow graph (Router + Search Agent) created successfully")

    return _graph_instance


def reset_graph():
    """Reset the cached graph instance. Call this if configuration changes."""
    global _graph_instance
    _graph_instance = None
    logger.info("Graph instance reset - will be recreated on next use")
