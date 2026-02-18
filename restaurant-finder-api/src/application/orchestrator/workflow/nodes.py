import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from src.application.orchestrator.workflow.state import OrchestratorState, IntentType
from src.application.orchestrator.workflow.chains import (
    get_search_agent_chain,
    get_router_chain,
    get_simple_response_chain,
)
from src.infrastructure.model import extract_text_content as _extract_text_content
from src.infrastructure.memory import get_memory_instance
from src.infrastructure.observability import get_observability_manager


async def search_agent_node(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Search agent node implementing the ReAct (Reasoning + Acting) pattern.

    This node handles restaurant search requests using the search agent chain
    which has tools bound for finding and researching restaurants.

    ReAct Pattern:
    1. Thought: Agent reasons about what to do next
    2. Action: Agent calls a tool OR provides Final Answer
    3. Observation: Tool results are returned (handled by ToolNode)
    4. Loop back to step 1 until Final Answer

    The LLM decides which tools to call based on the user's request:
    - memory_retrieval_tool: For fetching user preferences/facts/summaries
    - restaurant_data_tool: For MCP Gateway searches
    - restaurant_explorer_tool: For web-based searches
    - restaurant_research_tool: For detailed restaurant research

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration with customer context.

    Returns:
        Updated state with the search agent's response.
    """
    observability = get_observability_manager()
    start_time = time.time()

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")
    session_id = configurable.get("thread_id", "unknown")
    actor_id = configurable.get("actor_id", "unknown")
    tool_call_count = state.get("tool_call_count", 0)
    react_iteration = tool_call_count + 1  # Track which ReAct loop iteration

    messages = list(state["messages"])

    # Get the chain and prompt metadata for tracing
    chain_result = get_search_agent_chain(customer_name=customer_name)
    prompt_meta = chain_result.prompt_metadata

    logger.debug(
        f"Search agent invoked: iteration={react_iteration}, "
        f"messages={len(messages)}, prompt_version={prompt_meta.version}"
    )

    # Build comprehensive span attributes for observability
    span_attributes = {
        # Customer/session context
        "customer.name": customer_name,
        "session.id": session_id,
        "actor.id": actor_id,
        # Prompt metadata (for prompt version tracking)
        "prompt.name": prompt_meta.name,
        "prompt.version": prompt_meta.version or "unknown",
        "prompt.id": prompt_meta.id or "unknown",
        # ReAct loop state
        "react.iteration": react_iteration,
        "react.tool_call_count": tool_call_count,
        # Request context
        "message.count": len(messages),
        "input.token_estimate": sum(
            len(str(m.content)) // 4 for m in messages if hasattr(m, "content")
        ),
    }

    with observability.create_span(
        "search_agent.invoke",
        attributes=span_attributes,
    ):
        response = await chain_result.chain.ainvoke(
            {"messages": messages},
            config,
        )

    # Track tool calls for efficiency limiting
    has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
    new_tool_count = tool_call_count + (len(response.tool_calls) if has_tool_calls else 0)
    tool_names = [tc.get("name", "unknown") for tc in response.tool_calls] if has_tool_calls else []

    # Record workflow step completion with comprehensive metadata
    duration_ms = (time.time() - start_time) * 1000
    observability.record_workflow_step(
        step_name="search_agent",
        step_type="node",
        duration_ms=duration_ms,
        success=True,
        metadata={
            # Prompt tracking
            "prompt.name": prompt_meta.name,
            "prompt.version": prompt_meta.version or "unknown",
            # ReAct state
            "react.iteration": str(react_iteration),
            "react.has_tool_calls": str(has_tool_calls),
            "react.tool_names": ",".join(tool_names) if tool_names else "none",
            # Response metrics
            "response.has_content": str(bool(response.content)),
            "output.token_estimate": str(len(str(response.content)) // 4) if response.content else "0",
        }
    )

    if has_tool_calls:
        logger.debug(f"Search agent requested tools: {tool_names}")
    else:
        logger.debug("Search agent provided Final Answer (no tool calls)")

    return {
        "messages": response,
        "tool_call_count": new_tool_count,
        "made_tool_calls": state.get("made_tool_calls", False) or has_tool_calls,
    }


async def router_node(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Router node that classifies user intent for routing decisions.

    Intent types:
    - restaurant_search: Route to the search agent with tools
    - simple: Route to simple response (no tools needed)
    - off_topic: Route to simple response with redirect

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration.

    Returns:
        Updated state with the classified intent.
    """
    observability = get_observability_manager()
    start_time = time.time()

    messages = list(state["messages"])

    router_chain = get_router_chain()

    with observability.create_span(
        "router.classify",
        attributes={"message.count": len(messages)},
    ):
        response = await router_chain.ainvoke(
            {"messages": messages},
            config,
        )

    # Parse the intent from the response
    response_text = _extract_text_content(response.content).strip().lower()

    # Map response to intent type
    if "restaurant_search" in response_text:
        intent: IntentType = "restaurant_search"
    elif "simple" in response_text:
        intent = "simple"
    elif "off_topic" in response_text:
        intent = "off_topic"
    else:
        # Default to restaurant_search if unclear
        logger.warning(f"Unclear intent classification: {response_text}, defaulting to restaurant_search")
        intent = "restaurant_search"

    duration_ms = (time.time() - start_time) * 1000
    observability.record_workflow_step(
        step_name="router",
        step_type="node",
        duration_ms=duration_ms,
        success=True,
        metadata={"intent": intent, "raw_response": response_text[:50]},
    )

    logger.info(f"Router classified intent: {intent}")

    return {"intent": intent}


async def simple_response_node(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Simple response node for handling non-restaurant queries.

    This node generates responses for:
    - Greetings and welcomes
    - Thanks and acknowledgments
    - Questions about the assistant's capabilities
    - Off-topic redirections

    No tools are invoked - just a direct LLM response.

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration with customer context.

    Returns:
        Updated state with the simple response.
    """
    observability = get_observability_manager()
    start_time = time.time()

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")
    intent = state.get("intent", "simple")

    messages = list(state["messages"])

    # Get the simple response chain
    simple_chain = get_simple_response_chain(customer_name=customer_name)

    with observability.create_span(
        "simple_response.generate",
        attributes={
            "customer.name": customer_name,
            "intent": intent,
        },
    ):
        response = await simple_chain.ainvoke(
            {"messages": messages},
            config,
        )

    duration_ms = (time.time() - start_time) * 1000
    observability.record_workflow_step(
        step_name="simple_response",
        step_type="node",
        duration_ms=duration_ms,
        success=True,
        metadata={"intent": intent},
    )

    logger.debug(f"Simple response generated for intent: {intent}")

    return {"messages": response}


async def memory_post_hook(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict:
    """
    Post-hook node: Save the conversation turn to memory after processing.

    This triggers all memory strategies configured in the CDK stack:
    - Extracts and stores user preferences
    - Extracts semantic facts from the conversation
    - Updates conversation summaries

    Args:
        state: The orchestrator state containing messages.
        config: Runtime configuration with actor/session identifiers.

    Returns:
        Empty dict (no state changes, just side effects).
    """
    observability = get_observability_manager()
    start_time = time.time()

    configurable = config.get("configurable", {})
    actor_id = configurable.get("actor_id", "user:default")
    session_id = configurable.get("thread_id", "default_session")

    messages = state.get("messages", [])

    # Find the latest user input and agent response
    user_input = ""
    agent_response = ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not agent_response:
            # Skip tool calls, get the actual response
            if msg.content and not msg.tool_calls:
                # Extract text from content (handles both string and list formats)
                agent_response = _extract_text_content(msg.content)
        elif isinstance(msg, HumanMessage) and not user_input:
            user_input = _extract_text_content(msg.content)

        if user_input and agent_response:
            break

    if not user_input or not agent_response:
        logger.debug("Missing user input or agent response, skipping memory save")
        observability.add_span_event(
            "memory.skipped",
            attributes={"reason": "missing_input_or_response"}
        )
        return {}

    memory = get_memory_instance()

    try:
        with observability.create_span(
            "memory.process_turn",
            attributes={
                "actor.id": actor_id,
                "session.id": session_id,
            }
        ):
            result = memory.process_turn(
                actor_id=actor_id,
                session_id=session_id,
                user_input=user_input,
                agent_response=agent_response,
            )

        if result.get("success"):
            logger.info(f"Saved conversation turn to memory for actor={actor_id}")
            duration_ms = (time.time() - start_time) * 1000
            observability.record_workflow_step(
                step_name="memory_post_hook",
                step_type="node",
                duration_ms=duration_ms,
                success=True,
                metadata={"actor_id": actor_id, "session_id": session_id}
            )
        else:
            logger.warning(f"Memory save returned error: {result.get('error')}")
            observability.add_span_event(
                "memory.error",
                attributes={"error": result.get("error", "unknown")}
            )

    except Exception as e:
        logger.error(f"Memory post-hook failed: {e}")
        observability.add_span_event(
            "memory.exception",
            attributes={"error.type": type(e).__name__, "error.message": str(e)}
        )

    return {}
