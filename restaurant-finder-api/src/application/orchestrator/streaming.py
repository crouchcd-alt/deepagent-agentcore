"""
Orchestrator streaming — the single module for response generation and streaming.

Provides the public entry point (get_streaming_response) and the two streaming
strategies it delegates to (token-level via astream_events, buffered via ainvoke).

Node filtering is required because LangGraph's astream_events emits tokens from
every model call in the graph. Without it, internal outputs (router classification
text like "restaurant_search", memory hook processing) would leak to the user.
We use event tags to decide what to stream — tags carry the originating node name.
"""

import re
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from loguru import logger

from src.application.orchestrator.workflow.graph import create_orchestrator_graph
from src.infrastructure.model import extract_text_content


# ---------------------------------------------------------------------------
# Node classification — which nodes produce user-facing output
# ---------------------------------------------------------------------------
_RESPONSE_NODES = {"search_agent_node", "simple_response_node"}
_INTERNAL_NODES = {"router_node", "memory_post_hook"}

# Patterns that indicate internal content the model may stream that should
# not be shown to the user (tool-call XML, thinking tags, etc.)
_FILTERED_PATTERNS = [
    "<function", "</function",
    "<antml", "</antml", "antml::", "antml:invoke", "antml:parameter",
    "<tml", "</tml",
    'name="restaurant_', 'name="memory_',
    "<tool_call>", "</tool_call>",
    "<thinking", "</thinking>",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def get_streaming_response(
    messages: str | list[str],
    customer_name: str = "Guest",
    conversation_id: str | None = None,
    enable_true_streaming: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Run the orchestrator workflow and yield response tokens.

    Args:
        messages: User message(s) to process.
        customer_name: Name of the customer for personalization.
        conversation_id: Unique ID for the conversation thread.
        enable_true_streaming: Token streaming (True) or buffer mode (False).

    Yields:
        Response content tokens/chunks.
    """
    graph = create_orchestrator_graph()

    try:
        thread_id = conversation_id or "default-thread"
        actor_id = _sanitize_actor_id(customer_name)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "customer_name": customer_name,
                "actor_id": actor_id,
            }
        }

        input_data = {
            "messages": _format_messages(messages),
            "customer_name": customer_name,
        }

        logger.info(f"Starting workflow execution (thread_id={thread_id}, streaming={enable_true_streaming})")

        streamer = _stream_with_events if enable_true_streaming else _stream_buffered
        async for chunk in streamer(graph, input_data, config):
            yield chunk

    except Exception as e:
        logger.error(f"Error in get_streaming_response: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Error running conversation workflow: {str(e)}") from e


# ---------------------------------------------------------------------------
# Streaming strategies
# ---------------------------------------------------------------------------

async def _stream_with_events(
    graph,
    input_data: dict,
    config: dict,
) -> AsyncGenerator[str, None]:
    """
    Token-by-token streaming via LangGraph astream_events.

    Filters events so that only user-facing response nodes are streamed.
    For the search agent, waits until tool calls finish before streaming
    the final answer.
    """
    has_tool_calls_pending = False
    streamed_any = False
    final_state = None

    try:
        async for event in graph.astream_events(
            input=input_data, config=config, version="v2",
        ):
            event_type = event.get("event")
            event_data = event.get("data", {})
            tags = event.get("tags", [])

            # Capture final state for fallback extraction
            if event_type == "on_chain_end":
                output = event_data.get("output")
                if output and isinstance(output, dict) and "messages" in output:
                    final_state = output
                # Reset tool-pending flag when a tool finishes
                event_name = event.get("name", "")
                if "tool" in event_name.lower():
                    has_tool_calls_pending = False

            elif event_type == "on_chat_model_stream":
                # Use tags to determine the originating node
                tag_str = str(tags)
                if any(n in tag_str for n in _INTERNAL_NODES):
                    continue

                if not any(n in tag_str for n in _RESPONSE_NODES):
                    continue

                chunk = event_data.get("chunk")
                if not chunk or not isinstance(chunk, AIMessageChunk):
                    continue

                # Tool call chunks — mark pending, don't stream
                if chunk.tool_calls or chunk.tool_call_chunks:
                    has_tool_calls_pending = True
                    continue

                # Skip streaming while tool calls are pending (search agent mid-loop)
                if has_tool_calls_pending:
                    continue

                content = extract_text_content(chunk.content)
                if content and not _should_filter(content):
                    streamed_any = True
                    yield content

        if streamed_any:
            logger.info("Streaming complete")
        elif final_state:
            # Nothing streamed — extract from final state as fallback
            logger.warning("No content streamed, extracting from final state")
            final_response = _extract_final_response(final_state)
            if final_response:
                yield final_response

    except Exception as e:
        logger.error(f"Streaming error: {type(e).__name__}: {str(e)}, falling back to buffer mode")
        async for chunk in _stream_buffered(graph, input_data, config):
            yield chunk


async def _stream_buffered(
    graph,
    input_data: dict,
    config: dict,
) -> AsyncGenerator[str, None]:
    """Run the workflow to completion, then yield the full response."""
    logger.info("Running workflow in buffer mode")

    result = await graph.ainvoke(input=input_data, config=config)
    final_response = _extract_final_response(result)

    if not final_response:
        logger.warning("No valid final response found in workflow result")
        final_response = "I apologize, but I wasn't able to generate a response. Please try again."

    logger.info(f"Final response ready: intent={result.get('intent', 'unknown')}, "
                f"tool_calls={result.get('tool_call_count', 0)}")
    yield final_response


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _extract_final_response(state: dict) -> str:
    """Extract the last non-tool-call AI message from a workflow state."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = extract_text_content(msg.content)
            if content and not _should_filter(content):
                return _strip_thinking_tags(content)
    return ""


def _should_filter(content: str) -> bool:
    """Check if content is internal model output that shouldn't be shown."""
    if not content:
        return False
    lower = content.lower()
    return any(p.lower() in lower for p in _FILTERED_PATTERNS)


def _strip_thinking_tags(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from text."""
    return re.sub(r'<thinking>.*?</thinking>\s*', '', text, flags=re.DOTALL)


def _sanitize_actor_id(name: str) -> str:
    """Format a customer name into an AgentCore actor ID (e.g. 'user:john-doe')."""
    sanitized = re.sub(r'[^a-zA-Z0-9\-_ ]', '', name)
    sanitized = sanitized.replace(' ', '-').lower()
    return f"user:{sanitized or 'guest'}"


def _format_messages(
    messages: str | list[dict[str, Any]],
) -> list[HumanMessage | AIMessage]:
    """Convert string / list / dict messages into LangChain message objects."""
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]

    if isinstance(messages, list):
        if not messages:
            return []
        if isinstance(messages[0], dict) and "role" in messages[0]:
            result = []
            for msg in messages:
                if msg["role"] == "user":
                    result.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    result.append(AIMessage(content=msg["content"]))
            return result
        return [HumanMessage(content=m) for m in messages]

    return []
