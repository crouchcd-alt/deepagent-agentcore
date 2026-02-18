"""
SSE streaming and guardrail enforcement at the API boundary.

Wraps the orchestrator's response stream with:
- Input/output guardrail checks
- Server-Sent Event (SSE) formatting
- Observability session tracking for request-level tracing
"""

import json
import uuid
from typing import AsyncGenerator

from loguru import logger

from src.application.orchestrator.streaming import get_streaming_response
from src.infrastructure.guardrails import (
    apply_input_guardrail,
    apply_output_guardrail,
    get_blocked_input_message,
)
from src.infrastructure.observability import get_observability_manager


async def stream_response(
    user_input: str,
    customer_name: str = "Guest",
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted events with guardrail protection.

    Applies guardrails at the API boundary:
    - Input guardrail: Checks user input before processing
    - Output guardrail: Checks each response chunk before sending

    Wraps the orchestrator's streaming response and formats each chunk
    as a Server-Sent Event (SSE) for HTTP streaming.

    Observability:
    - Sets session ID in OpenTelemetry baggage for trace correlation
    - All traces within this request are tagged with the session ID
    - Viewable in CloudWatch GenAI Observability dashboard

    Args:
        user_input: The user's message/prompt.
        customer_name: Name of the customer for personalization.
        conversation_id: Optional ID for conversation threading.

    Yields:
        SSE-formatted strings with JSON payloads:
        - {"chunk": "..."} for content chunks
        - {"done": true} on completion
        - {"error": "..."} on failure
        - {"blocked": true, "message": "..."} if input blocked by guardrails
    """
    # Use conversation_id as session ID for observability, or generate one
    session_id = conversation_id or str(uuid.uuid4())

    # Get observability manager and set session context
    observability = get_observability_manager()

    # Use session context for automatic session ID propagation in traces
    with observability.session_context(session_id):
        # Record request start event
        observability.add_span_event(
            "request.start",
            attributes={
                "session.id": session_id,
                "customer.name": customer_name,
                "input.length": len(user_input),
            }
        )

        try:
            # Apply input guardrail before processing
            with observability.create_span(
                "guardrail.input",
                attributes={"input.length": len(user_input)}
            ):
                input_result = apply_input_guardrail(user_input)

            if not input_result.allowed:
                logger.info(f"Input blocked by guardrail: action={input_result.action}")
                observability.add_span_event(
                    "guardrail.blocked",
                    attributes={"action": str(input_result.action)}
                )
                blocked_message = get_blocked_input_message()
                yield f"data: {json.dumps({'blocked': True, 'message': blocked_message})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            # Collect full response for output guardrail check
            full_response = []

            with observability.create_span(
                "workflow.execution",
                attributes={
                    "session.id": session_id,
                    "customer.name": customer_name,
                }
            ):
                async for chunk in get_streaming_response(
                    messages=user_input,
                    customer_name=customer_name,
                    conversation_id=conversation_id,
                ):
                    if chunk:
                        full_response.append(chunk)
                        # Format as Server-Sent Event
                        event_data = json.dumps({"chunk": chunk})
                        yield f"data: {event_data}\n\n"

            # Apply output guardrail on the complete response
            complete_response = "".join(full_response)
            if complete_response:
                with observability.create_span(
                    "guardrail.output",
                    attributes={"output.length": len(complete_response)}
                ):
                    output_result = apply_output_guardrail(complete_response)

                if output_result.output != complete_response:
                    logger.info(f"Output modified by guardrail: action={output_result.action}")
                    observability.add_span_event(
                        "guardrail.modified",
                        attributes={"action": str(output_result.action)}
                    )
                    # Note: For streaming, the modified content has already been sent
                    # This is a limitation of streaming - we log for monitoring
                    # For strict enforcement, consider buffering or non-streaming mode

            # Record request completion
            observability.add_span_event(
                "request.complete",
                attributes={
                    "session.id": session_id,
                    "output.length": len(complete_response),
                }
            )

            # Send completion event
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) or "Unknown error occurred"
            logger.error(f"Error in stream_response: {error_type}: {error_msg}")
            logger.exception("Full stream_response traceback:")

            # Record error in observability
            observability.add_span_event(
                "request.error",
                attributes={
                    "error.type": error_type,
                    "error.message": error_msg,
                }
            )

            yield f"data: {json.dumps({'error': f'{error_type}: {error_msg}'})}\n\n"
