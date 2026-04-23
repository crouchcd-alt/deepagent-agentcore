"""OTEL tracing and structured logging for the LangGraph agent."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry import trace

TRACER_NAME = "agent"

tracer = trace.get_tracer(TRACER_NAME)
log = logging.getLogger(TRACER_NAME)


# ---------------------------------------------------------------------------
# 1. Utility: custom trace context manager
# ---------------------------------------------------------------------------


@contextmanager
def agent_span(name: str, attributes: dict[str, Any] | None = None):
    """Create a custom OTEL span nested under the current trace.

    Usage::

        with agent_span("my-step", {"key": "value"}) as span:
            ...
            span.set_attribute("extra", 42)
    """
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span


# ---------------------------------------------------------------------------
# 2. LangChain callback handler — logging + span enrichment
# ---------------------------------------------------------------------------


def _message_summary(msg: BaseMessage) -> dict[str, Any]:
    """Return a compact dict summarizing a single message."""
    summary: dict[str, Any] = {"role": msg.type}
    content = msg.content
    if isinstance(content, str):
        summary["content"] = content[:300]
    elif isinstance(content, list):
        summary["content"] = str(content)[:300]

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        summary["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls
        ]
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        summary["tool_call_id"] = msg.tool_call_id
    return summary


class OtelCallbackHandler(BaseCallbackHandler):
    """Enriches OTEL traces with ordered chat history and logs tool calls."""

    # ---- chat model ----

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = trace.get_current_span()
        for batch_idx, batch in enumerate(messages):
            for msg_idx, msg in enumerate(batch):
                prefix = f"chat.messages.{batch_idx}.{msg_idx}"
                summary = _message_summary(msg)
                span.set_attribute(f"{prefix}.role", summary["role"])
                if "content" in summary:
                    span.set_attribute(f"{prefix}.content", summary["content"])
                if "tool_calls" in summary:
                    span.set_attribute(
                        f"{prefix}.tool_calls", json.dumps(summary["tool_calls"])
                    )
                if "tool_call_id" in summary:
                    span.set_attribute(
                        f"{prefix}.tool_call_id", summary["tool_call_id"]
                    )
            span.set_attribute(f"chat.messages.{batch_idx}.count", len(batch))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = trace.get_current_span()
        for gen_idx, generations in enumerate(response.generations):
            for choice_idx, gen in enumerate(generations):
                prefix = f"llm.response.{gen_idx}.{choice_idx}"
                span.set_attribute(prefix, str(gen.text)[:500])

    # ---- tools ----

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        log.info("Tool call: %s | input: %s", tool_name, input_str)

        span = trace.get_current_span()
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.input", input_str[:500])
        span.add_event("tool_start", {"tool.name": tool_name, "tool.input": input_str[:500]})

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        output_str = str(output)
        log.info("Tool result: %s", output_str[:500])

        span = trace.get_current_span()
        span.set_attribute("tool.output", output_str[:500])
        span.add_event("tool_end", {"tool.output": output_str[:500]})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        log.error("Tool error: %s", error)

        span = trace.get_current_span()
        span.set_status(trace.StatusCode.ERROR, str(error))
        span.record_exception(error)
