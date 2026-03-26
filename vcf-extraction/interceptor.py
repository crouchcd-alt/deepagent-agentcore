"""
AgentInterceptor: a LangChain callback handler that wires into the deepagent's
execution graph and provides two capabilities:

1. **Real-time Langfuse tracing** – opens a Langfuse child span at the start
   of every LLM invocation and tool call and closes it when the call ends.
   This means Langfuse receives events as they occur, not batch-uploaded only
   after the whole agent run returns.

2. **Inline limit enforcement** – checks AGENT_MAX_ITERATIONS at the very beginning of each LLM invocation
   (``on_llm_start`` / ``on_chat_model_start``).  When a limit is breached it
   raises ``RuntimeError`` directly inside the LangGraph
   event loop, propagating back to the pipeline's ``main()`` caller.

Usage
-----
    interceptor = AgentInterceptor(
        langfuse_parent=trace,
        max_iterations=AGENT_MAX_ITERATIONS,
        model_id=_BEDROCK_MODEL_ID,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"callbacks": [interceptor]},
    )

Because callbacks are wired into the agent via ``config`` rather than
constructor arguments, they propagate automatically to every sub-chain and
nested tool invocation inside the LangGraph state machine.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# ---------------------------------------------------------------------------
# Module logger with rotating file handler
# ---------------------------------------------------------------------------

_LOG_FILE = "agent_interceptor.log"
_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_LOG_BACKUP_COUNT = 5  # keep up to 5 rotated files

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE,
    maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT,
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))
logger.addHandler(_file_handler)


class AgentInterceptor(BaseCallbackHandler):
    """
    LangChain callback handler for deepagent observability and limit control.

    Parameters
    ----------
    langfuse_parent:
        A Langfuse trace or span object.  All child observations created by
        this interceptor are attached to this parent.
    max_iterations:
        Maximum number of LLM invocations (a reliable proxy for agent
        "reasoning steps") before the run is interrupted.
    model_id:
        Bedrock / model identifier string; used to annotate Langfuse
        generation spans so token costs are attributed correctly.
    """

    # Ensure LangChain propagates exceptions raised inside callbacks instead
    # of swallowing them.
    raise_error = True

    def __init__(
        self,
        langfuse_parent: Any,
        max_iterations: int,
        model_id: str,
    ) -> None:
        super().__init__()
        self._parent = langfuse_parent
        self._max_iterations = max_iterations
        self._model_id = model_id

        self._start_time: float = time.monotonic()
        self._llm_call_count: int = 0

        # Active Langfuse spans and their wall-clock start times, keyed by the
        # string representation of the LangChain run_id UUID.
        self._spans: dict[str, Any] = {}
        self._run_start: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_limits(self) -> None:
        """
        Raise immediately when a configured limit has been exceeded.

        This is called at the *start* of every LLM invocation so that limits
        are enforced as soon as the agent attempts a new reasoning step.

        Raises:
            RuntimeError: When the number of LLM calls reaches
                ``max_iterations``.
        """
        if self._llm_call_count >= self._max_iterations:
            logger.warning(
                "Max iterations reached: count=%d limit=%d",
                self._llm_call_count,
                self._max_iterations,
            )
            raise RuntimeError(f"Agent exceeded maximum iteration count of {self._max_iterations}.")

    def _elapsed(self) -> float:
        """Seconds elapsed since this interceptor was constructed."""
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called before a plain completion LLM is invoked.

        Checks limits first (raising if exceeded), then opens a Langfuse
        generation span that will be closed by ``on_llm_end``.
        """
        self._check_limits()
        self._llm_call_count += 1
        run_key = str(run_id)
        self._run_start[run_key] = time.monotonic()
        logger.info(
            "LLM call #%d starting (model=%s, run_id=%s, elapsed=%.3fs)",
            self._llm_call_count,
            self._model_id,
            run_key,
            self._elapsed(),
        )
        span = self._parent.start_observation(
            metadata={
                "llm_call": self._llm_call_count,
                "elapsed_seconds": round(self._elapsed(), 3),
                "run_id": run_key,
            },
        )
        self._spans[run_key] = span

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """
        Called before a chat model (e.g. ChatBedrock) is invoked.

        Chat models use ``on_chat_model_start`` rather than ``on_llm_start``;
        both are implemented so this interceptor works with any LLM backend.
        """
        self._check_limits()
        self._llm_call_count += 1
        run_key = str(run_id)
        self._run_start[run_key] = time.monotonic()
        logger.info(
            "Chat model call #%d starting (model=%s, run_id=%s, elapsed=%.3fs)",
            self._llm_call_count,
            self._model_id,
            run_key,
            self._elapsed(),
        )
        # Serialise the message list to a JSON-safe structure for Langfuse.
        serialised_msgs = [
            [{"role": getattr(m, "type", "unknown"), "content": _msg_content(m)} for m in turn]
            for turn in messages
        ]
        span = self._parent.start_observation(
            name=f"llm_call_{self._llm_call_count}",
            as_type="generation",
            model=self._model_id,
            input={"messages": serialised_msgs},
            metadata={
                "llm_call": self._llm_call_count,
                "elapsed_seconds": round(self._elapsed(), 3),
                "run_id": run_key,
            },
        )
        self._spans[run_key] = span

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Close the active Langfuse generation span with token usage and output."""
        run_key = str(run_id)
        span = self._spans.pop(run_key, None)
        if span is None:
            return
        call_elapsed = time.monotonic() - self._run_start.pop(run_key, time.monotonic())
        usage = _parse_usage(response)
        logger.info(
            "LLM call completed (run_id=%s, elapsed=%.3fs, usage=%s)",
            run_key,
            call_elapsed,
            usage or "n/a",
        )
        output_texts = [gen.text for gens in response.generations for gen in gens]
        span.update(
            output={"texts": output_texts},
            usage_details=usage or None,
            metadata={
                "elapsed_seconds": round(call_elapsed, 3),
                "total_elapsed_seconds": round(self._elapsed(), 3),
            },
        )
        span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Close the active LLM span with an error status."""
        run_key = str(run_id)
        span = self._spans.pop(run_key, None)
        if span:
            logger.error("LLM call failed (run_id=%s): %s", run_key, error)
            span.update(level="ERROR", status_message=str(error))
            span.end()
        self._run_start.pop(run_key, None)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Open a Langfuse tool span when a tool execution begins."""
        run_key = str(run_id)
        self._run_start[run_key] = time.monotonic()
        tool_name = serialized.get("name", "unknown_tool")
        logger.info(
            "Tool '%s' starting (run_id=%s, elapsed=%.3fs)",
            tool_name,
            run_key,
            self._elapsed(),
        )
        span = self._parent.start_observation(
            name=f"tool.{tool_name}",
            as_type="tool",
            input={"input": input_str, "tool": tool_name},
            metadata={
                "elapsed_seconds": round(self._elapsed(), 3),
                "run_id": run_key,
            },
        )
        self._spans[run_key] = span

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Close the active tool span with the tool's output."""
        run_key = str(run_id)
        span = self._spans.pop(run_key, None)
        if span is None:
            return
        call_elapsed = time.monotonic() - self._run_start.pop(run_key, time.monotonic())
        logger.info(
            "Tool completed (run_id=%s, elapsed=%.3fs)",
            run_key,
            call_elapsed,
        )
        span.update(
            output={"output": str(output)},
            metadata={
                "elapsed_seconds": round(call_elapsed, 3),
                "total_elapsed_seconds": round(self._elapsed(), 3),
            },
        )
        span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Close the active tool span with an error status."""
        run_key = str(run_id)
        span = self._spans.pop(run_key, None)
        if span:
            logger.error("Tool failed (run_id=%s): %s", run_key, error)
            span.update(level="ERROR", status_message=str(error))
            span.end()
        self._run_start.pop(run_key, None)

    # ------------------------------------------------------------------
    # Properties exposed to the pipeline after the run
    # ------------------------------------------------------------------

    @property
    def llm_call_count(self) -> int:
        """Total number of LLM invocations recorded by this interceptor."""
        return self._llm_call_count

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since this interceptor was constructed."""
        return self._elapsed()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _msg_content(msg: BaseMessage) -> str | list[Any]:
    """Return message content in a JSON-serialisable form."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [
            block
            if isinstance(block, str)
            else dict(block)
            if hasattr(block, "items")
            else str(block)
            for block in content
        ]
    return str(content)


def _parse_usage(response: LLMResult) -> dict[str, int]:
    """
    Extract token counts from an LLMResult in a backend-agnostic way.

    Tries, in order:
    1. ``response.llm_output["token_usage"]``  (OpenAI / standard LangChain)
    2. ``response.llm_output["usage"]``        (Bedrock via langchain-aws)
    3. ``response.generations[0][0].generation_info["usage"]``

    Returns a dict with ``"input"``, ``"output"``, and ``"total"`` keys
    suitable for Langfuse's ``usage_details``, or ``{}`` if no usage data was
    found.
    """
    if response.llm_output:
        for key in ("token_usage", "usage"):
            raw = response.llm_output.get(key)
            if isinstance(raw, dict) and raw:
                return {
                    "input": raw.get("prompt_tokens") or raw.get("inputTokens", 0),
                    "output": raw.get("completion_tokens") or raw.get("outputTokens", 0),
                    "total": raw.get("total_tokens") or raw.get("totalTokens", 0),
                }
    # Fall back to the first generation's metadata.
    try:
        gen_info = response.generations[0][0].generation_info or {}
        raw = gen_info.get("usage", {})
        if isinstance(raw, dict) and raw:
            return {
                "input": raw.get("inputTokens", 0),
                "output": raw.get("outputTokens", 0),
                "total": raw.get("totalTokens", 0),
            }
    except (IndexError, AttributeError):
        pass
    return {}
