"""
VCF Extraction Pipeline via Deepagent + AWS AgentCore.

Workflow
--------
1. Load environment configuration from .env.
2. Initialise Langfuse for real-time trace streaming.
3. Provision an AWS AgentCore code-interpreter sandbox session.
4. Seed the sandbox with schema.py and variants.vcf.
5. Verify that pydantic is importable inside the remote runtime.
6. Construct a DeepAgent (Claude 3.5 Sonnet) bound to the AgentCore
   code interpreter as its primary tool.
7. Run the agent with interrupt logic (wall-clock timeout + step cap).
8. Stream every reasoning step and tool-call to Langfuse in real-time.
9. Capture the final JSON output, validate it locally, and persist it.
10. Tear down the sandbox session and flush Langfuse traces.

Usage
-----
    python pipeline.py

Environment variables are read from a .env file in the same directory.
See .env.example for the full list of required variables.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the directory that contains schema.py is importable before the
# module-level import below runs.
sys.path.insert(0, str(Path(__file__).parent))

import boto3
from dotenv import load_dotenv
from langfuse import Langfuse
from pydantic import ValidationError

from schema import VariantExtractionResult

# ---------------------------------------------------------------------------
# Bootstrap: load environment variables
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY: str = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY: str = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

AGENT_TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

OUTPUT_JSON_PATH: Path = Path(os.getenv("OUTPUT_JSON_PATH", "vcf_extraction_results.json"))

# Paths to local assets that are uploaded to the sandbox
_SCRIPT_DIR = Path(__file__).parent
_SCHEMA_PATH = _SCRIPT_DIR / "schema.py"
_VCF_PATH = _SCRIPT_DIR / "variants.vcf"

# The code-interpreter identifier used by all AWS AgentCore regions
_CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

# Sandbox destination directory
_SANDBOX_DATA_DIR = "/mnt/data"

# ---------------------------------------------------------------------------
# Langfuse initialisation
# ---------------------------------------------------------------------------


def _init_langfuse() -> Langfuse:
    """Initialise and return a configured Langfuse client."""
    client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    return client


# ---------------------------------------------------------------------------
# AgentCore sandbox helpers
# ---------------------------------------------------------------------------


def _create_agentcore_client() -> Any:
    """Return a boto3 bedrock-agentcore client."""
    return boto3.client("bedrock-agentcore", region_name=AWS_REGION)


def _start_sandbox(client: Any) -> str:
    """
    Start a new AgentCore code-interpreter session.

    Returns:
        The session ID string for subsequent API calls.
    """
    response = client.start_code_interpreter_session(
        codeInterpreterIdentifier=_CODE_INTERPRETER_ID
    )
    session_id: str = response["sessionId"]
    return session_id


def _run_code(client: Any, session_id: str, code: str) -> str:
    """
    Execute *code* inside the sandbox and return stdout as a string.

    Args:
        client:     The bedrock-agentcore boto3 client.
        session_id: The active sandbox session ID.
        code:       Python source code to execute.

    Returns:
        Combined stdout / result text from the sandbox.
    """
    response = client.invoke_code_interpreter(
        codeInterpreterIdentifier=_CODE_INTERPRETER_ID,
        sessionId=session_id,
        name="executeCode",
        arguments={"language": "python", "code": code},
    )
    # The response body is a streaming blob; read it fully.
    output_parts: list[str] = []
    body = response.get("body") or response.get("output") or ""
    if hasattr(body, "read"):
        raw = body.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        output_parts.append(raw)
    elif isinstance(body, (list, tuple)):
        for part in body:
            if isinstance(part, dict):
                output_parts.append(part.get("text", str(part)))
            else:
                output_parts.append(str(part))
    elif isinstance(body, str):
        output_parts.append(body)
    return "\n".join(output_parts)


def _upload_file(client: Any, session_id: str, local_path: Path, remote_path: str) -> None:
    """
    Write *local_path* contents into the sandbox at *remote_path*.

    The file contents are base64-encoded before being embedded in the
    generated code string so that no special-character escaping is needed
    regardless of what the source file contains.
    """
    raw_bytes = local_path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    upload_code = f"""\
import base64, pathlib
_dest = pathlib.Path("{remote_path}")
_dest.parent.mkdir(parents=True, exist_ok=True)
_dest.write_bytes(base64.b64decode("{b64}"))
print(f"Uploaded {remote_path} ({{_dest.stat().st_size}} bytes)")
"""
    _run_code(client, session_id, upload_code)


def _verify_pydantic(client: Any, session_id: str) -> str:
    """Return the pydantic version string available in the sandbox."""
    return _run_code(
        client,
        session_id,
        "import pydantic; print(f'pydantic {pydantic.__version__} OK')",
    )


def _stop_sandbox(client: Any, session_id: str) -> None:
    """Terminate the AgentCore sandbox session."""
    client.stop_code_interpreter_session(
        codeInterpreterIdentifier=_CODE_INTERPRETER_ID,
        sessionId=session_id,
    )


# ---------------------------------------------------------------------------
# AgentCore tool wrapper (callable by the DeepAgent)
# ---------------------------------------------------------------------------


def _make_agentcore_tool(client: Any, session_id: str):  # noqa: ANN201
    """
    Return a plain Python callable that the DeepAgent can invoke as a tool.

    The callable accepts a ``code`` keyword argument and returns the sandbox
    stdout as a string.
    """

    def execute_code(code: str) -> str:
        """
        Execute Python code in the AgentCore sandbox and return stdout.

        Args:
            code: Python source code to run inside the remote sandbox.

        Returns:
            The standard output produced by the code execution.
        """
        return _run_code(client, session_id, code)

    return execute_code


# ---------------------------------------------------------------------------
# DeepAgent construction and execution
# ---------------------------------------------------------------------------


def _build_agent(agentcore_tool):  # noqa: ANN001, ANN201
    """
    Build and return a DeepAgent bound to the AgentCore code interpreter.

    Uses Claude 3.5 Sonnet as the underlying LLM.
    """
    from deepagents import create_deep_agent  # type: ignore[import]

    agent = create_deep_agent(
        model="claude-3-5-sonnet-20241022",
        tools=[agentcore_tool],
        system_prompt=(
            "You are a bioinformatics assistant specialising in VCF file processing. "
            "You have access to a remote Python sandbox via the `execute_code` tool. "
            "Use it to read, parse, and validate variant data."
        ),
    )
    return agent


_EXTRACTION_PROMPT = (
    "Extract variant data from /mnt/data/variants.vcf using the Pydantic model "
    "defined in /mnt/data/schema.py. "
    "Validate all fields, correct formatting errors where possible (for example, "
    "coerce strings to floats for allele frequency), and return a JSON array of "
    "validated records under the key 'records', plus a 'total' count and a "
    "'validation_errors' list."
)


def _run_agent_with_limits(
    agent,  # noqa: ANN001
    langfuse_trace,  # noqa: ANN001
) -> str:
    """
    Run *agent* with wall-clock timeout and step-count limits.

    Each agent step is sent to Langfuse as a child span so that the full
    reasoning chain is captured in real-time.

    Args:
        agent:          The DeepAgent instance.
        langfuse_trace: An active Langfuse trace object.

    Returns:
        The agent's final text output.

    Raises:
        TimeoutError: If the wall-clock limit is exceeded.
        RuntimeError: If the agent exceeds the maximum iteration count.
    """
    start_time = time.monotonic()
    step_count = 0
    last_output = ""

    # ------------------------------------------------------------------
    # Graceful SIGALRM-based timeout (Unix only).  On Windows the signal
    # is silently skipped and only the monotonic-clock check applies.
    # ------------------------------------------------------------------
    _timeout_fired = False

    def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        nonlocal _timeout_fired
        _timeout_fired = True

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(AGENT_TIMEOUT_SECONDS)

    try:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": _EXTRACTION_PROMPT}
        ]

        while True:
            # --- Check limits ---
            elapsed = time.monotonic() - start_time
            if _timeout_fired or elapsed > AGENT_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"Agent exceeded wall-clock limit of {AGENT_TIMEOUT_SECONDS}s "
                    f"(elapsed: {elapsed:.1f}s)."
                )
            if step_count >= AGENT_MAX_ITERATIONS:
                raise RuntimeError(
                    f"Agent exceeded maximum iteration count of {AGENT_MAX_ITERATIONS}."
                )

            step_count += 1
            span = langfuse_trace.span(
                name=f"agent_step_{step_count}",
                input={"messages": messages},
                metadata={"step": step_count, "elapsed_seconds": elapsed},
            )

            # --- Execute one step ---
            # deepagents exposes .step() for single-turn execution;
            # fall back to .invoke() for libraries that only expose invoke.
            if hasattr(agent, "step"):
                result = agent.step({"messages": messages})
            else:
                result = agent.invoke({"messages": messages})

            # Extract the assistant reply
            assistant_msg = _extract_assistant_message(result)
            last_output = assistant_msg

            span.end(
                output={"assistant": assistant_msg},
                metadata={"step": step_count, "elapsed_seconds": time.monotonic() - start_time},
            )

            # Append the assistant reply to the message history
            messages.append({"role": "assistant", "content": assistant_msg})

            # --- Termination check ---
            # The agent signals it is done when no further tool calls are
            # pending.  deepagents returns a graph state dict; if
            # 'next' is empty the run is complete.
            if _is_done(result):
                break

    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # Cancel any pending alarm

    return last_output


def _extract_assistant_message(result: Any) -> str:
    """Extract the human-readable assistant reply from a deepagents result."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # LangGraph state dict: messages list
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                content = last.content
            elif isinstance(last, dict):
                content = last.get("content", "")
            else:
                content = str(last)
            if isinstance(content, list):
                parts = [
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in content
                ]
                return "".join(parts)
            return str(content)
    return str(result)


def _is_done(result: Any) -> bool:
    """
    Return True when the agent graph has reached a terminal state.

    deepagents / LangGraph signals completion by returning a state dict
    with an empty or absent 'next' key.
    """
    if isinstance(result, dict):
        next_steps = result.get("next", [])
        return not next_steps
    return True  # For plain string returns, treat as terminal


# ---------------------------------------------------------------------------
# JSON extraction and local validation
# ---------------------------------------------------------------------------


def _extract_json_from_output(output: str) -> dict[str, Any]:
    """
    Parse the agent's final output as JSON.

    The agent may wrap the JSON in a markdown code block; strip that first.
    """
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (e.g. ```json) and closing fence
        inner_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner_lines.append(line)
        text = "\n".join(inner_lines)

    # Try to find the first JSON object in the output
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start != -1:
        text = text[start:]

    return json.loads(text)


def _validate_results_locally(data: dict[str, Any]) -> dict[str, Any]:
    """
    Re-validate the agent's JSON output against the local Pydantic schema.

    This provides a second layer of validation independent of the sandbox
    execution, ensuring the data is correct before it is persisted.
    """
    # Normalise: the agent may return a bare list under 'records' or a
    # full VariantExtractionResult-shaped dict.
    if isinstance(data, list):
        data = {"records": data, "total": len(data)}

    try:
        validated = VariantExtractionResult(**data)
        return validated.model_dump()
    except ValidationError as exc:
        # Return the raw data with a warning; do not discard partial results
        data.setdefault("validation_errors", [])
        data["validation_errors"].append(f"Local re-validation error: {exc}")
        return data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the full VCF extraction pipeline."""
    # --- Step 1: Initialise Langfuse ---
    print("Initialising Langfuse...")
    lf = _init_langfuse()
    trace = lf.trace(
        name="vcf-extraction-pipeline",
        metadata={
            "timeout_seconds": AGENT_TIMEOUT_SECONDS,
            "max_iterations": AGENT_MAX_ITERATIONS,
            "vcf_path": str(_VCF_PATH),
        },
    )

    agentcore_client = None
    session_id: str | None = None
    success = False
    final_output: dict[str, Any] = {}

    try:
        # --- Step 2: Provision sandbox ---
        print("Starting AgentCore code-interpreter session...")
        agentcore_client = _create_agentcore_client()
        session_id = _start_sandbox(agentcore_client)
        print(f"Sandbox session started: {session_id}")
        trace.event(name="sandbox_started", metadata={"session_id": session_id})

        # --- Step 3: Seed sandbox ---
        print("Uploading schema.py and variants.vcf to sandbox...")
        _upload_file(agentcore_client, session_id, _SCHEMA_PATH, f"{_SANDBOX_DATA_DIR}/schema.py")
        _upload_file(agentcore_client, session_id, _VCF_PATH, f"{_SANDBOX_DATA_DIR}/variants.vcf")
        trace.event(name="files_uploaded", metadata={"files": ["schema.py", "variants.vcf"]})

        # --- Step 4: Verify pydantic ---
        pydantic_check = _verify_pydantic(agentcore_client, session_id)
        print(f"Pydantic check: {pydantic_check.strip()}")
        trace.event(name="pydantic_verified", metadata={"output": pydantic_check})

        # --- Step 5: Build agent ---
        print("Building DeepAgent with AgentCore code interpreter...")
        execute_code_tool = _make_agentcore_tool(agentcore_client, session_id)
        agent = _build_agent(execute_code_tool)
        trace.event(name="agent_built", metadata={"model": "claude-3-5-sonnet-20241022"})

        # --- Step 6: Run agent ---
        print(
            f"Running agent (timeout={AGENT_TIMEOUT_SECONDS}s, "
            f"max_iterations={AGENT_MAX_ITERATIONS})..."
        )
        raw_output = _run_agent_with_limits(agent, trace)
        print("Agent run complete.")
        trace.event(name="agent_complete", metadata={"output_preview": raw_output[:500]})

        # --- Step 7: Parse and validate output ---
        print("Parsing and validating agent output...")
        try:
            raw_data = _extract_json_from_output(raw_output)
        except json.JSONDecodeError as exc:
            print(f"WARNING: Could not parse agent output as JSON: {exc}")
            raw_data = {"records": [], "total": 0, "validation_errors": [str(exc)]}

        final_output = _validate_results_locally(raw_data)
        record_count = len(final_output.get("records", []))
        print(f"Extracted and validated {record_count} variant record(s).")
        trace.event(
            name="validation_complete",
            metadata={
                "record_count": record_count,
                "validation_errors": final_output.get("validation_errors", []),
            },
        )

        success = True

    except TimeoutError as exc:
        msg = str(exc)
        print(f"ERROR (timeout): {msg}", file=sys.stderr)
        trace.event(name="timeout_error", metadata={"error": msg})

    except RuntimeError as exc:
        msg = str(exc)
        print(f"ERROR (iteration limit): {msg}", file=sys.stderr)
        trace.event(name="iteration_limit_error", metadata={"error": msg})

    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"ERROR: {msg}", file=sys.stderr)
        trace.event(name="pipeline_error", metadata={"error": msg})

    finally:
        # --- Step 8: Persist results ---
        if final_output:
            OUTPUT_JSON_PATH.write_text(
                json.dumps(final_output, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Results saved to {OUTPUT_JSON_PATH}")
            trace.event(name="results_saved", metadata={"path": str(OUTPUT_JSON_PATH)})

        # --- Step 9: Terminate sandbox ---
        if agentcore_client is not None and session_id is not None:
            print("Terminating sandbox session...")
            try:
                _stop_sandbox(agentcore_client, session_id)
                print("Sandbox session terminated.")
                trace.event(name="sandbox_stopped", metadata={"session_id": session_id})
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: Failed to stop sandbox session: {exc}", file=sys.stderr)

        # --- Step 10: Flush Langfuse ---
        trace.update(
            output={"success": success, "record_count": len(final_output.get("records", []))},
            metadata={"success": success},
        )
        print("Flushing Langfuse traces...")
        lf.flush()
        print("Pipeline complete." if success else "Pipeline finished with errors.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
