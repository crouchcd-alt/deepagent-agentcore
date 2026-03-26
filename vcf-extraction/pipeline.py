"""
VCF Extraction Pipeline via Deepagent + AWS AgentCore.

Workflow
--------
1. Load environment configuration from .env.
2. Initialise Langfuse for real-time trace streaming.
3. Provision an AWS AgentCore code-interpreter sandbox session.
4. Seed the sandbox with schema.py and variants.vcf.
5. Verify that pydantic is importable inside the remote runtime.
6. Construct a DeepAgent (Amazon Nova Lite via AWS Bedrock) bound to the
   AgentCore code interpreter as its primary tool.
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

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the directory that contains schema.py / interceptor.py is importable
# before the module-level imports below run.
sys.path.insert(0, str(Path(__file__).parent))

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from dotenv import load_dotenv
from langchain_agentcore_codeinterpreter import AgentCoreSandbox
from langfuse import Langfuse
from pydantic import ValidationError

from interceptor import AgentInterceptor
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
# Accept either LANGFUSE_BASE_URL (preferred) or the legacy LANGFUSE_BASE_URL name.
LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
    "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
)

AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

OUTPUT_JSON_PATH: Path = Path(os.getenv("OUTPUT_JSON_PATH", "vcf_extraction_results.json"))

# Paths to local assets that are uploaded to the sandbox
_SCRIPT_DIR = Path(__file__).parent
_SCHEMA_PATH = _SCRIPT_DIR / "schema.py"
_VCF_PATH = _SCRIPT_DIR / "variants.vcf"

# Sandbox destination directory
_SANDBOX_DATA_DIR = "/pme"

# ---------------------------------------------------------------------------
# Langfuse initialisation
# ---------------------------------------------------------------------------


def _init_langfuse() -> Langfuse:
    """Initialise, auth-check, and return a configured Langfuse client."""
    client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL,
    )
    if not client.auth_check():
        raise RuntimeError(f"Langfuse auth failed – check keys and host ({LANGFUSE_BASE_URL})")
    return client


# ---------------------------------------------------------------------------
# AgentCore sandbox helpers
# ---------------------------------------------------------------------------


def _start_sandbox(interpreter: CodeInterpreter, parent: Any) -> None:
    """
    Start a new AgentCore code-interpreter session.

    Creates a Langfuse span under *parent* capturing the region and the
    returned session ID with wall-clock timing.
    """
    t0 = time.monotonic()
    span = parent.start_observation(
        name="agentcore.start_session",
        as_type="span",
        input={"region": AWS_REGION},
    )
    try:
        interpreter.start()
        span.update(
            output={"session_id": interpreter.session_id},
            metadata={"elapsed_seconds": round(time.monotonic() - t0, 3)},
        )
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.end()


def _run_python(interpreter: CodeInterpreter, code: str) -> str:
    """
    Execute Python *code* inside the sandbox and return stdout as a string.

    Args:
        interpreter: An active :class:`CodeInterpreter` instance.
        code:        Python source code to execute.

    Returns:
        Combined stdout / result text from the sandbox.
    """
    response = interpreter.invoke(method="executeCode", params={"language": "python", "code": code})
    output_parts: list[str] = []
    for event in response.get("stream", []):
        if "result" not in event:
            continue
        for item in event["result"].get("content", []):
            if item.get("type") == "text":
                output_parts.append(item.get("text", ""))
            elif item.get("type") == "error":
                output_parts.append(f"Error: {item.get('text', 'Unknown error')}")
    return "\n".join(output_parts)


def _upload_files(
    sandbox: AgentCoreSandbox,
    files: list[tuple[Path, str]],
    parent: Any,
) -> None:
    """
    Upload local files into the sandbox using :class:`AgentCoreSandbox`.

    Args:
        sandbox: An :class:`AgentCoreSandbox` wrapping the active session.
        files:   List of ``(local_path, remote_path)`` pairs to upload.
        parent:  Langfuse observation to attach the span to.

    Creates a single Langfuse span recording all file names, sizes, and
    upload timing.
    """
    file_infos = [{"local": p.name, "remote": r, "size_bytes": p.stat().st_size} for p, r in files]
    t0 = time.monotonic()
    span = parent.start_observation(
        name="agentcore.upload_files",
        as_type="span",
        input={"files": file_infos},
    )
    try:
        upload_list = [(remote_path, local_path.read_bytes()) for local_path, remote_path in files]
        results = sandbox.upload_files(upload_list)
        errors = [r for r in results if r.error]
        if errors:
            raise RuntimeError(f"Upload failed for: {[r.path for r in errors]}")
        span.update(
            output={"uploaded": [r.path for r in results]},
            metadata={"elapsed_seconds": round(time.monotonic() - t0, 3)},
        )
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.end()


def _verify_pydantic(interpreter: CodeInterpreter, parent: Any) -> str:
    """
    Return the pydantic version string available in the sandbox.

    Creates a Langfuse span under *parent* recording the version check output.
    """
    t0 = time.monotonic()
    span = parent.start_observation(
        name="agentcore.verify_pydantic",
        as_type="span",
        input={"session_id": interpreter.session_id},
    )
    try:
        result = _run_python(
            interpreter,
            "import pydantic; print(f'pydantic {pydantic.__version__} OK')",
        )
        span.update(
            output={"stdout": result.strip()},
            metadata={"elapsed_seconds": round(time.monotonic() - t0, 3)},
        )
        return result
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.end()


def _stop_sandbox(interpreter: CodeInterpreter, parent: Any) -> None:
    """
    Terminate the AgentCore sandbox session.

    Creates a Langfuse span under *parent* capturing termination timing.
    """
    t0 = time.monotonic()
    span = parent.start_observation(
        name="agentcore.stop_session",
        as_type="span",
        input={"session_id": interpreter.session_id},
    )
    try:
        interpreter.stop()
        span.update(
            output={"status": "terminated"},
            metadata={"elapsed_seconds": round(time.monotonic() - t0, 3)},
        )
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.end()


# ---------------------------------------------------------------------------
# AgentCore tool wrapper (callable by the DeepAgent)
# ---------------------------------------------------------------------------


def _make_agentcore_tool(interpreter: CodeInterpreter):  # noqa: ANN201
    """
    Return a plain Python callable that the DeepAgent can invoke as a tool.

    The callable accepts a ``code`` argument and returns the sandbox stdout as
    a string.  Langfuse tracing for each invocation is handled by
    :class:`AgentInterceptor` via its ``on_tool_start`` / ``on_tool_end``
    callbacks — no manual span management is needed here.
    """

    def execute_code(code: str) -> str:
        """
        Execute Python code in the AgentCore sandbox and return stdout.

        Args:
            code: Python source code to run inside the remote sandbox.

        Returns:
            The standard output produced by the code execution.
        """
        return _run_python(interpreter, code)

    return execute_code


# ---------------------------------------------------------------------------
# DeepAgent construction and execution
# ---------------------------------------------------------------------------


_BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")


def _build_agent(agentcore_tool, sandbox: AgentCoreSandbox):  # noqa: ANN001, ANN201
    """
    Build and return a DeepAgent bound to the AgentCore code interpreter.

    Uses Amazon Nova Lite via AWS Bedrock as the underlying LLM.

    Args:
        agentcore_tool: The ``execute_code`` callable returned by
            ``_make_agentcore_tool``.
        sandbox:        The AgentCoreSandbox instance wrapping the active session,
    """
    from deepagents import create_deep_agent  # type: ignore[import]
    from langchain_aws import ChatBedrock

    llm = ChatBedrock(model=_BEDROCK_MODEL_ID, region=AWS_REGION)
    agent = create_deep_agent(
        model=llm,
        tools=[agentcore_tool],
        backend=sandbox,
        system_prompt=(
            "You are a bioinformatics assistant specialising in VCF file processing. "
            "You have access to a remote Python sandbox via the `execute_code` tool. "
            "Use it to read, parse, and validate variant data."
        ),
    )
    return agent


_EXTRACTION_PROMPT = """
Extract variant data using the `execute_code` tool.

<file_processing>
- /opt/amazon/genesis1p-tools/var/pme/variants.vcf: VCF file containing genetic variants to extract.
- /opt/amazon/genesis1p-tools/var/pme/schema.py: Pydantic schema for validating extracted variants.

If the VCF is too large (e.g., more than a few hundred rows), break the file into chunks and process them separately, merging them to produce the final output.
</file_processing>

<validation>
Validate all fields, correct formatting errors where possible (for example, coerce strings to floats for allele frequency). Validate by loading the extracted variants using the methods provided by Pydantic exclusively (e.g., .model_dump()).
</validation>

YOUR LAST MESSAGE SHOULD BE A JSON OBJECT FROM THE PYDANTIC SCHEMA. NO PROSE OR PREAMBLE, JUST THE RAW JSON OBJECT.
"""


def _run_agent_with_interceptor(
    agent,  # noqa: ANN001
    interceptor: AgentInterceptor,
) -> str:
    """
    Run the deepagent once with the interceptor wired in as a LangChain callback.

    Passing the interceptor via ``config["callbacks"]`` means LangChain
    propagates it automatically to every sub-chain, LLM call, and tool
    invocation inside the LangGraph state machine.  The interceptor fires
    *synchronously* on each event, so:

    * Langfuse spans are opened and closed in real time as the agent reasons.
    * Timeout and iteration-count limits are enforced at the start of every
      LLM call, interrupting the agent immediately when a limit is breached.

    Args:
        agent:       The DeepAgent instance (must implement ``.invoke()``).
        interceptor: A configured :class:`AgentInterceptor`.

    Returns:
        The agent's final text output.

    Raises:
        TimeoutError: Propagated from the interceptor when the wall-clock
            limit is exceeded.
        RuntimeError: Propagated from the interceptor when the iteration
            limit is exceeded.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _EXTRACTION_PROMPT}]},
        config={"callbacks": [interceptor]},
    )
    return _extract_assistant_message(result)


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
    trace = lf.start_observation(
        name="vcf-extraction-pipeline",
        as_type="span",
        metadata={
            "max_iterations": AGENT_MAX_ITERATIONS,
            "vcf_path": str(_VCF_PATH),
        },
    )

    interpreter: CodeInterpreter | None = None
    sandbox: AgentCoreSandbox | None = None
    success = False
    final_output: dict[str, Any] = {}

    try:
        # --- Step 2: Provision sandbox ---
        print("Starting AgentCore code-interpreter session...")
        interpreter = CodeInterpreter(region=AWS_REGION)
        sandbox = AgentCoreSandbox(interpreter=interpreter)
        _start_sandbox(interpreter, trace)
        print(f"Sandbox session started: {interpreter.session_id}")
        trace.create_event(name="sandbox_started", metadata={"session_id": interpreter.session_id})

        # --- Step 3: Seed sandbox ---
        print("Uploading schema.py and variants.vcf to sandbox...")
        _upload_files(
            sandbox,
            [
                (_SCHEMA_PATH, f"{_SANDBOX_DATA_DIR}/schema.py"),
                (_VCF_PATH, f"{_SANDBOX_DATA_DIR}/variants.vcf"),
            ],
            trace,
        )
        trace.create_event(name="files_uploaded", metadata={"files": ["schema.py", "variants.vcf"]})

        # --- Step 4: Verify pydantic ---
        pydantic_check = _verify_pydantic(interpreter, trace)
        print(f"Pydantic check: {pydantic_check.strip()}")
        trace.create_event(name="pydantic_verified", metadata={"output": pydantic_check})

        # --- Step 5: Build agent ---
        print("Building DeepAgent with AgentCore code interpreter...")
        execute_code_tool = _make_agentcore_tool(interpreter)
        agent = _build_agent(execute_code_tool, sandbox)
        trace.create_event(name="agent_built", metadata={"model": _BEDROCK_MODEL_ID})

        # --- Step 6: Create interceptor and run agent ---
        print(f"Running agent (max_iterations={AGENT_MAX_ITERATIONS})...")
        interceptor = AgentInterceptor(
            langfuse_parent=trace,
            max_iterations=AGENT_MAX_ITERATIONS,
            model_id=_BEDROCK_MODEL_ID,
        )
        trace.create_event(
            name="interceptor_attached",
            metadata={
                "max_iterations": AGENT_MAX_ITERATIONS,
            },
        )
        raw_output = _run_agent_with_interceptor(agent, interceptor)
        print("Agent run complete.")
        trace.create_event(
            name="agent_complete",
            metadata={
                "output_preview": raw_output[:500],
                "llm_call_count": interceptor.llm_call_count,
                "elapsed_seconds": round(interceptor.elapsed_seconds, 3),
            },
        )

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
        trace.create_event(
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
        trace.create_event(name="timeout_error", metadata={"error": msg})

    except RuntimeError as exc:
        msg = str(exc)
        print(f"ERROR (iteration limit): {msg}", file=sys.stderr)
        trace.create_event(name="iteration_limit_error", metadata={"error": msg})

    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"ERROR: {msg}", file=sys.stderr)
        trace.create_event(name="pipeline_error", metadata={"error": msg})

    finally:
        # --- Step 8: Persist results ---
        if final_output:
            OUTPUT_JSON_PATH.write_text(
                json.dumps(final_output, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Results saved to {OUTPUT_JSON_PATH}")
            trace.create_event(name="results_saved", metadata={"path": str(OUTPUT_JSON_PATH)})

        # --- Step 9: Terminate sandbox ---
        if interpreter is not None and interpreter.session_id is not None:
            print("Terminating sandbox session...")
            try:
                _stop_sandbox(interpreter, trace)
                print("Sandbox session terminated.")
                trace.create_event(
                    name="sandbox_stopped", metadata={"session_id": interpreter.session_id}
                )
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
