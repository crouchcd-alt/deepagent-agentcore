# %%
import json
import os
from pathlib import Path

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_model_call, wrap_tool_call
from langchain_agentcore_codeinterpreter import AgentCoreSandbox
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, BaseMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from loguru import logger

assert load_dotenv()

os.environ["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"]
os.environ["AWS_REGION"]
os.environ["AWS_PROFILE"]
os.environ["BEDROCK_AGENT_MODEL_ID"]

# %%

# Initialize Langfuse client
langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# Initialize Langfuse CallbackHandler for LangChain (tracing)
langfuse_handler = CallbackHandler()

# %%


class EmptyMessage(BaseMessage):
    """A placeholder message with empty content."""

    type: str = "empty"

    def __init__(self, **kwargs):
        super().__init__(content="", **kwargs)


# logger.remove(0)
logger.add(
    "logs/deepagent_pathology.log",
    rotation="10 MB",
    retention=5,
    format="{time} {level} {message}",
)


@wrap_tool_call
def _log_tool_call(request, handler):
    logger.info(
        "Last message | last_message={}",
        request.state.get("messages", [EmptyMessage()])[-1].content,
    )
    logger.info(
        "Tool called | tool_name={} tool_args={}",
        request.tool.name,
        request.tool_call.get("args", {}),
    )
    return handler(request)


@wrap_model_call
def _log_model_call(request, handler):
    messages = request.state.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is not None:
        if last_ai.tool_calls:
            for tc in last_ai.tool_calls:
                logger.info(
                    "Last AI tool_call | name={} args={}",
                    tc.get("name"),
                    tc.get("args", {}),
                )
        if last_ai.content:
            logger.info("Last AI content | content={}", last_ai.content)
    logger.info("Model called")
    return handler(request)


agent_loggers = [
    _log_tool_call,
    _log_model_call,
]

REPORT_PATH = Path("../data/synthetic_pathology_report_AML.pdf").resolve()
SCHEMA_PATH = Path("./src/schemas/pathology.py").resolve()


def start_sandbox():
    """Provision the AgentCore code-interpreter sandbox."""
    interpreter = CodeInterpreter(region=os.environ["AWS_REGION"])
    interpreter.start()
    sandbox = AgentCoreSandbox(interpreter=interpreter)
    logger.info(
        "AgentCore sandbox started | session_id={}", getattr(interpreter, "session_id", "?")
    )
    return interpreter, sandbox


def end_sandbox(interpreter):
    """Stop the AgentCore sandbox."""
    interpreter.stop()
    logger.info(
        "AgentCore sandbox stopped | session_id={}", getattr(interpreter, "session_id", "?")
    )


def sandbox_info(sandbox):
    cwd = sandbox.execute("pwd").output.strip()
    logger.info("Sandbox working directory: {}", cwd)
    py_version = sandbox.execute("python --version").output.strip()
    logger.info("Python version: {}", py_version)
    py_path = sandbox.execute("python -c 'import sys; print(sys.executable)'").output.strip()
    logger.info("Python path: {}", py_path)
    return {"cwd": cwd, "py_version": py_version, "py_path": py_path}


def upload_files(sandbox, cwd: str) -> tuple[str, str]:
    with open(REPORT_PATH, "rb") as f:
        sandbox.upload_files([("pathology_report.pdf", f.read())])
    with open(SCHEMA_PATH, "rb") as f:
        sandbox.upload_files([("schema.py", f.read())])

    report_abs = f"{cwd}/pathology_report.pdf"
    schema_abs = f"{cwd}/schema.py"

    # Verify both files are visible at the expected paths before handing them
    # to the agent — fail fast rather than letting the agent churn.
    check = sandbox.execute(f"ls {report_abs} {schema_abs}")
    if check.exit_code != 0:
        raise RuntimeError(
            f"Uploaded files not found at expected paths (cwd={cwd}): {check.output.strip()}"
        )

    logger.info("Files confirmed in sandbox | report={} schema={}", report_abs, schema_abs)
    return report_abs, schema_abs


OUTPUT_FILE = "pathology_results.json"

PROMPT = """\
Parse the pathology report PDF at {report_path} (your current working directory is {cwd}).

**guidelines:**
- Use `read_file` with offsets + limits to prevent reading the entire file into context. Some files are large (~2MB) and 3000+ rows.

Python {py_version} is provided at {py_path}. You can use the following script to list the available Python packages:
```python
import pkg_resources
installed_packages = [d for d in pkg_resources.working_set]
for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
    print(f"{{package.project_name}}=={{package.version}}")
```
If there's a package you want to use that's not installed, you should end your turn with an explanation for why the package is needed
so that the user can install it and re-run you.

Your goal is to extract the following fields from the pathology report and validate them against the Pydantic schema at {schema_path}:
- **age**: The patient's age (integer).
- **primary_diagnosis**: The primary diagnosis from the report.
- **performance_status**: The ECOG or similar performance status, if present.

**Citation rules — every extracted value MUST include a verifiable citation:**
- Each field is a `CitedField` with a `value` and a `citation`.
- A `citation` contains `page` (1-indexed page number) and `text` (the **exact** quote from the PDF that supports the value — copy it verbatim, do not paraphrase).
- The `text` must be a substring that can be found on the cited page when the PDF is re-parsed, so it can be programmatically verified.
- If a field is not present in the report, set it to `null` rather than fabricating a citation.

**Output rules:**
1. Write a Python script that extracts text from the PDF, parses the required fields along with their citations,
   validates each record with the Pydantic schema, and writes the result to `{cwd}/{output_file}` using json.dump().
2. The output file must contain the results produced by the Pydantic model's `.model_dump()`.
3. Your final message should be ONLY a JSON status (no prose):
   {{"total_fields_extracted": <int>, "file": "{cwd}/{output_file}", "file_size_bytes": <size>}}
"""


def _read_result_file(sandbox, file_path: str) -> dict:
    """Read the JSON result file from the sandbox."""
    logger.info("Reading result file from sandbox: {}", file_path)
    result = sandbox.execute(f"cat {file_path}")
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to read {file_path}: {result.output.strip()}")
    return json.loads(result.output)


def run(sandbox, sandbox_infos):
    report_path, schema_path = upload_files(sandbox, sandbox_infos["cwd"])

    agent = create_deep_agent(
        model=ChatBedrock(
            model="us.anthropic.claude-sonnet-4-6",
            region=os.environ["AWS_REGION"],
            max_tokens=4096,
        ),
        backend=sandbox,
        middleware=[
            *agent_loggers,
            ToolCallLimitMiddleware(
                run_limit=40,
                exit_behavior="end",
            ),
        ],
    )

    # Invoke the agent with Langfuse tracing
    logger.info("Invoking agent with Langfuse tracing...")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        report_path=report_path,
                        schema_path=schema_path,
                        output_file=OUTPUT_FILE,
                        **sandbox_infos,
                    ),
                }
            ]
        },
        config={"callbacks": [langfuse_handler]},
    )

    # Log the agent's summary message
    final_msg = result["messages"][-1]
    raw_text = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
    logger.info("Agent final message: {}", raw_text[:500])

    # Read the result file directly from the sandbox
    data = _read_result_file(sandbox, f"{sandbox_infos['cwd']}/{OUTPUT_FILE}")
    logger.info(
        "Result file loaded | records={} errors={}",
        len(data.get("records", [])),
        len(data.get("validation_errors", [])),
    )
    return data


try:
    interpreter, sandbox = start_sandbox()
    sandbox_infos = sandbox_info(sandbox)
    result = run(sandbox=sandbox, sandbox_infos=sandbox_infos)
    logger.info(
        "Extraction complete | records={} errors={}",
        len(result.get("records", [])),
        len(result.get("validation_errors", [])),
    )
finally:
    if "interpreter" in locals():
        end_sandbox(interpreter)
