import logging
import os
import uuid
from functools import lru_cache
from typing import Annotated, Any, AsyncGenerator, TypedDict

from ag_ui.core import RunAgentInput
from ag_ui_langgraph import LangGraphAgent
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from dotenv import load_dotenv
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph.state import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from model.load import load_model
from observability import OtelCallbackHandler, agent_span

load_dotenv()

MEMORY_ID = os.environ["MEMORY_ID"]
AWS_REGION = os.environ["AWS_REGION"]

checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID, region_name=AWS_REGION)

app = BedrockAgentCoreApp(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ],
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.info("Starting AgentCore App...")


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


TOOLS = [add, subtract, multiply, divide]


@lru_cache(maxsize=1)
def get_model():
    return load_model().bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state: State) -> State:
    if len(state["messages"]) == 0:
        log.info("No messages in state yet...")
    llm = get_model()
    return {"messages": [llm.invoke(state["messages"])]}


@lru_cache(maxsize=1)
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("calculator", ToolNode(TOOLS))
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "calculator", END: END},
    )
    graph_builder.add_edge("calculator", "agent")
    return graph_builder.compile(checkpointer=checkpointer)


def get_agui_agent() -> LangGraphAgent:
    return LangGraphAgent(
        name="myassistant",
        graph=build_graph(),
        description="LangGraph calculator agent streamed via AG-UI",
        config={
            "configurable": {"actor_id": "agent"},
            "callbacks": [OtelCallbackHandler()],
        },
    )


def _coerce_run_input(payload: dict[str, Any], session_id: str) -> RunAgentInput:
    """Accept either an AG-UI RunAgentInput payload or a legacy {"prompt": "..."} body."""
    if "messages" in payload and "thread_id" in payload:
        return RunAgentInput.model_validate(payload)

    prompt = payload.get("prompt", "")
    return RunAgentInput.model_validate(
        {
            "thread_id": payload.get("thread_id") or session_id,
            "run_id": payload.get("run_id") or str(uuid.uuid4()),
            "state": {},
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": prompt,
                }
            ],
            "tools": [],
            "context": [],
            "forwarded_props": {},
        }
    )


@app.entrypoint
async def invoke(payload, context) -> AsyncGenerator[str, None]:
    log.info("Invoking Agent.....")
    session_id = context.session_id or str(uuid.uuid4())

    run_input = _coerce_run_input(payload, session_id)
    log.info(f"AG-UI run: thread={run_input.thread_id} run={run_input.run_id}")

    request_agent = get_agui_agent()

    with agent_span(
        "agent-invoke",
        {
            "session_id": session_id,
            "thread_id": run_input.thread_id,
            "run_id": run_input.run_id,
        },
    ):
        async for event in request_agent.run(run_input):
            # AgentCore's runtime auto-wraps each yielded value as "data: <json>\n\n",
            # so we hand off the AG-UI event as a JSON-safe dict and let it do the SSE framing.
            yield event.model_dump(by_alias=True, exclude_none=True)


if __name__ == "__main__":
    app.run()
