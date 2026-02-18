from typing import NamedTuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from loguru import logger

from src.config import settings
from src.domain.prompts import (
    SEARCH_AGENT_PROMPT,
    ROUTER_PROMPT,
    SIMPLE_RESPONSE_PROMPT,
)
from src.application.orchestrator.workflow.tools import get_orchestrator_tools
from src.infrastructure.model import get_model, ModelType


class PromptMetadata(NamedTuple):
    """Metadata about the prompt used for observability tracing."""
    name: str
    version: str | None
    id: str | None
    arn: str | None


class SearchAgentChainResult(NamedTuple):
    """Result containing the search agent chain and its prompt metadata for tracing."""
    chain: Runnable
    prompt_metadata: PromptMetadata


def _escape_braces(text: str) -> str:
    """
    Escape curly braces to prevent ChatPromptTemplate from interpreting them as variables.

    Args:
        text: The text to escape

    Returns:
        Text with { replaced by {{ and } replaced by }}
    """
    return text.replace("{", "{{").replace("}", "}}")


def get_search_agent_prompt_metadata() -> PromptMetadata:
    """
    Get metadata about the search agent prompt for observability tracing.

    Returns:
        PromptMetadata with name, version, id, and arn from Bedrock sync.
    """
    bedrock_meta = SEARCH_AGENT_PROMPT.bedrock_metadata
    if bedrock_meta:
        return PromptMetadata(
            name=bedrock_meta.get("name", SEARCH_AGENT_PROMPT.name),
            version=bedrock_meta.get("version"),
            id=bedrock_meta.get("id"),
            arn=bedrock_meta.get("arn"),
        )
    return PromptMetadata(
        name=SEARCH_AGENT_PROMPT.name,
        version=None,
        id=None,
        arn=None,
    )


def get_search_agent_chain(
    customer_name: str = "Guest",
    include_browser_tools: bool | None = None,
) -> SearchAgentChainResult:
    """
    Create the search agent chain with bound tools for restaurant searching.

    The search agent uses tools for:
    - memory_retrieval_tool: On-demand memory retrieval (preferences, facts, summaries)
    - restaurant_data_tool: MCP Gateway for structured restaurant data
    - restaurant_explorer_tool: For searching/exploring restaurants via browser (optional)
    - restaurant_research_tool: For detailed restaurant research (optional)

    Tool availability depends on the ENABLE_BROWSER_TOOLS config setting:
    - When False (default): Only MCP-based tools are available (faster)
    - When True: Browser-based tools are also available (more comprehensive)

    Memory is retrieved ON-DEMAND via the memory_retrieval_tool rather than
    being pre-loaded into the prompt. This gives the agent control over when
    and what type of memory to retrieve.

    Args:
        customer_name: The customer's name for personalization.
        include_browser_tools: Override for browser tools inclusion.
                              If None, uses ENABLE_BROWSER_TOOLS from config.

    Returns:
        SearchAgentChainResult with:
        - chain: A runnable chain (prompt | model) with tools bound
        - prompt_metadata: Metadata about the prompt for tracing
    """
    # Determine browser tools setting
    use_browser = include_browser_tools if include_browser_tools is not None else settings.ENABLE_BROWSER_TOOLS

    logger.info(f"Creating search agent chain (browser_tools={'enabled' if use_browser else 'disabled'})")

    model = get_model(temperature=0.5, model_type=ModelType.ORCHESTRATOR)

    # Get the appropriate tools based on config
    tools = get_orchestrator_tools(include_browser_tools=use_browser)
    tool_names = [t.name for t in tools]

    # Bind the tools to the model
    model = model.bind_tools(tools)

    # Escape braces in dynamic content to prevent ChatPromptTemplate from
    # interpreting them as template variables
    safe_customer_name = _escape_braces(customer_name)

    # Get the search agent prompt and its metadata
    system_message = SEARCH_AGENT_PROMPT.format(
        customer_name=safe_customer_name,
    )
    prompt_metadata = get_search_agent_prompt_metadata()

    logger.debug(
        f"Search agent prompt: name={prompt_metadata.name}, "
        f"version={prompt_metadata.version}, tools={tool_names}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return SearchAgentChainResult(
        chain=prompt | model,
        prompt_metadata=prompt_metadata,
    )


def get_router_chain() -> Runnable:
    """
    Create the router chain for intent classification.

    This is a lightweight chain that classifies user intent into:
    - restaurant_search: User wants to find/search restaurants
    - simple: Greetings, thanks, questions about the assistant
    - off_topic: Unrelated questions

    Uses a fast model for low latency routing decisions.

    Returns:
        A runnable chain (prompt | model) for intent classification.
    """
    logger.debug("Creating router chain for intent classification")

    # Use the dedicated router model (low temperature for deterministic classification)
    model = get_model(temperature=0.0, model_type=ModelType.ROUTER)

    system_message = ROUTER_PROMPT.prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model


def get_simple_response_chain(customer_name: str = "Guest") -> Runnable:
    """
    Create the simple response chain for non-restaurant queries.

    This chain handles:
    - Greetings and welcomes
    - Thanks and acknowledgments
    - Questions about capabilities
    - Off-topic redirections

    Args:
        customer_name: The customer's name for personalization.

    Returns:
        A runnable chain (prompt | model) for simple responses.
    """
    logger.debug(f"Creating simple response chain for {customer_name}")

    model = get_model(temperature=0.7, model_type=ModelType.ORCHESTRATOR)

    # Escape braces in dynamic content
    safe_customer_name = _escape_braces(customer_name)

    system_message = SIMPLE_RESPONSE_PROMPT.format(
        customer_name=safe_customer_name,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model
