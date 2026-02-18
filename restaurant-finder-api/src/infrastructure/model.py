"""
Model infrastructure for Bedrock LLM access.

Provides centralized model loading with configurable parameters and
a helper for parsing Bedrock's response content formats.
"""

from enum import Enum
from typing import Any

from langchain_aws import ChatBedrockConverse

from src.config import settings


def extract_text_content(content: Any) -> str:
    """
    Extract text from a Bedrock model response.

    Handles both plain string content and Bedrock's list-of-blocks format:
    [{'type': 'text', 'text': '...', 'index': 0}]

    Args:
        content: Response content (string, list of content blocks, or other).

    Returns:
        Extracted text as a string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                text_parts.append(str(block))
        return "".join(text_parts)

    return str(content) if content else ""


class ModelType(str, Enum):
    """Model types for different components of the system."""

    ORCHESTRATOR = "orchestrator"
    EXTRACTION = "extraction"
    ROUTER = "router"


def _get_model_id_for_type(model_type: ModelType) -> str:
    """Get the configured model ID for a given model type."""
    model_map = {
        ModelType.ORCHESTRATOR: settings.ORCHESTRATOR_MODEL_ID,
        ModelType.EXTRACTION: settings.EXTRACTION_MODEL_ID,
        ModelType.ROUTER: settings.ROUTER_MODEL_ID,
    }
    return model_map.get(model_type, settings.ORCHESTRATOR_MODEL_ID)


def get_model(
    temperature: float = 0.7,
    model_id: str | None = None,
    model_type: ModelType = ModelType.ORCHESTRATOR,
) -> ChatBedrockConverse:
    """
    Get a ChatBedrockConverse model instance.

    Args:
        temperature: Model temperature (0.0-1.0).
        model_id: Model ID override. Takes precedence over model_type.
        model_type: The type of model to use (orchestrator, extraction, router).

    Returns:
        ChatBedrockConverse: Configured model instance.
    """
    resolved_model_id = model_id or _get_model_id_for_type(model_type)

    return ChatBedrockConverse(
        model=resolved_model_id,
        temperature=temperature,
    )
