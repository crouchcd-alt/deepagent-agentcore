"""
API entrypoint for the Restaurant Finder Agent.

Provides the main entrypoint for BedrockAgentCoreApp with startup hooks
for initializing infrastructure components.
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from loguru import logger

from src.infrastructure.startup import initialize_infrastructure
from src.infrastructure.streaming import stream_response

# Initialize BedrockAgentCoreApp
app = BedrockAgentCoreApp()


@app.on_event("startup")
async def startup_event():
    """
    Application startup hook.

    Initializes infrastructure components:
    - Observability (OpenTelemetry with CloudWatch GenAI Observability)
    - Memory system (creates or retrieves existing memory)
    - Guardrails (creates or retrieves existing guardrail)
    """
    logger.info("Application startup initiated")
    results = await initialize_infrastructure()

    # Log initialization summary
    observability_status = results.get("observability", {}).get("status", "unknown")
    guardrail_status = results.get("guardrails", {}).get("status", "unknown")

    logger.info(
        f"Startup complete - Observability: {observability_status}, "
        f"Guardrails: {guardrail_status}"
    )

    if observability_status == "error":
        logger.warning(
            f"Observability initialization error: {results['observability'].get('error')}"
        )

    if guardrail_status == "error":
        logger.warning(
            f"Guardrail initialization error: {results['guardrails'].get('error')}"
        )


@app.entrypoint
async def invoke(payload: dict):
    """
    Main entrypoint for the agent.

    Expected payload:
    {
        "prompt": "<user input>",               # Required
        "customer_name": "<customer name>",     # Optional, defaults to "Guest"
        "conversation_id": "<conversation id>"  # Optional, unique ID for conversation thread
    }

    Returns an async generator for streaming, or dict for errors.
    """
    user_input = payload.get("prompt", "")
    if not user_input:
        return {"error": "No prompt provided in the payload."}

    # Extract optional customer context
    customer_name = payload.get("customer_name", "Guest")
    conversation_id = payload.get("conversation_id")

    # Return async generator - BedrockAgentCoreApp handles streaming automatically
    return stream_response(
        user_input=user_input,
        customer_name=customer_name,
        conversation_id=conversation_id,
    )


if __name__ == "__main__":
    app.run()
