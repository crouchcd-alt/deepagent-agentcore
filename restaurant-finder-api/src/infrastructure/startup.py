"""
Application startup initialization.

Handles initialization of infrastructure components like memory, guardrails,
and observability before the application starts accepting requests.
"""

from loguru import logger

from src.config import settings
from src.infrastructure.guardrails import get_guardrail_manager
from src.infrastructure.observability import initialize_observability


_startup_complete: bool = False


async def initialize_infrastructure() -> dict:
    """
    Initialize all infrastructure components on application startup.

    This function is idempotent - calling it multiple times will only
    initialize components once.

    Initializes:
    - Observability (OpenTelemetry with CloudWatch integration)
    - Guardrails (Bedrock content moderation)

    Returns:
        Dictionary with initialization status for each component.
    """
    global _startup_complete

    if _startup_complete:
        logger.debug("Infrastructure already initialized, skipping")
        return {"status": "already_initialized"}

    logger.info("Initializing application infrastructure...")

    results = {
        "observability": {"status": "pending"},
        "guardrails": {"status": "pending"},
    }

    # Initialize Observability
    if settings.AGENT_OBSERVABILITY_ENABLED:
        try:
            logger.info("Initializing observability...")
            observability_manager = initialize_observability(
                service_name=settings.OTEL_SERVICE_NAME,
                enabled=settings.AGENT_OBSERVABILITY_ENABLED,
            )
            results["observability"] = {
                "status": "success",
                "service_name": settings.OTEL_SERVICE_NAME,
                "enabled": observability_manager.enabled,
            }
            logger.info(
                f"Observability initialized: service={settings.OTEL_SERVICE_NAME}, "
                f"enabled={observability_manager.enabled}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize observability: {e}")
            results["observability"] = {
                "status": "error",
                "error": str(e),
            }
    else:
        results["observability"] = {"status": "disabled"}
        logger.info("Observability disabled by configuration")

    # Initialize Guardrails
    if settings.GUARDRAIL_ENABLED:
        try:
            logger.info("Initializing guardrails...")
            guardrail_manager = get_guardrail_manager()
            guardrail_info = guardrail_manager.create_or_get_guardrail()

            # Update settings with the guardrail ID for use elsewhere
            settings.BEDROCK_GUARDRAIL_ID = guardrail_info["id"]
            settings.BEDROCK_GUARDRAIL_VERSION = guardrail_info["version"]

            results["guardrails"] = {
                "status": "success",
                "guardrail_id": guardrail_info["id"],
                "guardrail_version": guardrail_info["version"],
            }
            logger.info(
                f"Guardrails initialized: {guardrail_info['id']} "
                f"(version: {guardrail_info['version']})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize guardrails: {e}")
            results["guardrails"] = {
                "status": "error",
                "error": str(e),
            }
    else:
        results["guardrails"] = {"status": "disabled"}
        logger.info("Guardrails disabled by configuration")

    _startup_complete = True
    logger.info("Infrastructure initialization complete")

    return results


def is_initialized() -> bool:
    """Check if infrastructure has been initialized."""
    return _startup_complete
