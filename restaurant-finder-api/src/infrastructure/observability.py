"""
Observability configuration for AgentCore.

Provides OpenTelemetry-based observability with CloudWatch integration for:
- Distributed tracing across agent workflows
- Session ID propagation for conversation tracking
- Custom span creation for detailed monitoring
- CloudWatch GenAI Observability dashboard integration

This module works with the AWS Distro for OpenTelemetry (ADOT) SDK which
automatically instruments the agent to capture telemetry data.

Environment Variables Required:
    AGENT_OBSERVABILITY_ENABLED: Enable observability (default: true)
    OTEL_PYTHON_DISTRO: Set to "aws_distro" for ADOT
    OTEL_PYTHON_CONFIGURATOR: Set to "aws_configurator" for ADOT
    OTEL_EXPORTER_OTLP_PROTOCOL: Set to "http/protobuf"
    OTEL_RESOURCE_ATTRIBUTES: Service name and resource attributes
    OTEL_EXPORTER_OTLP_LOGS_HEADERS: CloudWatch log group configuration

Usage:
    Run with automatic instrumentation:
        opentelemetry-instrument python -m src.main
"""

from typing import Optional
from contextlib import contextmanager

from loguru import logger

# OpenTelemetry imports - gracefully handle if not installed
try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.context import attach, detach
    from opentelemetry.trace import SpanKind, Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Define stub types to prevent NameError at class definition time
    trace = None  # type: ignore
    baggage = None  # type: ignore
    context = None  # type: ignore
    attach = None  # type: ignore
    detach = None  # type: ignore
    SpanKind = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    logger.warning(
        "OpenTelemetry packages not installed. "
        "Observability features will be disabled."
    )


class ObservabilityManager:
    """
    Manages OpenTelemetry observability for the AgentCore application.

    Provides utilities for:
    - Session ID propagation via OpenTelemetry baggage
    - Custom span creation for workflow steps
    - Trace context management
    """

    def __init__(
        self,
        service_name: str = "restaurant-finder-agent",
        enabled: bool = True,
    ):
        """
        Initialize the observability manager.

        Args:
            service_name: Name of the service for tracing attribution
            enabled: Whether observability is enabled
        """
        self.service_name = service_name
        self.enabled = enabled and OTEL_AVAILABLE
        self._tracer = None  # Type: Optional[trace.Tracer] when OTEL available

        if self.enabled and trace is not None:
            self._tracer = trace.get_tracer(
                instrumenting_module_name=service_name,
                tracer_provider=trace.get_tracer_provider(),
            )
            logger.info(f"Observability initialized for service: {service_name}")
        else:
            logger.info("Observability disabled or OpenTelemetry not available")

    def set_session_id(self, session_id: str) -> Optional[object]:
        """
        Set session ID in OpenTelemetry baggage for trace correlation.

        Session IDs enable grouping of traces across multiple requests
        in the same conversation, viewable in CloudWatch GenAI Observability.

        Args:
            session_id: Unique session/conversation identifier

        Returns:
            Context token for detaching, or None if disabled
        """
        if not self.enabled or baggage is None or attach is None:
            return None

        try:
            ctx = baggage.set_baggage("session.id", session_id)
            token = attach(ctx)
            logger.debug(f"Session ID set in observability context: {session_id}")
            return token
        except Exception as e:
            logger.warning(f"Failed to set session ID in baggage: {e}")
            return None

    def clear_session_context(self, token: object) -> None:
        """
        Clear the session context from OpenTelemetry baggage.

        Args:
            token: Context token from set_session_id
        """
        if not self.enabled or token is None or detach is None:
            return

        try:
            detach(token)
            logger.debug("Session context cleared from observability")
        except Exception as e:
            logger.warning(f"Failed to clear session context: {e}")

    @contextmanager
    def session_context(self, session_id: str):
        """
        Context manager for session-scoped observability.

        Automatically sets and clears session ID in OpenTelemetry baggage.

        Args:
            session_id: Unique session/conversation identifier

        Usage:
            with observability.session_context("conversation-123"):
                # All traces within this block will be tagged with session ID
                result = await process_request(...)
        """
        token = self.set_session_id(session_id)
        try:
            yield
        finally:
            self.clear_session_context(token)

    @contextmanager
    def create_span(
        self,
        name: str,
        kind=None,
        attributes: Optional[dict] = None,
    ):
        """
        Create a custom span for detailed workflow tracing.

        Use this to add custom instrumentation for specific operations
        like tool invocations, LLM calls, or business logic steps.

        Args:
            name: Name of the span (e.g., "orchestrator.tool_selection")
            kind: Type of span (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
            attributes: Custom attributes to attach to the span

        Usage:
            with observability.create_span("memory_retrieval", attributes={"actor_id": "user-123"}):
                memories = await memory.retrieve(...)
        """
        if not self.enabled or self._tracer is None:
            # Yield a no-op context when disabled
            yield None
            return

        # Default to INTERNAL span kind when not specified
        if kind is None and SpanKind is not None:
            kind = SpanKind.INTERNAL

        with self._tracer.start_as_current_span(
            name=name,
            kind=kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def add_span_attribute(self, key: str, value: str) -> None:
        """
        Add an attribute to the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled or trace is None:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(key, value)
        except Exception as e:
            logger.warning(f"Failed to add span attribute: {e}")

    def add_span_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """
        Add an event to the current span.

        Events represent discrete occurrences within a span, useful for
        logging tool invocations, LLM responses, or state transitions.

        Args:
            name: Event name (e.g., "tool_invoked", "llm_response_received")
            attributes: Event attributes
        """
        if not self.enabled or trace is None:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.add_event(name, attributes=attributes or {})
        except Exception as e:
            logger.warning(f"Failed to add span event: {e}")

    def record_workflow_step(
        self,
        step_name: str,
        step_type: str,
        duration_ms: Optional[float] = None,
        success: bool = True,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record a workflow step as a span event with standard attributes.

        Args:
            step_name: Name of the workflow step (e.g., "orchestrator", "memory_post_hook")
            step_type: Type of step (e.g., "node", "edge", "tool")
            duration_ms: Duration of the step in milliseconds
            success: Whether the step succeeded
            metadata: Additional metadata to record
        """
        attributes = {
            "workflow.step.name": step_name,
            "workflow.step.type": step_type,
            "workflow.step.success": success,
        }

        if duration_ms is not None:
            attributes["workflow.step.duration_ms"] = duration_ms

        if metadata:
            for key, value in metadata.items():
                # Prefix custom metadata to avoid conflicts
                attributes[f"workflow.step.{key}"] = str(value)

        self.add_span_event(f"workflow.{step_name}", attributes)


# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager() -> ObservabilityManager:
    """
    Get the global observability manager instance.

    Returns:
        ObservabilityManager singleton instance
    """
    global _observability_manager

    if _observability_manager is None:
        from src.config import settings

        _observability_manager = ObservabilityManager(
            service_name=settings.OTEL_SERVICE_NAME,
            enabled=settings.AGENT_OBSERVABILITY_ENABLED,
        )

    return _observability_manager


def initialize_observability(
    service_name: str = "restaurant-finder-agent",
    enabled: bool = True,
) -> ObservabilityManager:
    """
    Initialize the global observability manager.

    Call this at application startup to configure observability.

    Args:
        service_name: Name of the service for tracing
        enabled: Whether observability is enabled

    Returns:
        Initialized ObservabilityManager instance
    """
    global _observability_manager

    _observability_manager = ObservabilityManager(
        service_name=service_name,
        enabled=enabled,
    )

    return _observability_manager
