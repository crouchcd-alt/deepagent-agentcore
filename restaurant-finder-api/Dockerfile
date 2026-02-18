FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1

COPY pyproject.toml pyproject.toml
# Install from requirements file
RUN uv pip install -r pyproject.toml

# Install AWS OpenTelemetry distro for observability
# Note: Version should be kept in sync with pyproject.toml dependencies
RUN uv pip install aws-opentelemetry-distro>=0.12.0

# Signal that this is running in Docker for host binding logic
ENV DOCKER_CONTAINER=1

# OpenTelemetry configuration for AWS CloudWatch observability
# These enable automatic instrumentation with AWS Distro for OpenTelemetry (ADOT)
ENV AGENT_OBSERVABILITY_ENABLED=true \
    OTEL_PYTHON_DISTRO=aws_distro \
    OTEL_PYTHON_CONFIGURATOR=aws_configurator \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    OTEL_TRACES_EXPORTER=otlp \
    OTEL_SERVICE_NAME=restaurant-finder-agent

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

# Copy entire project (respecting .dockerignore)
COPY . .

# Run with OpenTelemetry automatic instrumentation
# This enables distributed tracing and CloudWatch GenAI Observability
CMD ["opentelemetry-instrument", "python", "-m", "src.main"]
