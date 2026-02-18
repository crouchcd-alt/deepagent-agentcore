from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model configurations ---
    ORCHESTRATOR_MODEL_ID: str = Field(
        default="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        description="Model for main orchestrator (tool selection, conversation management).",
    )
    EXTRACTION_MODEL_ID: str = Field(
        default="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        description="Model for data extraction tasks (JSON parsing, structured data).",
    )
    ROUTER_MODEL_ID: str = Field(
        default="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        description="Model for router/intent classification (lightweight, fast).",
    )

    # --- Browser Tools Configuration ---
    ENABLE_BROWSER_TOOLS: bool = Field(
        default=True,
        description="Enable browser-based tools (restaurant_explorer, restaurant_research).",
    )

    # --- Guardrails configurations ---
    BEDROCK_GUARDRAIL_NAME: str = Field(
        default="restaurant-finder-guardrail",
        description="The name for the Bedrock guardrail (used for create-or-get).",
    )
    BEDROCK_GUARDRAIL_ID: str = Field(
        default="",
        description="The Bedrock guardrail ID (auto-populated on startup if empty).",
    )
    BEDROCK_GUARDRAIL_VERSION: str = Field(
        default="DRAFT",
        description="The Bedrock guardrail version (e.g., 'DRAFT', '1', '2').",
    )
    GUARDRAIL_ENABLED: bool = Field(
        default=True,
        description="Enable or disable guardrails globally.",
    )

    # --- AWS configurations ---
    AWS_REGION: str = Field(
        default="us-east-2",
        description="The AWS region where services are hosted.",
    )

    # --- AgentCore Gateway configurations ---
    GATEWAY_URL: str = Field(
        default="",
        description="The AgentCore MCP Gateway URL for tool routing.",
    )
    GATEWAY_ID: str = Field(
        default="",
        description="The AgentCore Gateway ID.",
    )
    RUNTIME_ID: str = Field(
        default="",
        description="The AgentCore Runtime ID.",
    )

    # --- AgentCore Memory configurations ---
    MEMORY_ID: str = Field(
        default="",
        description="The AgentCore Memory ID. Must be set from CDK stack output.",
    )

    # --- Observability configurations ---
    AGENT_OBSERVABILITY_ENABLED: bool = Field(
        default=True,
        description="Enable OpenTelemetry-based observability for CloudWatch GenAI Observability.",
    )
    OTEL_SERVICE_NAME: str = Field(
        default="restaurant-finder-agent",
        description="Service name for OpenTelemetry tracing attribution.",
    )

    # --- Evaluation configurations ---
    EVALUATION_ENABLED: bool = Field(
        default=True,
        description="Enable AgentCore Evaluations for agent quality monitoring.",
    )
    EVALUATION_SAMPLING_RATE: int = Field(
        default=10,
        description="Percentage of sessions to evaluate in online mode (1-100).",
    )
    EVALUATION_OUTPUT_DIR: str = Field(
        default="evaluation_results",
        description="Directory for storing evaluation results.",
    )


settings = Settings()
