"""
Guardrail infrastructure for AWS Bedrock Guardrails.

Provides automatic creation or retrieval of guardrails on application startup,
with configurable content filtering policies for the restaurant finder agent.
"""

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from src.config import settings


class GuardrailManager:
    """
    Manages Bedrock Guardrails lifecycle - creates or retrieves existing guardrails.

    Similar pattern to ShortTermMemory for consistency.
    """

    def __init__(self, guardrail_name: str | None = None):
        self.guardrail_name = guardrail_name or settings.BEDROCK_GUARDRAIL_NAME
        self._client = boto3.client("bedrock", region_name=settings.AWS_REGION)
        self._guardrail_id: str | None = None
        self._guardrail_version: str | None = None

    @property
    def guardrail_id(self) -> str | None:
        """Get the guardrail ID (available after initialization)."""
        return self._guardrail_id

    @property
    def guardrail_version(self) -> str | None:
        """Get the guardrail version (available after initialization)."""
        return self._guardrail_version

    def _get_default_content_policy(self) -> dict:
        """
        Get default content filter policy for the restaurant finder.

        Configures filters for harmful content categories with appropriate
        strengths for a restaurant recommendation application.
        """
        return {
            "filtersConfig": [
                {
                    "type": "HATE",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH",
                },
                {
                    "type": "INSULTS",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH",
                },
                {
                    "type": "SEXUAL",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH",
                },
                {
                    "type": "VIOLENCE",
                    "inputStrength": "MEDIUM",
                    "outputStrength": "MEDIUM",
                },
                {
                    "type": "MISCONDUCT",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH",
                },
                {
                    "type": "PROMPT_ATTACK",
                    "inputStrength": "MEDIUM",  # HIGH causes too many false positives
                    "outputStrength": "NONE",
                },
            ]
        }

    def _get_default_topic_policy(self) -> dict:
        """
        Get default topic policy for the restaurant finder.

        Blocks topics that are not relevant to restaurant recommendations.

        Note: We intentionally don't include a broad "off-topic" topic filter
        as it tends to block legitimate queries. The model's system prompt
        handles staying on topic more gracefully.
        """
        return {
            "topicsConfig": [
                {
                    "name": "competitor-services",
                    "definition": "Discussions about competing food delivery or restaurant booking services, apps, or platforms.",
                    "examples": [
                        "Can you order from DoorDash instead?",
                        "Is UberEats better than this?",
                        "Compare this to Yelp",
                    ],
                    "type": "DENY",
                },
                {
                    "name": "illegal-activities",
                    "definition": "Any requests related to illegal activities, substances, or services.",
                    "examples": [
                        "Where can I buy drugs?",
                        "Find restaurants that serve illegal wildlife",
                    ],
                    "type": "DENY",
                },
            ]
        }

    def _get_default_sensitive_info_policy(self) -> dict:
        """
        Get default sensitive information policy.

        Masks PII in inputs and outputs for privacy protection.
        """
        return {
            "piiEntitiesConfig": [
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"},
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
            ]
        }

    def _get_default_word_policy(self) -> dict:
        """
        Get default word filter policy.

        Blocks profanity and configures managed word lists.
        """
        return {
            "managedWordListsConfig": [
                {"type": "PROFANITY"},
            ]
        }

    def _find_existing_guardrail(self) -> dict | None:
        """
        Find an existing guardrail by name.

        Returns:
            Guardrail details if found, None otherwise.
        """
        try:
            paginator = self._client.get_paginator("list_guardrails")
            for page in paginator.paginate():
                for guardrail in page.get("guardrails", []):
                    if guardrail.get("name") == self.guardrail_name:
                        logger.info(f"Found existing guardrail: {guardrail['id']}")
                        return guardrail
        except ClientError as e:
            logger.warning(f"Error listing guardrails: {e}")
        return None

    def _create_guardrail(self) -> dict:
        """
        Create a new guardrail with default policies.

        Returns:
            Created guardrail details.
        """
        logger.info(f"Creating new guardrail: {self.guardrail_name}")

        response = self._client.create_guardrail(
            name=self.guardrail_name,
            description="Content guardrail for the Restaurant Finder AI agent. "
                        "Filters harmful content, blocks off-topic requests, "
                        "and protects user privacy.",
            contentPolicyConfig=self._get_default_content_policy(),
            topicPolicyConfig=self._get_default_topic_policy(),
            sensitiveInformationPolicyConfig=self._get_default_sensitive_info_policy(),
            wordPolicyConfig=self._get_default_word_policy(),
            blockedInputMessaging=(
                "I'm sorry, but I can't process that request. "
                "I'm here to help you find great restaurants! "
                "Please ask me about dining options, cuisines, or restaurant recommendations."
            ),
            blockedOutputsMessaging=(
                "I apologize, but I cannot provide that information. "
                "Let me help you find a wonderful restaurant instead!"
            ),
        )

        logger.info(
            f"Created guardrail: {response['guardrailId']} "
            f"(version: {response['version']})"
        )

        return {
            "id": response["guardrailId"],
            "version": response["version"],
            "arn": response["guardrailArn"],
        }

    def create_or_get_guardrail(self) -> dict:
        """
        Create a new guardrail or retrieve an existing one by name.

        Returns:
            Dictionary with guardrail id, version, and arn.
        """
        # Check if already initialized
        if self._guardrail_id:
            return {
                "id": self._guardrail_id,
                "version": self._guardrail_version,
            }

        # Try to find existing guardrail
        existing = self._find_existing_guardrail()

        if existing:
            self._guardrail_id = existing["id"]
            self._guardrail_version = existing.get("version", "DRAFT")
            return {
                "id": self._guardrail_id,
                "version": self._guardrail_version,
            }

        # Create new guardrail
        created = self._create_guardrail()
        self._guardrail_id = created["id"]
        self._guardrail_version = created["version"]

        return {
            "id": self._guardrail_id,
            "version": self._guardrail_version,
        }

    def publish_version(self, description: str = "Production version") -> str:
        """
        Publish a new version of the guardrail (moves from DRAFT to numbered version).

        Args:
            description: Description for the new version.

        Returns:
            The new version number.
        """
        if not self._guardrail_id:
            raise ValueError("Guardrail not initialized. Call create_or_get_guardrail first.")

        response = self._client.create_guardrail_version(
            guardrailIdentifier=self._guardrail_id,
            description=description,
        )

        new_version = response["version"]
        logger.info(f"Published guardrail version: {new_version}")

        self._guardrail_version = new_version
        return new_version

    def get_guardrail_config(self) -> dict | None:
        """
        Get the guardrail configuration for use with ChatBedrockConverse.

        Returns:
            Guardrail config dict or None if disabled/not initialized.
        """
        if not settings.GUARDRAIL_ENABLED:
            return None

        if not self._guardrail_id:
            return None

        return {
            "guardrailIdentifier": self._guardrail_id,
            "guardrailVersion": self._guardrail_version or "DRAFT",
        }


# Singleton instance for application-wide use
_guardrail_manager: GuardrailManager | None = None


def get_guardrail_manager() -> GuardrailManager:
    """Get or create the singleton GuardrailManager instance."""
    global _guardrail_manager
    if _guardrail_manager is None:
        _guardrail_manager = GuardrailManager()
    return _guardrail_manager


# =============================================================================
# Graph-Level Guardrail Functions (ApplyGuardrail API)
# =============================================================================

class GuardrailResult:
    """Result of a guardrail check."""

    def __init__(
        self,
        allowed: bool,
        output: str,
        action: str,
        assessments: list | None = None,
    ):
        self.allowed = allowed
        self.output = output
        self.action = action
        self.assessments = assessments or []

    def __repr__(self) -> str:
        return f"GuardrailResult(allowed={self.allowed}, action={self.action})"


def apply_input_guardrail(text: str) -> GuardrailResult:
    """
    Apply guardrail to user input at graph entry.

    Uses the ApplyGuardrail API to check content independently of model calls.
    This is more efficient than applying guardrails at each model invocation.

    Args:
        text: The user input text to check.

    Returns:
        GuardrailResult with allowed status and potentially modified output.
    """
    if not settings.GUARDRAIL_ENABLED:
        return GuardrailResult(allowed=True, output=text, action="NONE")

    manager = get_guardrail_manager()
    if not manager.guardrail_id:
        logger.warning("Guardrail not initialized, skipping input check")
        return GuardrailResult(allowed=True, output=text, action="NONE")

    try:
        # Use bedrock-runtime for ApplyGuardrail
        runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
        )

        response = runtime_client.apply_guardrail(
            guardrailIdentifier=manager.guardrail_id,
            guardrailVersion=manager.guardrail_version or "DRAFT",
            source="INPUT",
            content=[{"text": {"text": text}}],
        )

        action = response.get("action", "NONE")
        outputs = response.get("outputs", [])
        assessments = response.get("assessments", [])

        # Get the output text (may be modified/blocked)
        if outputs:
            output_text = outputs[0].get("text", text)
        else:
            output_text = text

        allowed = action != "GUARDRAIL_INTERVENED"

        if not allowed:
            logger.info(f"Input guardrail blocked content: action={action}")
            # Log detailed assessment info for debugging
            for assessment in assessments:
                if assessment.get("topicPolicy"):
                    topics = assessment["topicPolicy"].get("topics", [])
                    for topic in topics:
                        if topic.get("action") == "BLOCKED":
                            logger.warning(f"  Blocked by topic: {topic.get('name')} - {topic.get('type')}")
                if assessment.get("contentPolicy"):
                    filters = assessment["contentPolicy"].get("filters", [])
                    for f in filters:
                        if f.get("action") == "BLOCKED":
                            logger.warning(f"  Blocked by content filter: {f.get('type')} (confidence: {f.get('confidence')})")
                if assessment.get("wordPolicy"):
                    words = assessment["wordPolicy"].get("customWords", []) + assessment["wordPolicy"].get("managedWordLists", [])
                    for w in words:
                        if w.get("action") == "BLOCKED":
                            logger.warning(f"  Blocked by word policy: {w.get('match', w.get('type'))}")

        return GuardrailResult(
            allowed=allowed,
            output=output_text,
            action=action,
            assessments=assessments,
        )

    except ClientError as e:
        logger.error(f"Error applying input guardrail: {e}")
        # Fail open - allow the request but log the error
        return GuardrailResult(allowed=True, output=text, action="ERROR")


def apply_output_guardrail(text: str) -> GuardrailResult:
    """
    Apply guardrail to model output at graph exit.

    Uses the ApplyGuardrail API to check content independently of model calls.

    Args:
        text: The model output text to check.

    Returns:
        GuardrailResult with allowed status and potentially modified output.
    """
    if not settings.GUARDRAIL_ENABLED:
        return GuardrailResult(allowed=True, output=text, action="NONE")

    manager = get_guardrail_manager()
    if not manager.guardrail_id:
        logger.warning("Guardrail not initialized, skipping output check")
        return GuardrailResult(allowed=True, output=text, action="NONE")

    try:
        runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
        )

        response = runtime_client.apply_guardrail(
            guardrailIdentifier=manager.guardrail_id,
            guardrailVersion=manager.guardrail_version or "DRAFT",
            source="OUTPUT",
            content=[{"text": {"text": text}}],
        )

        action = response.get("action", "NONE")
        outputs = response.get("outputs", [])
        assessments = response.get("assessments", [])

        if outputs:
            output_text = outputs[0].get("text", text)
        else:
            output_text = text

        allowed = action != "GUARDRAIL_INTERVENED"

        if not allowed:
            logger.info(f"Output guardrail modified content: action={action}")

        return GuardrailResult(
            allowed=allowed,
            output=output_text,
            action=action,
            assessments=assessments,
        )

    except ClientError as e:
        logger.error(f"Error applying output guardrail: {e}")
        return GuardrailResult(allowed=True, output=text, action="ERROR")


def get_blocked_input_message() -> str:
    """Get the default message to show when input is blocked."""
    return (
        "I'm sorry, but I can't process that request. "
        "I'm here to help you find great restaurants! "
        "Please ask me about dining options, cuisines, or restaurant recommendations."
    )


def get_blocked_output_message() -> str:
    """Get the default message to show when output is blocked."""
    return (
        "I apologize, but I cannot provide that information. "
        "Let me help you find a wonderful restaurant instead!"
    )
