import re

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from src.config import settings


class Prompt:
    """
    A prompt template with Bedrock synchronization.

    Wraps a prompt text template with automatic syncing to AWS Bedrock
    Prompt Management using CHAT template type for proper system/user
    role separation.

    The prompt text is stored as the system message, with a generic
    user message placeholder for Bedrock's template representation.
    At runtime, actual user messages are assembled by the calling code
    via LangChain's ChatPromptTemplate.
    """

    def __init__(self, name: str, prompt: str) -> None:
        self.name = name
        self.__prompt_text = prompt
        self.__variables = self._extract_variables(prompt)

        try:
            # Register/sync with Bedrock for version management
            self.__bedrock_metadata = PromptManager().get_or_create_prompt(
                name=name, prompt_text=prompt
            )
            logger.info(f"Prompt '{name}' synced with Bedrock: {self.__bedrock_metadata.get('id')}")
        except Exception as e:
            logger.warning(f"Failed to sync prompt '{self.name}' with Bedrock: {e}")
            self.__bedrock_metadata = None

    @staticmethod
    def _extract_variables(prompt_text: str) -> list[str]:
        """Extract variable names from {{variable}} syntax."""
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, prompt_text)
        # Remove duplicates while preserving order
        seen = set()
        return [v for v in matches if not (v in seen or seen.add(v))]

    @property
    def prompt(self) -> str:
        """Return the actual prompt text (template with {{variables}})."""
        return self.__prompt_text

    @property
    def variables(self) -> list[str]:
        """Return list of variable names in this prompt."""
        return self.__variables

    @property
    def bedrock_metadata(self) -> dict | None:
        """Return Bedrock metadata if available."""
        return self.__bedrock_metadata

    def format(self, **kwargs) -> str:
        """
        Format the prompt by substituting {{variable}} placeholders with values.

        Args:
            **kwargs: Variable names and their values.

        Returns:
            The formatted prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = set(self.__variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        result = self.__prompt_text
        for var_name, value in kwargs.items():
            result = result.replace(f"{{{{{var_name}}}}}", str(value))
        return result

    def __str__(self) -> str:
        return self.prompt

    def __repr__(self) -> str:
        return self.__str__()


class PromptManager:
    """
    Manages prompts in AWS Bedrock using CHAT template type.

    Uses CHAT template type for proper system/user role separation:
    - System message: Contains the prompt instructions with XML tags
    - User message: Placeholder for runtime user input

    Operations:
    - Creates new prompts if they don't exist
    - Returns existing prompts if unchanged
    - Creates new versions if prompt content has changed
    """

    def __init__(self):
        self.bedrock_client = boto3.client(
            service_name='bedrock-agent',
            region_name=settings.AWS_REGION
        )

    def get_or_create_prompt(self, name: str, prompt_text: str, description: str = "") -> dict:
        """
        Get an existing prompt or create a new one.
        If the prompt exists but content has changed, creates a new version.

        Args:
            name: Unique name for the prompt
            prompt_text: The prompt template text (used as system message)
            description: Optional description for the prompt

        Returns:
            dict with prompt details (id, arn, version, name)
        """
        existing_prompt = self._find_prompt_by_name(name)

        if existing_prompt is None:
            # Prompt doesn't exist, create it
            logger.info(f"Prompt '{name}' not found. Creating new prompt.")
            return self._create_prompt(name, prompt_text, description)

        # Prompt exists, check if content has changed
        prompt_id = existing_prompt['id']
        current_content = self._get_prompt_content(prompt_id)

        if current_content != prompt_text:
            # Content changed, create new version
            logger.info(f"Prompt '{name}' content changed. Creating new version.")
            return self._create_new_version(prompt_id, prompt_text, description)

        # Content unchanged, return existing prompt with variables
        logger.info(f"Prompt '{name}' unchanged. Returning existing prompt.")
        input_variables = self.extract_variables(prompt_text)
        existing_prompt['variables'] = [v['name'] for v in input_variables]
        return existing_prompt

    @staticmethod
    def extract_variables(prompt_text: str) -> list[dict]:
        """
        Extract variables from prompt text using {{variable}} syntax.

        Returns:
            List of dicts with 'name' key for each variable found.
        """
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, prompt_text)
        # Remove duplicates while preserving order
        seen = set()
        unique_vars = []
        for var in matches:
            if var not in seen:
                seen.add(var)
                unique_vars.append({'name': var})
        return unique_vars

    def _build_chat_template_config(self, prompt_text: str) -> dict:
        """
        Build a CHAT template configuration for Bedrock Prompt Management.

        Structures the prompt as:
        - system: The prompt text with XML-tagged instructions
        - messages: A user message placeholder with {{user_input}}

        Args:
            prompt_text: The system prompt text.

        Returns:
            Template configuration dict for CHAT type.
        """
        # Extract variables from the system prompt text
        input_variables = self.extract_variables(prompt_text)

        # Add user_input variable for the user message placeholder
        var_names = {v['name'] for v in input_variables}
        if 'user_input' not in var_names:
            input_variables.append({'name': 'user_input'})

        chat_config = {
            'system': [{'text': prompt_text}],
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': '{{user_input}}'}],
                }
            ],
        }

        if input_variables:
            chat_config['inputVariables'] = input_variables

        return chat_config

    def _find_prompt_by_name(self, name: str) -> dict | None:
        """Find a prompt by its name."""
        try:
            paginator = self.bedrock_client.get_paginator('list_prompts')
            for page in paginator.paginate():
                for prompt in page.get('promptSummaries', []):
                    if prompt.get('name') == name:
                        return {
                            'id': prompt['id'],
                            'arn': prompt['arn'],
                            'version': prompt.get('version', 'DRAFT'),
                            'name': prompt['name']
                        }
        except ClientError as e:
            logger.error(f"Error listing prompts: {e}")
            raise
        return None

    def _get_prompt_content(self, prompt_id: str) -> str | None:
        """
        Get the current system prompt content from a Bedrock prompt.

        Handles both CHAT and TEXT template types for backwards compatibility.
        """
        try:
            response = self.bedrock_client.get_prompt(promptIdentifier=prompt_id)
            variants = response.get('variants', [])
            if variants:
                template_config = variants[0].get('templateConfiguration', {})

                # Try CHAT format first (new format with system/user separation)
                chat_config = template_config.get('chat', {})
                if chat_config:
                    system_messages = chat_config.get('system', [])
                    if system_messages:
                        return system_messages[0].get('text')

                # Fall back to TEXT format (legacy prompts)
                text_config = template_config.get('text', {})
                return text_config.get('text')
        except ClientError as e:
            logger.error(f"Error getting prompt content: {e}")
            raise
        return None

    def _create_prompt(self, name: str, prompt_text: str, description: str) -> dict:
        """Create a new prompt using CHAT template type."""
        try:
            chat_config = self._build_chat_template_config(prompt_text)
            input_variables = self.extract_variables(prompt_text)

            response = self.bedrock_client.create_prompt(
                name=name,
                description=description or f"Prompt: {name}",
                variants=[
                    {
                        'name': 'default',
                        'templateType': 'CHAT',
                        'templateConfiguration': {
                            'chat': chat_config,
                        },
                    }
                ],
                defaultVariant='default'
            )

            prompt_id = response['id']
            prompt_arn = response['arn']

            # Create an initial version to make the prompt usable
            version_response = self.bedrock_client.create_prompt_version(
                promptIdentifier=prompt_id,
                description=f"Initial version of {name}"
            )

            return {
                'id': prompt_id,
                'arn': prompt_arn,
                'version': version_response.get('version', 'DRAFT'),
                'name': name,
                'variables': [v['name'] for v in input_variables]
            }

        except ClientError as e:
            logger.error(f"Error creating prompt: {e}")
            raise

    def _create_new_version(self, prompt_id: str, prompt_text: str, description: str) -> dict:
        """Update prompt draft with CHAT template and create a new version."""
        chat_config = self._build_chat_template_config(prompt_text)
        input_variables = self.extract_variables(prompt_text)

        # First, update the DRAFT with new content
        try:
            update_response = self.bedrock_client.update_prompt(
                promptIdentifier=prompt_id,
                name=self._get_prompt_name(prompt_id),
                description=description or "Updated prompt",
                variants=[
                    {
                        'name': 'default',
                        'templateType': 'CHAT',
                        'templateConfiguration': {
                            'chat': chat_config,
                        },
                    }
                ],
                defaultVariant='default'
            )
        except ClientError as e:
            logger.error(f"Error updating prompt draft: {e}")
            raise

        # Then create a new version from the updated draft
        try:
            version_response = self.bedrock_client.create_prompt_version(
                promptIdentifier=prompt_id,
                description=description or "New version with updated content"
            )
        except ClientError as e:
            # Check if we hit the max versions limit
            if e.response.get('Error', {}).get('Code') == 'ValidationException' and 'max-number-versions-per-prompt' in str(e):
                logger.warning(f"Max version limit reached for prompt {prompt_id}. Deleting oldest version.")
                self._delete_oldest_version(prompt_id)
                # Retry creating the version
                version_response = self.bedrock_client.create_prompt_version(
                    promptIdentifier=prompt_id,
                    description=description or "New version with updated content"
                )
            else:
                logger.error(f"Error creating new version: {e}")
                raise

        return {
            'id': prompt_id,
            'arn': update_response['arn'],
            'version': version_response.get('version'),
            'name': update_response['name'],
            'variables': [v['name'] for v in input_variables]
        }

    def _list_prompt_versions(self, prompt_id: str) -> list[dict]:
        """List all versions of a prompt, sorted by version number."""
        try:
            versions = []
            paginator = self.bedrock_client.get_paginator('list_prompt_versions')
            for page in paginator.paginate(promptIdentifier=prompt_id):
                for version in page.get('promptSummaries', []):
                    # Skip DRAFT version
                    if version.get('version') != 'DRAFT':
                        versions.append({
                            'version': version.get('version'),
                            'arn': version.get('arn'),
                            'createdAt': version.get('createdAt'),
                        })
            # Sort by version number (ascending, so oldest first)
            versions.sort(key=lambda v: int(v['version']))
            return versions
        except ClientError as e:
            logger.error(f"Error listing prompt versions: {e}")
            raise

    def _delete_oldest_version(self, prompt_id: str) -> None:
        """Delete the oldest version of a prompt to make room for a new one."""
        versions = self._list_prompt_versions(prompt_id)
        if not versions:
            logger.warning(f"No versions found to delete for prompt {prompt_id}")
            return

        oldest_version = versions[0]['version']
        try:
            self.bedrock_client.delete_prompt(
                promptIdentifier=prompt_id,
                promptVersion=oldest_version
            )
            logger.info(f"Deleted oldest version {oldest_version} of prompt {prompt_id}")
        except ClientError as e:
            logger.error(f"Error deleting prompt version {oldest_version}: {e}")
            raise

    def _get_prompt_name(self, prompt_id: str) -> str:
        """Get the name of a prompt by its ID."""
        try:
            response = self.bedrock_client.get_prompt(promptIdentifier=prompt_id)
            return response['name']
        except ClientError as e:
            logger.error(f"Error getting prompt name: {e}")
            raise

    def get_prompt(self, name: str) -> dict | None:
        """Get a prompt by name without creating it."""
        return self._find_prompt_by_name(name)

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt by its ID."""
        try:
            self.bedrock_client.delete_prompt(promptIdentifier=prompt_id)
            logger.info(f"Prompt {prompt_id} deleted successfully.")
            return True
        except ClientError as e:
            logger.error(f"Error deleting prompt: {e}")
            raise
