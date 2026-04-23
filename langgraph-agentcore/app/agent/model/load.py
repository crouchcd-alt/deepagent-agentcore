from langchain_aws import ChatBedrockConverse

MODEL_ID = "us.amazon.nova-lite-v1:0"


def load_model() -> ChatBedrockConverse:
    """Get Bedrock model client using IAM credentials."""
    return ChatBedrockConverse(model=MODEL_ID)
