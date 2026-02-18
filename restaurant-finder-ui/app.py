import json
import uuid
import os

import aiohttp
import chainlit as cl
from chainlit.input_widget import TextInput


# --- Connection mode configuration ---
# Set AGENT_CONNECTION_MODE to "aws" to invoke the deployed AgentCore Runtime on AWS.
# Set to "local" (or leave unset) to call the local API server.
AGENT_CONNECTION_MODE = os.environ.get("AGENT_CONNECTION_MODE", "local").lower()

# Local mode settings
AGENTCORE_API_URL = os.environ.get("AGENTCORE_API_URL", "http://localhost:8080/invocations")

# AWS mode settings
AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Lazily initialized boto3 client for AWS mode
_agentcore_client = None


def _get_agentcore_client():
    """Get or create the boto3 bedrock-agentcore client (lazy init)."""
    global _agentcore_client
    if _agentcore_client is None:
        import boto3
        _agentcore_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    return _agentcore_client


@cl.on_settings_update
async def settings_update(settings):
    """Handle settings updates."""
    cl.user_session.set("customer_name", settings.get("customer_name", "Guest"))

    await cl.Message(
        content=f"Settings updated! Welcome, {settings.get('customer_name', 'Guest')}! Ready to find your perfect restaurant."
    ).send()


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with settings."""
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="customer_name",
                label="Your Name",
                placeholder="Enter your name",
                initial="Guest"
            ),
        ]
    ).send()

    cl.user_session.set("customer_name", settings.get("customer_name", "Guest"))

    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)

    customer_name = settings.get("customer_name", "Guest")
    await cl.Message(
        content=f"Welcome, {customer_name}! I'm your restaurant finder assistant. What kind of dining experience are you looking for today?\n\n*Tip: Click the settings icon to update your profile.*"
    ).send()


async def _local_sse_lines(payload):
    """Yield SSE lines from the local HTTP API."""
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            AGENTCORE_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            line_buffer = ""
            async for chunk_bytes in response.content.iter_any():
                line_buffer += chunk_bytes.decode("utf-8", errors="replace")
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    yield line.strip()


async def _aws_sse_lines(payload, conversation_id):
    """Yield SSE lines from the AWS Bedrock AgentCore Runtime."""
    import asyncio

    client = _get_agentcore_client()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            qualifier="DEFAULT",
            runtimeSessionId=conversation_id,
            payload=json.dumps(payload),
        ),
    )

    content_type = response.get("contentType", "")
    if "text/event-stream" in content_type:
        for line in response["response"].iter_lines(chunk_size=1):
            if line:
                yield line.decode("utf-8")
    else:
        for event in response.get("response", []):
            chunk = event.decode("utf-8") if isinstance(event, bytes) else str(event)
            yield f'data: {json.dumps({"chunk": chunk})}'


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages by calling the AgentCore API (local or AWS)."""
    customer_name = cl.user_session.get("customer_name", "Guest")
    conversation_id = cl.user_session.get("conversation_id")

    msg = cl.Message(content="")
    await msg.send()

    await _invoke_agent(msg, message.content, customer_name, conversation_id)


async def _invoke_agent(
    msg: cl.Message,
    user_input: str,
    customer_name: str,
    conversation_id: str,
):
    """Invoke the agent via local API or AWS AgentCore Runtime based on config."""
    if AGENT_CONNECTION_MODE == "aws" and not AGENT_RUNTIME_ARN:
        msg.content = "Configuration Error: AGENT_RUNTIME_ARN is required when AGENT_CONNECTION_MODE=aws."
        await msg.update()
        return

    payload = {
        "prompt": user_input,
        "customer_name": customer_name,
        "conversation_id": conversation_id,
    }

    full_response = ""

    try:
        if AGENT_CONNECTION_MODE == "aws":
            lines = _aws_sse_lines(payload, conversation_id)
        else:
            lines = _local_sse_lines(payload)

        async for line in lines:
            if not line or not line.startswith("data: "):
                continue

            json_str = line[6:]
            try:
                data = json.loads(json_str)

                # Handle nested SSE format
                if isinstance(data, str) and data.startswith("data: "):
                    data = json.loads(data[6:].strip())

                if isinstance(data, dict):
                    if "chunk" in data:
                        chunk = data["chunk"]
                        await msg.stream_token(chunk)
                        full_response += chunk

                    elif "error" in data:
                        msg.content = f"Error: {data['error']}"
                        await msg.update()
                        return

            except json.JSONDecodeError:
                continue

        msg.content = full_response if full_response else "No response received."
        await msg.update()

    except aiohttp.ClientResponseError as e:
        msg.content = f"API Error: {e.status}"
        await msg.update()
    except aiohttp.ClientError:
        msg.content = "Connection Error: Could not connect to the local API. Please ensure the API server is running."
        await msg.update()
    except Exception as e:
        if AGENT_CONNECTION_MODE == "aws":
            error_name = type(e).__name__
            msg.content = f"AWS Runtime Error ({error_name}): {str(e)}"
        else:
            msg.content = "An unexpected error occurred. Please try again."
        await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
