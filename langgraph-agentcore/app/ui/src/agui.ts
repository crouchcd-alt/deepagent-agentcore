// Minimal AG-UI client: POSTs RunAgentInput, parses SSE, yields typed events.
// Avoids extra deps so the React app stays small.

export type AGUIEvent =
  | { type: "RUN_STARTED"; threadId: string; runId: string }
  | { type: "RUN_FINISHED"; threadId: string; runId: string }
  | { type: "RUN_ERROR"; message: string }
  | { type: "STEP_STARTED"; stepName: string }
  | { type: "STEP_FINISHED"; stepName: string }
  | { type: "TEXT_MESSAGE_START"; messageId: string; role: "assistant" }
  | { type: "TEXT_MESSAGE_CONTENT"; messageId: string; delta: string }
  | { type: "TEXT_MESSAGE_END"; messageId: string }
  | { type: "TOOL_CALL_START"; toolCallId: string; toolCallName: string; parentMessageId?: string }
  | { type: "TOOL_CALL_ARGS"; toolCallId: string; delta: string }
  | { type: "TOOL_CALL_END"; toolCallId: string }
  | { type: "TOOL_CALL_RESULT"; toolCallId: string; messageId: string; content: string }
  | { type: "STATE_SNAPSHOT"; snapshot: unknown }
  | { type: "STATE_DELTA"; delta: unknown }
  | { type: string; [k: string]: unknown };

export interface AGUIMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
}

export interface RunAgentInput {
  threadId: string;
  runId: string;
  state: Record<string, unknown>;
  messages: AGUIMessage[];
  tools: unknown[];
  context: unknown[];
  forwardedProps?: Record<string, unknown>;
}

export async function* runAgent(
  url: string,
  input: RunAgentInput,
  signal?: AbortSignal,
): AsyncGenerator<AGUIEvent> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": input.threadId,
    },
    body: JSON.stringify({
      thread_id: input.threadId,
      run_id: input.runId,
      state: input.state,
      messages: input.messages,
      tools: input.tools,
      context: input.context,
      forwarded_props: input.forwardedProps ?? {},
    }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}: ${await res.text().catch(() => "")}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      // A frame may have multiple `data:` lines — concatenate per spec.
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length === 0) continue;
      try {
        yield JSON.parse(dataLines.join("\n")) as AGUIEvent;
      } catch (err) {
        console.warn("Failed to parse AG-UI event", err, dataLines);
      }
    }
  }
}
