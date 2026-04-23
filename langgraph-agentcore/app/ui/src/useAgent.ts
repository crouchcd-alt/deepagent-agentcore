import { useCallback, useEffect, useRef, useState } from "react";
import { runAgent, type AGUIEvent, type AGUIMessage } from "./agui";

const ENDPOINT = import.meta.env.VITE_AGENT_URL ?? "http://localhost:8080/invocations";
const THREAD_KEY = "agentcore.threadId";

export type ChatItem =
  | { kind: "message"; id: string; role: "user" | "assistant"; content: string }
  | { kind: "tool"; id: string; name: string; args: string; result?: string };

function ensureThreadId(): string {
  const existing = localStorage.getItem(THREAD_KEY);
  if (existing) return existing;
  // AgentCore session id requires length >= 33.
  const id = `ui-${crypto.randomUUID()}-${crypto.randomUUID().slice(0, 8)}`;
  localStorage.setItem(THREAD_KEY, id);
  return id;
}

export function useAgent() {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [status, setStatus] = useState<string>("");
  const [running, setRunning] = useState(false);
  const threadIdRef = useRef<string>("");

  useEffect(() => {
    threadIdRef.current = ensureThreadId();
  }, []);

  const reset = useCallback(() => {
    localStorage.removeItem(THREAD_KEY);
    threadIdRef.current = ensureThreadId();
    setItems([]);
    setStatus("");
  }, []);

  const send = useCallback(async (prompt: string) => {
    if (running || !prompt.trim()) return;

    const userId = crypto.randomUUID();
    const userMsg: ChatItem = { kind: "message", id: userId, role: "user", content: prompt };
    setItems((prev) => [...prev, userMsg]);
    setRunning(true);
    setStatus("starting…");

    const messagesForRun: AGUIMessage[] = [{ id: userId, role: "user", content: prompt }];

    try {
      const stream = runAgent(ENDPOINT, {
        threadId: threadIdRef.current,
        runId: crypto.randomUUID(),
        state: {},
        messages: messagesForRun,
        tools: [],
        context: [],
      });

      for await (const ev of stream) {
        applyEvent(ev, setItems, setStatus);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`error: ${msg}`);
    } finally {
      setRunning(false);
      setStatus((s) => (s.startsWith("error") ? s : ""));
    }
  }, [running]);

  return { items, status, running, send, reset, threadId: threadIdRef.current };
}

function applyEvent(
  ev: AGUIEvent,
  setItems: React.Dispatch<React.SetStateAction<ChatItem[]>>,
  setStatus: React.Dispatch<React.SetStateAction<string>>,
) {
  switch (ev.type) {
    case "RUN_STARTED":
      setStatus("running…");
      return;
    case "STEP_STARTED":
      setStatus(`step: ${(ev as any).stepName ?? ""}`);
      return;
    case "STEP_FINISHED":
      setStatus("");
      return;
    case "TEXT_MESSAGE_START": {
      const e = ev as Extract<AGUIEvent, { type: "TEXT_MESSAGE_START" }>;
      setItems((prev) =>
        prev.some((it) => it.kind === "message" && it.id === e.messageId)
          ? prev
          : [...prev, { kind: "message", id: e.messageId, role: "assistant", content: "" }],
      );
      return;
    }
    case "TEXT_MESSAGE_CONTENT": {
      const e = ev as Extract<AGUIEvent, { type: "TEXT_MESSAGE_CONTENT" }>;
      setItems((prev) => {
        const exists = prev.some((it) => it.kind === "message" && it.id === e.messageId);
        if (!exists) {
          return [
            ...prev,
            { kind: "message", id: e.messageId, role: "assistant", content: e.delta ?? "" },
          ];
        }
        return prev.map((it) =>
          it.kind === "message" && it.id === e.messageId
            ? { ...it, content: it.content + (e.delta ?? "") }
            : it,
        );
      });
      return;
    }
    case "TOOL_CALL_START": {
      const e = ev as Extract<AGUIEvent, { type: "TOOL_CALL_START" }>;
      setItems((prev) => [
        ...prev,
        { kind: "tool", id: e.toolCallId, name: e.toolCallName, args: "" },
      ]);
      return;
    }
    case "TOOL_CALL_ARGS": {
      const e = ev as Extract<AGUIEvent, { type: "TOOL_CALL_ARGS" }>;
      setItems((prev) =>
        prev.map((it) =>
          it.kind === "tool" && it.id === e.toolCallId
            ? { ...it, args: it.args + (e.delta ?? "") }
            : it,
        ),
      );
      return;
    }
    case "TOOL_CALL_RESULT": {
      const e = ev as Extract<AGUIEvent, { type: "TOOL_CALL_RESULT" }>;
      setItems((prev) =>
        prev.map((it) =>
          it.kind === "tool" && it.id === e.toolCallId
            ? { ...it, result: e.content }
            : it,
        ),
      );
      return;
    }
    case "RUN_ERROR": {
      const e = ev as Extract<AGUIEvent, { type: "RUN_ERROR" }>;
      setStatus(`error: ${e.message}`);
      return;
    }
    case "RUN_FINISHED":
      setStatus("");
      return;
    default:
      return;
  }
}
