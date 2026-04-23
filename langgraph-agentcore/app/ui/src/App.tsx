import { useState } from "react";
import { useAgent } from "./useAgent";

export function App() {
  const { items, status, running, send, reset, threadId } = useAgent();
  const [draft, setDraft] = useState("");

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft;
    setDraft("");
    await send(text);
  };

  return (
    <div className="app">
      <h1>
        AgentCore Chat <span style={{ opacity: 0.4, fontWeight: 400, fontSize: 12 }}>· thread {threadId.slice(0, 12)}…</span>
        <button
          type="button"
          onClick={reset}
          style={{ float: "right", padding: "4px 10px", fontSize: 12 }}
          disabled={running}
        >
          new thread
        </button>
      </h1>

      <div className="status">{status}</div>

      <div className="feed">
        {items.map((it) =>
          it.kind === "message" ? (
            <div key={it.id} className={`msg ${it.role}`}>
              <div className="role">{it.role}</div>
              {it.content || (it.role === "assistant" && running ? "…" : "")}
            </div>
          ) : (
            <div key={it.id} className="tool">
              <div>
                🔧 <strong>{it.name}</strong>(<span className="args">{it.args}</span>)
              </div>
              {it.result !== undefined && <div className="result">→ {it.result}</div>}
            </div>
          ),
        )}
      </div>

      <form onSubmit={onSubmit}>
        <div>
          <input
            type="text"
            placeholder={running ? "thinking…" : "ask the agent (e.g. 12 * 9 + 4)"}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={running}
            autoFocus
          />
          <button type="submit" disabled={running || !draft.trim()}>
            send
          </button>
        </div>
      </form>
    </div>
  );
}
