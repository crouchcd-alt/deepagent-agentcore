# AgentCore UI

Minimal React + Vite chat client that streams AG-UI events from the LangGraph
AgentCore app in `app/agent`.

## Run

```sh
# 1. start the agent (separate terminal)
cd ../agent && uv run python main.py

# 2. install + start the UI
cd ../ui
npm install
npm run dev
```

Open http://localhost:5173. The UI POSTs `RunAgentInput` to
`http://localhost:8080/invocations` and reduces the SSE event stream into:

- text messages (token-streamed via `TEXT_MESSAGE_CONTENT`)
- tool-call panels (live `TOOL_CALL_ARGS` + `TOOL_CALL_RESULT`)
- a status line (driven by `STEP_STARTED`/`STEP_FINISHED`)

The `threadId` is persisted to `localStorage` so the AgentCore memory
checkpointer threads conversations across page refreshes. Use the **new thread**
button to start fresh.

Override the endpoint at build/dev time via:

```sh
VITE_AGENT_URL=http://other-host:8080/invocations npm run dev
```
