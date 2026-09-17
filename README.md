# CareAI — Phase 1: Plain Python

Home-monitoring agent for patients. This is the first version of the
project: no frameworks, so you understand what a framework hides from you
before you start using one.

## What this version does

- Talks to the patient to collect a daily check-in (text)
- Uses the **OpenAI SDK's native function calling** (no LangChain) to
  extract the data in a structured way
- Applies **manual alert rules** (`rules.py`) on top of the extracted data
- Saves every check-in to SQLite (`storage.py`)
- Keeps the conversation history by hand, in a plain list (no framework
  memory abstraction)

## How to run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python main.py
```

## Structure

```
cuidaia/
├── main.py         # CLI that ties everything together
├── agent.py         # system prompt, tool definition, OpenAI API call
├── models.py         # CheckIn and VitalSigns (dataclasses)
├── rules.py          # alert rules engine
├── storage.py         # SQLite persistence
└── requirements.txt
```

## What you learned in this phase

- Manual prompt engineering (system prompt with explicit role and limits)
- Native function calling (`tools` + parsing `tool_calls`)
- Manual conversation history handling (no "memory" abstraction)
- Simple persistence with raw SQL

## Known limitations (to be solved in later phases)

- No RAG: the agent can't answer questions about care protocols while
  citing sources → **Phase 2 (LangChain)**
- The flow is linear (conversation → tool call), no explicit decision
  branches (e.g. escalate to emergency vs. normal notification) →
  **Phase 3 (LangGraph)**
- The "tools" (saving to the DB, notifying) are all hardcoded in the same
  process → **Phase 3.5 (MCP)**

## Next step

Phase 2: rebuild this with LangChain — real conversational memory, RAG
over care protocols, and structured output parsing with Pydantic.
# ai-engineering
