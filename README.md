# 🎯 CareerCoach AI

**An agentic AI system that helps Sri Lankan IT undergraduates prepare for
technical interviews, get CV feedback, and receive grounded career advice.**

Built for **IT41043 — Intelligent Systems (Agentic AI)**, Horizon Campus.

**Live Demo:** [https://careercoach-ai-sithmini.streamlit.app/](https://careercoach-ai-sithmini.streamlit.app/)

**GitHub Repo:** [https://github.com/sithminisandamali/CareerCoach](https://github.com/sithminisandamali/CareerCoach)

## 1. Problem statement

Final-year IT undergraduates in Sri Lanka often prepare for interviews using
generic, non-localised resources that don't reflect the tech stacks (MERN,
Spring Boot), interview formats, or job-market realities specific to local
IT/BPM employers. **CareerCoach AI** grounds every answer in a curated
domain corpus — technical Q&A banks, CV guidance, and Sri Lankan IT
job-market notes — so the advice given is specific and actionable rather
than generic chatbot output. This directly avoids the "generic PDF Q&A bot"
trap the assignment brief warns against, since the corpus, question style,
and career context are all local and domain-specific.

## 2. Architecture

```
                          ┌───────────────────────┐
                          │          User            │
                          └───────────┬────────────┘
                                      │ message
                                      ▼
                          ┌───────────────────────┐
                          │      Router Agent        │   Groq · llama-3.1-8b-instant
                          │  classifies user intent   │
                          └───────────┬────────────┘
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
      interview_practice        cv_feedback          career_advice
                 │                    │                    │
                 ▼                    └──────────┬─────────┘
      ┌────────────────────┐                     ▼
      │   Interview Agent     │          ┌───────────────────────┐
      │  (ReAct + tool-use)    │          │  Career Advisor Agent    │
      │  1. retrieve()          │          │  (RAG-grounded, single-  │
      │  2. rerank()            │          │   turn)                   │
      │  3. synthesize Q&A      │          └───────────────────────┘
      └───────────┬────────┘
                  │ critique_request (structured A2A message)
                  ▼
      ┌────────────────────┐
      │    Critic Agent       │   OpenRouter · claude-3.5-sonnet
      │  reflection pattern    │
      └────────────────────┘
```

All retrieval (`retrieve`, `rerank`) reads from a FAISS index built over
`data/corpus/*.txt` at app startup.

## 3. Agentic design patterns used (≥3 required)

| # | Pattern | Implementation | File |
|---|---------|-----------------|------|
| 1 | **Router** | Cheap Groq model classifies each message into `interview_practice`, `cv_feedback`, or `career_advice` | `src/agents/router_agent.py` |
| 2 | **Orchestrator–Worker** | `run_orchestrator()` reads the router's decision and dispatches to the correct worker agent(s) | `src/agents/orchestrator.py` |
| 3 | **ReAct (Reason + Act) with tool-use** | Explicit *Thought → Action (calls retrieval tool) → Observation → Answer* loop | `src/agents/interview_agent.py` |
| 4 | **Reflection / self-critique** | Reviews the student's answer against the model answer and returns a structured critique + rewritten improved answer | `src/agents/critic_agent.py` |

## 4. Agent-to-agent communication

`InterviewAgent` and `CriticAgent` exchange structured JSON messages via a
lightweight custom protocol (inspired by the MCP/A2A message shape), built
in `src/agents/critic_agent.py::build_message()`:

```json
{
  "sender": "InterviewAgent",
  "receiver": "CriticAgent",
  "type": "critique_request",
  "payload": {
    "question": "...",
    "model_answer": "...",
    "user_answer": "..."
  }
}
```

### Sequence diagram

```
User            Orchestrator        RouterAgent       InterviewAgent        CriticAgent
 │  message           │                    │                  │                    │
 │──────────────────► │                    │                  │                    │
 │                    │──── classify ────► │                  │                    │
 │                    │ ◄─── category ───── │                  │                    │
 │                    │───────────── dispatch ─────────────► │                    │
 │                    │                    │      retrieve() → rerank()             │
 │                    │                    │      synthesize question + answer      │
 │                    │ ◄──────────────── result ───────────  │                    │
 │  submits answer     │                    │                  │                    │
 │──────────────────► │                    │                  │─ critique_request ►│
 │                    │                    │                  │                    │ critique
 │                    │ ◄──────────────────────── critique_response ────────────── │
 │ ◄──── feedback ──── │                    │                  │                    │
```

Every message exchanged is appended to an in-memory `log` list and rendered
live in the Streamlit sidebar — useful for walking a marker through the full
agent conversation during the viva.

## 5. Model selection strategy (≥2 models required)

| Sub-task | Model (provider) | Why chosen |
|---|---|---|
| Intent routing / classification | `llama-3.1-8b-instant` (Groq) | Extremely low latency, near-free per call; the task only needs 1-of-3 labels, so a small model's reasoning is more than sufficient |
| Retrieval re-ranking | `llama-3.3-70b-versatile` (Groq) | Needs slightly better judgment than routing to score passage relevance, but must score several candidates fast — Groq's inference speed prevents this from bottlenecking the pipeline |
| Interview question synthesis & critique/reflection | `anthropic/claude-3.5-sonnet` (OpenRouter) | The highest-reasoning-quality sub-tasks: generating a well-scoped interview question from retrieved context, and giving nuanced, structured feedback on free-text answers. Higher cost/latency is justified since output quality here directly determines the app's value |

All three model IDs are centralised in `src/models/llm_clients.py` so they
can be swapped or upgraded in one place — a good thing to point to live in
the viva when asked to make a small code change.


## 6. Secrets management

- Real keys live only in `.streamlit/secrets.toml` (gitignored) locally, or
  in Streamlit Cloud's Secrets manager in production — **never** in
  committed source code.
- `.streamlit/secrets.toml.example` documents the required format without
  real values.
- `.gitignore` excludes `secrets.toml`, `.env`, and the generated FAISS
  index/pickle files.


## 7. Repo & branching practice

One branch per feature, merged via Pull Requests with descriptive titles:

| Branch | Covers |
|---|---|
| `feature/rag-pipeline` | `src/rag/ingest.py`, `src/rag/retriever.py`, `data/corpus/` |
| `feature/agent-orchestration` | router, interview, critic agents, orchestrator |
| `feature/streamlit-ui` | `app.py` |
| `feature/model-router` | `src/models/llm_clients.py`, model selection table |


