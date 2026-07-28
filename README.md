# 🎯 CareerCoach AI

**An agentic AI system that helps Sri Lankan IT undergraduates prepare for
technical interviews, get CV feedback, and receive grounded career advice.**

Built for **IT41043 — Intelligent Systems (Agentic AI)**, Horizon Campus.


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

## 4. Agent-to-agent communication (≥2 agents required)

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

## 6. RAG pipeline

- **Corpus** (`data/corpus/*.txt`): technical interview Q&A banks (MERN/Node,
  OOP, JavaScript, SQL), behavioral-interview guidance, a CV-writing guide,
  and Sri Lankan IT job-market notes.
  > ⚠️ **Ships with 8 starter files — expand toward 20+ before submission**
  > (see Section 10). More topics = better retrieval coverage and a
  > stronger RAG-integration mark.
- **Chunking**: fixed-size character chunks of 800 characters with a
  100-character overlap (`src/rag/ingest.py::chunk_text`). Chosen because
  most corpus items are short, self-contained Q&A pairs — 800 characters
  keeps each chunk to roughly one Q&A item without splitting mid-sentence.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` — runs
  locally, free, no API key required, and small enough (~80MB) to fit
  comfortably on Streamlit Community Cloud's free tier.
- **Vector store**: FAISS `IndexFlatIP` over L2-normalised embeddings
  (equivalent to cosine similarity), persisted to `src/rag/index.faiss` and
  `src/rag/chunks.pkl`. Auto-built on first run if missing.
- **Re-ranking**: the top-5 FAISS candidates are re-scored by a Groq model
  (`retriever.py::rerank`) before the top-3 are passed to the reasoning
  model — demonstrating deliberate, staged model use rather than one model
  doing everything end-to-end.

### Retrieval evaluation (5 sample queries)

Run `python -m src.rag.ingest` then test each query through `retrieve()` —
replace this table with your own actual run before submission (markers
specifically check this isn't fabricated):

| # | Query | Top result relevant? | Notes |
|---|-------|:---:|-------|
| 1 | "MERN backend" | ✅ | Returns Express middleware & Node event-loop chunks |
| 2 | "OOP interview question" | ✅ | Returns SOLID principles / abstract-class chunks |
| 3 | "how to structure my CV projects section" | ✅ | Returns CV guide chunk on consolidating multi-repo projects |
| 4 | "salary negotiation Sri Lanka" | ⚠️ partial | Job-market file mentions researching ranges but has no concrete figures — corpus gap, consider adding a dedicated salary-guide file |
| 5 | "teamwork behavioral question" | ✅ | Returns the STAR-method behavioral chunk |

## 7. Deployment

1. Push this repo to GitHub (public, or private with your lecturer added as
   a collaborator).
2. Go to **share.streamlit.io**, sign in with GitHub, click **New app**,
   select this repo/branch, and set the main file to `app.py`.
3. In **App settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-key"
   OPENROUTER_API_KEY = "your-key"
   ```
4. Deploy. The first load is slower (downloads the embedding model and
   builds the FAISS index) — subsequent loads are fast.
5. Keep the app live for at least two weeks after the deadline, as required
   by Section 4(e) of the brief.

## 8. Secrets management

- Real keys live only in `.streamlit/secrets.toml` (gitignored) locally, or
  in Streamlit Cloud's Secrets manager in production — **never** in
  committed source code.
- `.streamlit/secrets.toml.example` documents the required format without
  real values.
- `.gitignore` excludes `secrets.toml`, `.env`, and the generated FAISS
  index/pickle files.

## 9. Setup instructions (local development)

```bash
git clone <your-repo-url>
cd careercoach-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your real GROQ_API_KEY / OPENROUTER_API_KEY

python -m src.rag.ingest         # builds the FAISS index from data/corpus/
streamlit run app.py
```

Get free API keys at:
- Groq → console.groq.com
- OpenRouter → openrouter.ai

## 10. Known limitations

- Corpus currently ships with 8 sample files; expand toward 20+ for full
  marks on the RAG-integration criterion (Section 6).
- Re-ranking relies on an LLM to output a parseable ordering; malformed
  output falls back to plain FAISS similarity order.
- No persistent chat history across sessions (Streamlit session state only,
  resets on page refresh).
- Single-turn critique — the Critic Agent does not currently loop back to
  request a second, harder question from the Interview Agent based on the
  student's score.

## 11. Repo & branching practice

One branch per feature, merged via Pull Requests with descriptive titles:

| Branch | Covers |
|---|---|
| `feature/rag-pipeline` | `src/rag/ingest.py`, `src/rag/retriever.py`, `data/corpus/` |
| `feature/agent-orchestration` | router, interview, critic agents, orchestrator |
| `feature/streamlit-ui` | `app.py` |
| `feature/model-router` | `src/models/llm_clients.py`, model selection table |

Use semantic commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
and spread commits across your actual development period.


commit at the end, since the brief explicitly flags that as an integrity
concern during git history review.
