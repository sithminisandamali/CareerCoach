"""
interview_agent.py
Implements the REACT pattern (explicit Thought -> Action -> Observation ->
Answer loop) combined with TOOL-USE (the retrieval tool over our RAG index).

The agent:
  1. THOUGHT: decides it needs domain context before answering.
  2. ACTION: calls the retrieval tool (rag.retriever.retrieve + rerank).
  3. OBSERVATION: reads back the retrieved chunks.
  4. ANSWER: synthesises a question + model answer/tips using the retrieved
     context, via the OpenRouter deep-reasoning model.
"""

from ..models.llm_clients import call_openrouter, MODEL_DEEP
from ..rag.retriever import retrieve, rerank


def run_interview_agent(topic: str, log: list) -> dict:
    """topic: e.g. 'MERN backend', 'OOP', 'behavioral - teamwork'."""

    # THOUGHT
    log.append({"agent": "InterviewAgent", "step": "thought",
                "content": f"I need reference material on '{topic}' before I can ask a good question."})

    # ACTION (tool use: retrieval)
    log.append({"agent": "InterviewAgent", "step": "action",
                "content": f"retrieve(query='{topic}', k=5) then rerank(top_n=3)"})
    candidates = retrieve(topic, k=5)
    top_chunks = rerank(topic, candidates, top_n=3)

    # OBSERVATION
    context = "\n---\n".join(c["text"] for c in top_chunks)
    log.append({"agent": "InterviewAgent", "step": "observation",
                "content": f"Retrieved {len(top_chunks)} relevant chunks from corpus."})

    # ANSWER (deep reasoning model synthesises the actual question + ideal answer)
    prompt = (
        f"You are an interview coach for Sri Lankan IT undergraduates. "
        f"Using ONLY the context below, write ONE realistic interview question "
        f"about '{topic}', followed by a concise model answer (max 120 words).\n\n"
        f"Context:\n{context}\n\n"
        f"Format:\nQUESTION: ...\nMODEL ANSWER: ..."
    )
    messages = [{"role": "user", "content": prompt}]
    result = call_openrouter(MODEL_DEEP, messages)

    log.append({"agent": "InterviewAgent", "step": "answer", "content": result})

    return {"topic": topic, "output": result, "context_used": top_chunks}
