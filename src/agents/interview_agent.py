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
    context = "\n---\n".join(c["text"] for c in top_chunks) if top_chunks else ""
    log.append({"agent": "InterviewAgent", "step": "observation",
                "content": f"Retrieved {len(top_chunks)} relevant chunks from corpus."})

    # ANSWER (deep reasoning model synthesises the actual question + ideal answer)
    if context:
        prompt = (
            f"You are an interview coach for Sri Lankan IT undergraduates. "
            f"Use the context below if it's relevant, but you are not limited to it — "
            f"draw on your own knowledge too. Write ONE realistic interview question "
            f"about '{topic}', followed by a concise model answer (max 120 words).\n\n"
            f"Context:\n{context}\n\n"
            f"Format:\nQUESTION: ...\nMODEL ANSWER: ..."
        )
    else:
        prompt = (
            f"You are an interview coach for Sri Lankan IT undergraduates. "
            f"Write ONE realistic interview question about '{topic}', followed by "
            f"a concise model answer (max 120 words), using your own knowledge.\n\n"
            f"Format:\nQUESTION: ...\nMODEL ANSWER: ..."
        )

    messages = [{"role": "user", "content": prompt}]
    result = call_openrouter(MODEL_DEEP, messages)

    log.append({"agent": "InterviewAgent", "step": "answer", "content": result})

    return {"topic": topic, "output": result, "context_used": top_chunks}