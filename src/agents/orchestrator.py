"""
orchestrator.py
Implements the ORCHESTRATOR-WORKER pattern. The orchestrator:
  1. Calls the Router Agent to classify intent.
  2. Dispatches to the right worker (currently: interview practice flow,
     which itself chains InterviewAgent -> CriticAgent via structured
     agent-to-agent messages).
  3. Returns a full transcript/log (used for the README sequence diagram
     and for the live viva walkthrough).
"""

from .router_agent import classify_query
from .interview_agent import run_interview_agent
from .critic_agent import run_critic_agent, build_message


def run_interview_flow(topic: str, user_answer: str = None) -> dict:
    log = []
    log.append({"agent": "Orchestrator", "step": "dispatch",
                "content": f"Routing interview_practice request for topic='{topic}' to InterviewAgent"})

    interview_result = run_interview_agent(topic, log)

    critique = None
    if user_answer:
        msg = build_message(
            sender="InterviewAgent", receiver="CriticAgent",
            msg_type="critique_request",
            payload={
                "question": interview_result["output"],
                "model_answer": interview_result["output"],
                "user_answer": user_answer,
            },
        )
        log.append({"agent": "InterviewAgent", "step": "sent_message", "content": msg})
        critique_msg = run_critic_agent(msg, log)
        critique = critique_msg["payload"]["critique"]

    return {"log": log, "interview": interview_result, "critique": critique}


def run_orchestrator(user_message: str, user_answer: str = None, cv_text: str = None) -> dict:
    category = classify_query(user_message)
    log = [{"agent": "Orchestrator", "step": "route_decision",
            "content": f"Router classified message as: {category}"}]

    if category == "interview_practice":
        result = run_interview_flow(user_message, user_answer)
        result["log"] = log + result["log"]
        result["category"] = category
        return result

    # cv_feedback / career_advice: simpler single-agent RAG-grounded response
    from ..rag.retriever import retrieve, rerank
    from ..models.llm_clients import call_openrouter, MODEL_DEEP

    candidates = retrieve(user_message, k=5)
    top_chunks = rerank(user_message, candidates, top_n=3)
    context = "\n---\n".join(c["text"] for c in top_chunks) if top_chunks else ""

    cv_section = f"\n\nStudent's CV content:\n{cv_text}\n" if cv_text else ""

    if context:
        prompt = (
            f"You are a career advisor for Sri Lankan IT undergraduates. "
            f"Use the context below if it's relevant, but you are not limited to it — "
            f"answer from your own knowledge too.\n\nContext:\n{context}"
            f"{cv_section}\n\nQuestion: {user_message}"
        )
    else:
        prompt = (
            f"You are a career advisor for Sri Lankan IT undergraduates. "
            f"Answer the student's question helpfully and concretely, using your own knowledge."
            f"{cv_section}\n\nQuestion: {user_message}"
        )

    answer = call_openrouter(MODEL_DEEP, [{"role": "user", "content": prompt}])
    log.append({"agent": "CareerAdvisorAgent", "step": "answer", "content": answer})

    return {"log": log, "category": category, "answer": answer, "context_used": top_chunks}