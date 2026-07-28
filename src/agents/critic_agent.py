"""
critic_agent.py
Implements the REFLECTION / SELF-CRITIQUE pattern.

This agent receives a structured message FROM the Interview Agent (via the
orchestrator) containing the question + model answer, plus the user's own
attempted answer, and produces a structured critique: score, strengths,
gaps, and a rewritten improved answer.

Agent-to-agent protocol: a plain dict message, inspired by A2A/MCP-style
structured messages:
    {
      "sender": "InterviewAgent",
      "receiver": "CriticAgent",
      "type": "critique_request",
      "payload": {"question": ..., "model_answer": ..., "user_answer": ...}
    }
"""

from ..models.llm_clients import call_openrouter, MODEL_DEEP


def build_message(sender: str, receiver: str, msg_type: str, payload: dict) -> dict:
    return {"sender": sender, "receiver": receiver, "type": msg_type, "payload": payload}


def run_critic_agent(message: dict, log: list) -> dict:
    assert message["type"] == "critique_request"
    payload = message["payload"]
    question = payload["question"]
    model_answer = payload["model_answer"]
    user_answer = payload["user_answer"]

    log.append({"agent": "CriticAgent", "step": "received_message", "content": message})

    prompt = (
        "You are a strict but encouraging interview coach reviewing a student's "
        "answer.\n\n"
        f"Question: {question}\n\n"
        f"Reference model answer: {model_answer}\n\n"
        f"Student's answer: {user_answer}\n\n"
        "Give a structured critique in this format:\n"
        "SCORE (0-10): ...\n"
        "STRENGTHS: ...\n"
        "GAPS: ...\n"
        "IMPROVED ANSWER: ..."
    )
    messages = [{"role": "user", "content": prompt}]
    critique = call_openrouter(MODEL_DEEP, messages)

    response_message = build_message(
        sender="CriticAgent", receiver="Orchestrator",
        msg_type="critique_response", payload={"critique": critique},
    )
    log.append({"agent": "CriticAgent", "step": "sent_message", "content": response_message})

    return response_message
