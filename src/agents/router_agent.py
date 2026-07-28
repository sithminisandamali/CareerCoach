"""
router_agent.py
Implements the ROUTER pattern: a cheap/fast model classifies the user's
intent so the orchestrator can dispatch to the right worker agent.
"""

from ..models.llm_clients import call_groq, MODEL_ROUTER

CATEGORIES = ["interview_practice", "cv_feedback", "career_advice"]

SYSTEM_PROMPT = (
    "You are an intent classifier for a career-coaching assistant used by "
    "Sri Lankan IT undergraduates. Classify the user's message into exactly "
    "ONE of these categories: interview_practice, cv_feedback, career_advice. "
    "Reply with ONLY the category label, nothing else."
)


def classify_query(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    result = call_groq(MODEL_ROUTER, messages, temperature=0.0, max_tokens=10)
    result = result.strip().lower()
    for cat in CATEGORIES:
        if cat in result:
            return cat
    return "career_advice"  # safe default
