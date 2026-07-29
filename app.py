"""
app.py - Streamlit UI for CareerCoach AI
Run locally with: streamlit run app.py
"""

import streamlit as st
from src.agents.orchestrator import run_orchestrator
from src.utils.file_extract import extract_text_from_upload

st.set_page_config(page_title="CareerCoach AI", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .hero {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        color: white;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        background-color: #1E1E2E;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #8B7FFF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: transform 0.1s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        opacity: 0.92;
    }

    .answer-card {
        background-color: rgba(108, 99, 255, 0.08);
        border-left: 4px solid #6C63FF;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
    }

    .log-entry {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #8B7FFF;
    }
    .log-agent {
        font-weight: 700;
        color: #8B7FFF;
    }
    .log-step {
        font-size: 0.8rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎯 CareerCoach AI</h1>
    <p>An agentic AI assistant that helps Sri Lankan IT undergraduates prepare
    for technical interviews, improve their CVs, and get grounded career advice.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Router Agent** (Groq, cheap/fast) classifies your message.
        2. If it's interview practice, the **Interview Agent** (ReAct + RAG
           tool-use) retrieves relevant material and asks a question.
        3. Submit your answer and the **Critic Agent** (reflection pattern)
           gives structured feedback.
        4. For CV / career questions, a RAG-grounded advisor answers directly.
        """
    )
    show_log = st.checkbox("Show agent message log (for demo/viva)", value=True)


def render_log(log: list):
    """Renders the agent trace with content visible immediately — no clicking to expand."""
    st.subheader("🔍 Agent message log")
    for entry in log:
        content = entry.get("content")
        st.markdown(
            f"""<div class="log-entry">
                <span class="log-agent">🤖 {entry.get('agent', '?')}</span>
                &nbsp;·&nbsp;<span class="log-step">{entry.get('step', '')}</span>
            </div>""",
            unsafe_allow_html=True,
        )
        if isinstance(content, dict):
            st.json(content, expanded=True)
        else:
            st.write(content)


tab1, tab2 = st.tabs(["💬 Interview Practice", "📄 CV / Career Advice"])

# ---------------------------------------------------------------------------
# Tab 1: Interview Practice
# ---------------------------------------------------------------------------
with tab1:
    topic = st.text_input("Topic to practice (e.g. 'MERN backend', 'OOP', 'teamwork behavioral')")
    if st.button("Generate question", key="gen_q"):
        if topic.strip():
            with st.spinner("Routing and retrieving context..."):
                result = run_orchestrator(topic)
            st.session_state["last_result"] = result
        else:
            st.warning("Enter a topic first.")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        if result.get("category") == "interview_practice":
            st.subheader("❓ Question & model answer")
            st.markdown(f'<div class="answer-card">{result["interview"]["output"]}</div>', unsafe_allow_html=True)

            user_answer = st.text_area("Your answer (optional — submit for feedback)")
            if st.button("Get feedback", key="get_fb"):
                with st.spinner("Critic agent reviewing your answer..."):
                    result2 = run_orchestrator(topic, user_answer=user_answer)
                st.session_state["last_result"] = result2
                st.rerun()

            if result.get("critique"):
                st.subheader("📝 Feedback")
                st.markdown(f'<div class="answer-card">{result["critique"]}</div>', unsafe_allow_html=True)

        if show_log:
            render_log(result["log"])

# ---------------------------------------------------------------------------
# Tab 2: CV / Career Advice
# ---------------------------------------------------------------------------
with tab2:
    uploaded_cv = st.file_uploader("Upload your CV (optional — PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    cv_text = ""
    if uploaded_cv is not None:
        with st.spinner("Reading your CV..."):
            cv_text = extract_text_from_upload(uploaded_cv)
        if cv_text.startswith("[Could not read") or cv_text.startswith("[Unsupported"):
            st.error(cv_text)
            cv_text = ""
        else:
            st.success(f"✅ Loaded {uploaded_cv.name} ({len(cv_text)} characters)")
            with st.expander("Preview extracted text"):
                st.text(cv_text[:1500] + ("..." if len(cv_text) > 1500 else ""))

    question = st.text_area(
        "Ask about your CV or career path",
        placeholder="e.g. 'How can I improve my CV for a frontend role?' or 'What skills am I missing for a MERN job?'"
    )

    if st.button("Ask advisor"):
        if question.strip():
            with st.spinner("Retrieving context and generating advice..."):
                result = run_orchestrator(question, cv_text=cv_text if cv_text else None)

            st.subheader("💡 Answer")
            st.markdown(f'<div class="answer-card">{result.get("answer", "No answer generated.")}</div>', unsafe_allow_html=True)

            if result.get("context_used"):
                with st.expander("📚 Sources used (RAG context)"):
                    for c in result["context_used"]:
                        st.markdown(f"**{c['source']}** (score: {c['score']:.2f})")
                        st.caption(c["text"][:300] + "...")

            if show_log:
                render_log(result["log"])
        else:
            st.warning("Type a question first.")

st.divider()
st.caption("Agentic AI Assignment")