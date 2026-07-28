"""
app.py - Streamlit UI for CareerCoach AI
Run locally with: streamlit run app.py
"""

import streamlit as st
from src.agents.orchestrator import run_orchestrator

st.set_page_config(page_title="CareerCoach AI", page_icon="🎯", layout="wide")

st.title("🎯 CareerCoach AI")
st.caption(
    "An agentic AI assistant that helps Sri Lankan IT undergraduates prepare "
    "for technical interviews, improve their CVs, and get grounded career advice."
)

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

tab1, tab2 = st.tabs(["💬 Interview Practice", "📄 CV / Career Advice"])

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
            st.subheader("Question & model answer")
            st.write(result["interview"]["output"])

            user_answer = st.text_area("Your answer (optional - submit for feedback)")
            if st.button("Get feedback", key="get_fb"):
                with st.spinner("Critic agent reviewing your answer..."):
                    result2 = run_orchestrator(topic, user_answer=user_answer)
                st.session_state["last_result"] = result2
                st.rerun()

            if result.get("critique"):
                st.subheader("Feedback")
                st.write(result["critique"])

        if show_log:
            st.subheader("Agent message log")
            for entry in result["log"]:
                st.json(entry, expanded=False)

with tab2:
    question = st.text_area("Ask about your CV or career path")
    if st.button("Ask advisor"):
        if question.strip():
            with st.spinner("Retrieving context and generating advice..."):
                result = run_orchestrator(question)
            st.subheader("Answer")
            st.write(result.get("answer", "No answer generated."))

            with st.expander("Sources used (RAG context)"):
                for c in result.get("context_used", []):
                    st.markdown(f"**{c['source']}** (score: {c['score']:.2f})")
                    st.caption(c["text"][:300] + "...")

            if show_log:
                st.subheader("Agent message log")
                for entry in result["log"]:
                    st.json(entry, expanded=False)
        else:
            st.warning("Type a question first.")

st.divider()
st.caption("IT41043 Agentic AI Assignment — Horizon Campus")
