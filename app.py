import os
import json
import streamlit as st

from config import DEFAULT_MODEL, TEMPERATURE
from agents.llm_client import LLMClient
from utils.router import build_prompt, route_query
from utils.api_mode import build_api_response

# ---- Logging ----
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "app.log"
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)

    logger = logging.getLogger("f1_app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


logger = setup_logger()

st.set_page_config(page_title="F1 Season/GP Chat", layout="wide")
st.title("🏎️ Formula 1 — Season & GP Chat (Mistral)")

# ---- Sidebar settings ----
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Mistral model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(TEMPERATURE), step=0.05)
    api_only = st.toggle("API-only (return JSON facts, no LLM)", value=False)
    st.caption("Set your MISTRAL_API_KEY in environment or .env before running.")

# ---- Multi-session chat state ----
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Default": [{"role": "system", "content": "You are a concise Formula 1 analyst."}]}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Default"

colA, colB = st.columns([3,2])
with colA:
    session_name = st.text_input("Current session name", value=st.session_state.current_session)
    if session_name != st.session_state.current_session:
        # rename session
        st.session_state.sessions[session_name] = st.session_state.sessions.pop(st.session_state.current_session, [])
        st.session_state.current_session = session_name

with colB:
    if st.button("➕ New session"):
        base = "Session"
        i = 1
        while f"{base} {i}" in st.session_state.sessions:
            i += 1
        new_name = f"{base} {i}"
        st.session_state.sessions[new_name] = [{"role": "system", "content": "You are a concise Formula 1 analyst."}]
        st.session_state.current_session = new_name

session_names = list(st.session_state.sessions.keys())
session_select = st.selectbox("Switch session", options=session_names, index=session_names.index(st.session_state.current_session))
if session_select != st.session_state.current_session:
    st.session_state.current_session = session_select

messages = st.session_state.sessions[st.session_state.current_session]

# ---- LLM client (if needed) ----
client = None
if not api_only:
    try:
        client = LLMClient(model=model, temperature=temperature)
    except Exception as e:
        st.error(str(e))
        logger.exception("LLMClient init failed")

# ---- Tabs ----
tab_chat, tab_tools = st.tabs(["💬 Chat", "🧰 Tools"])

with tab_chat:
    # Render history
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("Ask about a season or a specific Grand Prix...")
    if user_query:
        messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        logger.info(f"Query: {user_query}")

        # Build prompt + facts (always so we can show facts even in API mode)
        prompt, facts = build_prompt(user_query)
        routed = route_query(user_query)

        if api_only:
            # API mode: return facts as JSON (no LLM call)
            resp = build_api_response(user_query)
            with st.chat_message("assistant"):
                st.json(resp)
            messages.append({"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)})
        else:
            # Call LLM
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    payload = [
                        {"role": "system", "content": "You are a concise Formula 1 analyst."},
                        {"role": "user", "content": prompt},
                    ]
                    try:
                        answer = client.chat(payload)
                    except Exception as e:
                        answer = f"Error calling LLM: {e}"
                        logger.exception("LLM call failed")

                    st.markdown(answer)

                    if facts:
                        with st.expander("Show facts used in the answer"):
                            st.code(facts, language="markdown")

            messages.append({"role": "assistant", "content": answer})

with tab_tools:
    st.subheader("Quick function tester")
    test_query = st.text_input("Test a query", placeholder="e.g., 'Summarize the 2024 season' or 'Who won Monaco GP 2023?'")
    if st.button("Route query"):
        qtype = route_query(test_query) if test_query else "unknown"
        st.write(f"Type: **{qtype}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build prompt + facts"):
            if not test_query:
                st.warning("Enter a test query first.")
            else:
                prompt, facts = build_prompt(test_query)
                st.markdown("**Prompt:**")
                st.code(prompt)
                st.markdown("**Facts:**")
                st.code(facts or "(empty)")

    with col2:
        if st.button("API-only JSON for test query"):
            if not test_query:
                st.warning("Enter a test query first.")
            else:
                resp = build_api_response(test_query)
                st.json(resp)
                st.download_button("Download JSON", data=json.dumps(resp, ensure_ascii=False, indent=2), file_name="facts.json", mime="application/json")
