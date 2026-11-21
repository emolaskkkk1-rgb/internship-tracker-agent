import os
import uuid
import streamlit as st
from langgraph.types import Command

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from agent import build_graph

st.title("Internship Application Tracker")
st.caption("Track applications, check company info and weather, delete safely with approval.")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.history = []
    st.session_state.pending = None

graph, config = st.session_state.graph, st.session_state.config


def handle(result):
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
    else:
        st.session_state.pending = None
        st.session_state.history.append(("assistant", result["messages"][-1].content))


for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

if st.session_state.pending:
    st.warning(st.session_state.pending)
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve"):
        handle(graph.invoke(Command(resume=True), config))
        st.rerun()
    if c2.button("❌ Deny"):
        handle(graph.invoke(Command(resume=False), config))
        st.rerun()
else:
    prompt = st.chat_input("Ask the agent...")
    if prompt:
        st.session_state.history.append(("user", prompt))
        try:
            handle(graph.invoke({"messages": [("user", prompt)]}, config))
        except Exception as e:
            st.session_state.history.append(("assistant", f"Error: {e}"))
        st.rerun()
