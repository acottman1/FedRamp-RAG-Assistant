"""FedRAMP Readiness Assistant – Streamlit chat interface.

Run with:
    streamlit run app/main.py
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root before any other imports that need env vars.
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Put app/ on sys.path so `import rag` resolves from the same directory.
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from rag import build_query_engine, query as rag_query

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = _PROJECT_ROOT / "logs" / "query_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log(question: str, result: dict) -> None:
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "chunks": [
            {
                "text": c["text"][:500],
                "score": c["score"],
                "citation": c["citation"],
            }
            for c in result["chunks"]
        ],
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Engine (cached across reruns) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading index…")
def get_engine():
    return build_query_engine()


# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedRAMP Readiness Assistant",
    page_icon="🔒",
    layout="wide",
)
st.title("🔒 FedRAMP Readiness Assistant")
st.caption(
    "Ask questions about FedRAMP authorization requirements. "
    "Answers are grounded solely in your uploaded documents."
)

# Sidebar: quick status
with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant uses **Retrieval-Augmented Generation** (RAG) to answer "
        "FedRAMP questions from a local document corpus.\n\n"
        "It will refuse to answer if the retrieved documents do not support a response."
    )
    st.divider()
    st.markdown(f"**Log file:** `{LOG_PATH.relative_to(_PROJECT_ROOT)}`")
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            count = sum(1 for _ in f)
        st.markdown(f"**Queries logged:** {count}")

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.caption("**Sources:** " + " · ".join(msg["citations"]))
        if msg.get("chunks"):
            with st.expander("Retrieved context", expanded=False):
                for chunk in msg["chunks"]:
                    label = chunk.get("citation", "chunk")
                    score = chunk.get("score")
                    header = f"**{label}**" + (f"  ·  score: {score}" if score else "")
                    st.markdown(header)
                    st.text(chunk["text"][:600])
                    st.divider()


for msg in st.session_state.messages:
    _render_message(msg)

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a FedRAMP question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                engine = get_engine()
                result = rag_query(engine, prompt)
            except FileNotFoundError as exc:
                result = {
                    "answer": f"⚠️ Index not found. {exc}",
                    "citations": [],
                    "chunks": [],
                }
            except Exception as exc:
                result = {
                    "answer": f"⚠️ An unexpected error occurred: {exc}",
                    "citations": [],
                    "chunks": [],
                }

        st.markdown(result["answer"])
        if result["citations"]:
            st.caption("**Sources:** " + " · ".join(result["citations"]))
        if result["chunks"]:
            with st.expander("Retrieved context", expanded=False):
                for chunk in result["chunks"]:
                    label = chunk.get("citation", "chunk")
                    score = chunk.get("score")
                    header = f"**{label}**" + (f"  ·  score: {score}" if score else "")
                    st.markdown(header)
                    st.text(chunk["text"][:600])
                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "chunks": result["chunks"],
        }
    )
    _log(prompt, result)
