import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from tools.kanoon_rag_tool import rag_retrieve_tool
from tools.router_tool import route_query, RoutingDecision

load_dotenv()


class SimpleResponse(BaseModel):
    argument: str
    conclusion: str


llm = ChatOpenAI(model="gpt-4.1-mini")

STATE_PATH = "chat_state.json"
WINDOW_TURNS = 6
MAX_RECENT_CHARS = 9000
SUMMARIZE_CHARS_AT_ONCE = 7000


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"summary": "", "recent": [], "last_evidence": None, "last_evidence_hint": ""}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "summary": str(data.get("summary", "")),
        "recent": list(data.get("recent", [])),
        "last_evidence": data.get("last_evidence", None),
        "last_evidence_hint": str(data.get("last_evidence_hint", "")),
    }


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def format_recent(recent: List[Dict[str, str]]) -> str:
    parts = []
    for m in recent:
        parts.append(f'{m.get("role","").upper()}: {m.get("content","")}')
    return "\n".join(parts)


def total_recent_chars(recent: List[Dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in recent)


def summarize_text(old_summary: str, new_text: str) -> str:
    prompt = f"""
You maintain a running conversation summary for a legal assistant.

Update the summary using the new transcript chunk.
Rules:
- Keep it compact.
- Include: user goals, constraints, key facts, decisions.
- Do NOT add facts not in the transcript.
- Output ONLY the updated summary text.

CURRENT SUMMARY:
{old_summary if old_summary else "(empty)"}

NEW TRANSCRIPT CHUNK:
{new_text}
""".strip()
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()


def enforce_window(state: Dict[str, Any]) -> Dict[str, Any]:
    recent = state["recent"]
    user_idxs = [i for i, m in enumerate(recent) if m.get("role") == "user"]
    if len(user_idxs) <= WINDOW_TURNS:
        return state
    cut_at = user_idxs[-WINDOW_TURNS]
    state["recent"] = recent[cut_at:]
    return state


def maybe_summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    if total_recent_chars(state["recent"]) <= MAX_RECENT_CHARS:
        return enforce_window(state)

    recent_text = format_recent(state["recent"])
    chunk = recent_text[:SUMMARIZE_CHARS_AT_ONCE]
    state["summary"] = summarize_text(state["summary"], chunk)

    while state["recent"] and total_recent_chars(state["recent"]) > (MAX_RECENT_CHARS // 2):
        state["recent"].pop(0)

    return enforce_window(state)


def build_answer_prompt(user_query: str, evidence_pack: dict, memory_summary: str, recent_msgs: List[Dict[str, str]]) -> str:
    return f"""
You are a legal assistant for Indian-law answers grounded ONLY in the retrieved evidence.

Conversation memory (for continuity only):
Summary:
{memory_summary if memory_summary else "(none)"}

Recent:
{format_recent(recent_msgs) if recent_msgs else "(none)"}

User question:
{user_query}

Retrieved Evidence (JSON):
{evidence_pack}

Rules:
- Use ONLY the retrieved excerpts for legal facts.
- If the excerpts do not contain enough info, say exactly:
"I don't know based on Indian Kanoon."
- Return ONLY valid JSON:
{{
  "argument": "...",
  "conclusion": "..."
}}
""".strip()


def build_explain_prompt(user_query: str, memory_summary: str, recent_msgs: List[Dict[str, str]]) -> str:
    return f"""
You are a helpful assistant. The user is asking for explanation/clarification.

Use conversation memory to explain clearly.
Do NOT introduce new legal facts that were not previously supported by retrieved excerpts.
If you lack context, ask 1 short clarifying question.

Conversation memory:
Summary:
{memory_summary if memory_summary else "(none)"}

Recent:
{format_recent(recent_msgs) if recent_msgs else "(none)"}

User message:
{user_query}

Return ONLY valid JSON:
{{
  "argument": "...",
  "conclusion": "..."
}}
""".strip()


def build_clarify_prompt(user_query: str) -> str:
    return f"""
Ask 1–3 short clarifying questions needed to answer the user's question.
Do not retrieve documents. Do not answer the legal question yet.

User message:
{user_query}

Return ONLY valid JSON:
{{
  "argument": "...",
  "conclusion": "..."
}}
""".strip()


def build_oos_prompt(user_query: str) -> str:
    return f"""
The user asked about law outside India or outside the available corpus.
Say you can't answer that from Indian Kanoon, and offer to help with Indian law or general concepts.

User message:
{user_query}

Return ONLY valid JSON:
{{
  "argument": "...",
  "conclusion": "..."
}}
""".strip()


def evidence_hint(evidence: Any) -> str:
    """
    Create a short hint string for router (avoid dumping large evidence JSON).
    """
    if not isinstance(evidence, dict):
        return ""
    q = evidence.get("query", "")
    returned = evidence.get("returned", 0)
    tids = evidence.get("tids_considered") or []
    top_titles = []
    for r in (evidence.get("results") or [])[:3]:
        t = r.get("title")
        if t:
            top_titles.append(t[:80])
    return f"query={q!r}, returned={returned}, tids={tids[:5]}, top_titles={top_titles}"


# -----------------------------
# Handlers (dispatch table)
# -----------------------------

def handle_retrieve(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    evidence = rag_retrieve_tool(
        user_query=decision.search_query or user_query,
        per_query_k=10,
        max_docs=30,
        top_chunks=8,
    )
    # store for reuse
    state["last_evidence"] = evidence
    state["last_evidence_hint"] = evidence_hint(evidence)

    prompt = build_answer_prompt(user_query, evidence, state["summary"], state["recent"])
    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        return json.dumps(
            {"argument": "I don't know based on Indian Kanoon.", "conclusion": "I don't know based on Indian Kanoon."},
            ensure_ascii=False,
            indent=2,
        )


def handle_reuse_last_evidence(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    evidence = state.get("last_evidence") or {}
    prompt = build_answer_prompt(user_query, evidence, state["summary"], state["recent"])
    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        return json.dumps(
            {"argument": "I don't know based on Indian Kanoon.", "conclusion": "I don't know based on Indian Kanoon."},
            ensure_ascii=False,
            indent=2,
        )


def handle_explain(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    prompt = build_explain_prompt(user_query, state["summary"], state["recent"])
    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        return json.dumps(
            {"argument": "I need more context to explain.", "conclusion": "Paste the exact part you want explained."},
            ensure_ascii=False,
            indent=2,
        )


def handle_clarify(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    prompt = build_clarify_prompt(user_query)
    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        return json.dumps(
            {"argument": "I need a few details to answer.", "conclusion": "What happened, when, where in India, and what outcome do you want?"},
            ensure_ascii=False,
            indent=2,
        )


def handle_oos(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    prompt = build_oos_prompt(user_query)
    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        return json.dumps(
            {"argument": "That is outside my Indian Kanoon corpus.", "conclusion": "If you want Indian law, ask the India-specific question."},
            ensure_ascii=False,
            indent=2,
        )


HANDLERS = {
    "RETRIEVE_KANOON": handle_retrieve,
    "REUSE_LAST_EVIDENCE": handle_reuse_last_evidence,
    "NO_RETRIEVAL_EXPLAIN": handle_explain,
    "ASK_CLARIFY": handle_clarify,
    "OUT_OF_SCOPE": handle_oos,
}


def main():
    state = load_state()
    print("Chat started. /quit to exit. /reset to clear memory.\n")

    while True:
        user_query = input("> ").strip()
        if not user_query:
            continue

        if user_query.lower() in {"/quit", "/exit"}:
            save_state(state)
            print("Bye.")
            break

        if user_query.lower() == "/reset":
            state = {"summary": "", "recent": [], "last_evidence": None, "last_evidence_hint": ""}
            save_state(state)
            print("Memory cleared.")
            continue

        state["recent"].append({"role": "user", "content": user_query})
        state = maybe_summarize(state)

        recent_text = format_recent(state["recent"])

        decision = route_query(
            llm=llm,
            user_query=user_query,
            memory_summary=state["summary"],
            recent_window_text=recent_text,
            last_evidence_available=bool(state.get("last_evidence")),
            last_evidence_hint=state.get("last_evidence_hint", ""),
        )

        handler = HANDLERS.get(decision.action, handle_clarify)
        assistant_text = handler(user_query, decision, state)

        print(assistant_text)

        state["recent"].append({"role": "assistant", "content": assistant_text})
        state = maybe_summarize(state)
        save_state(state)


if __name__ == "__main__":
    main()