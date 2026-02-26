import json
import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from tools.kanoon_rag_tool import rag_retrieve_tool
from tools.router_tool import route_query, RoutingDecision
from tools.procedure_rag_tool import procedure_retrieve_tool
from tools.form_filling_tool import (
    plan_forms_for_case,
    start_form_session,
    form_step,
)
load_dotenv()


class SimpleResponse(BaseModel):
    argument: str
    conclusion: str


llm = ChatOpenAI(model="gpt-4.1-mini")

# -----------------------------
# Per-user storage
# -----------------------------
USERS_ROOT = "data_users"  # parent folder for all users
WINDOW_TURNS = 6
MAX_RECENT_CHARS = 9000
SUMMARIZE_CHARS_AT_ONCE = 7000


def _sanitize_user_id(user_id: str) -> str:
    """
    Keep it simple/safe for folder names. You said numeric IDs for now,
    but this supports any basic string.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return "unknown"
    # allow only letters/numbers/_-
    user_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id)
    return user_id


def get_user_dir(user_id: str) -> str:
    uid = _sanitize_user_id(user_id)
    user_dir = os.path.join(USERS_ROOT, f"data_user_id_{uid}")
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_state_path(user_dir: str) -> str:
    return os.path.join(user_dir, "chat_state.json")


def default_state() -> Dict[str, Any]:
    return {
        "summary": "",
        "recent": [],
        "last_evidence": None,
        "last_evidence_hint": "",
        "last_evidence_source": "",
        "form_session": None,
        "last_form_plan": None,
    }


def load_state(state_path: str) -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return default_state()
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "summary": str(data.get("summary", "")),
        "recent": list(data.get("recent", [])),
        "last_evidence": data.get("last_evidence", None),
        "last_evidence_hint": str(data.get("last_evidence_hint", "")),
        "last_evidence_source": str(data.get("last_evidence_source", "")),
        "form_session": data.get("form_session", None),
        "last_form_plan": data.get("last_form_plan", None),
    }


def save_state(state_path: str, state: Dict[str, Any]) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
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


def build_procedure_answer_prompt(user_query: str, evidence_pack: dict, memory_summary: str, recent_msgs: List[Dict[str, str]]) -> str:
    return f"""
You are a legal assistant for India. Answer grounded ONLY in the retrieved procedure/forms manuals evidence.

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
- Use ONLY the retrieved excerpts for procedural/form facts (filing steps, portal steps, required documents, form formats).
- Do NOT invent forms, steps, portals, fees, timelines, or requirements.
- If the excerpts do not contain enough info, say exactly:
"I don't know based on the procedure/forms corpus."
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
# Handlers
# -----------------------------
def handle_retrieve(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    evidence = rag_retrieve_tool(
        user_query=decision.search_query or user_query,
        per_query_k=10,
        max_docs=30,
        top_chunks=8,
    )
    state["last_evidence"] = evidence
    state["last_evidence_hint"] = evidence_hint(evidence)
    state["last_evidence_source"] = "kanoon"

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


def handle_retrieve_procedure(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    evidence = procedure_retrieve_tool(
        user_query=decision.search_query or user_query,
        top_chunks=8,
    )
    state["last_evidence"] = evidence
    state["last_evidence_hint"] = evidence_hint(evidence)
    state["last_evidence_source"] = "procedure"

    prompt = build_procedure_answer_prompt(user_query, evidence, state["summary"], state["recent"])
    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        return json.dumps(
            {"argument": "I don't know based on the procedure/forms corpus.", "conclusion": "I don't know based on the procedure/forms corpus."},
            ensure_ascii=False,
            indent=2,
        )

def handle_start_form_fill(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    # user_dir is needed to save JSON in that user's folder
    user_dir = state.get("user_dir") or "."
    
    # Always ground "forms/procedure" planning in procedure corpus
    proc_ev = procedure_retrieve_tool(
        user_query=decision.search_query or user_query,
        top_chunks=8,
    )

    # Plan forms + steps (V0 planner; later you can make it LLM-based)
    plan = plan_forms_for_case(
        user_query=user_query,
        kanoon_evidence=None,
        procedure_evidence=proc_ev,
    )
    state["last_form_plan"] = plan

    forms = plan.get("forms") or []
    if not forms:
        # Hard safe fallback
        return json.dumps(
            {"argument": "No forms were identified for this filing request.", "conclusion": "Tell me what you want to file (civil/criminal/family) and where in India."},
            ensure_ascii=False,
            indent=2,
        )

    top = forms[0]

    # Start a form session + write a JSON file into user folder
    context_text = (state.get("summary", "") or "") + "\n\n" + (format_recent(state.get("recent", [])) or "")
    state["form_session"] = start_form_session(llm, user_dir, top, context_text)

    first_q = "Tell me anything you already know for this filing (party names, court/city, amount, dates). I’ll fill what I can and ask only what’s missing."

    return json.dumps(
        {
            "argument": {
                "forms_ranked": [
                    {
                        "priority": f.get("priority"),
                        "form_id": f.get("form_id"),
                        "title": f.get("title"),
                        "reason": f.get("reason"),
                    }
                    for f in forms
                ],
                "procedure_steps": top.get("procedure_steps") or [],
            },
            "conclusion": first_q,
        },
        ensure_ascii=False,
        indent=2,
    )

def handle_reuse_last_evidence(user_query: str, decision: RoutingDecision, state: Dict[str, Any]) -> str:
    evidence = state.get("last_evidence") or {}
    src = (state.get("last_evidence_source") or "kanoon").strip().lower()

    if src == "procedure":
        prompt = build_procedure_answer_prompt(user_query, evidence, state["summary"], state["recent"])
    else:
        prompt = build_answer_prompt(user_query, evidence, state["summary"], state["recent"])

    resp = llm.invoke(prompt)
    try:
        structured = SimpleResponse.model_validate_json(resp.content)
        return structured.model_dump_json(indent=2)
    except Exception:
        msg = "I don't know based on the procedure/forms corpus." if src == "procedure" else "I don't know based on Indian Kanoon."
        return json.dumps({"argument": msg, "conclusion": msg}, ensure_ascii=False, indent=2)


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
    "RETRIEVE_PROCEDURE": handle_retrieve_procedure,
    "REUSE_LAST_EVIDENCE": handle_reuse_last_evidence,
    "START_FORM_FILL": handle_start_form_fill,
    "NO_RETRIEVAL_EXPLAIN": handle_explain,
    "ASK_CLARIFY": handle_clarify,
    "OUT_OF_SCOPE": handle_oos,
}


def main():
    os.makedirs(USERS_ROOT, exist_ok=True)

    user_id = input("User ID (number): ").strip()
    user_dir = get_user_dir(user_id)
    state_path = get_state_path(user_dir)

    state = load_state(state_path)
    state["user_dir"] = user_dir
    print(f"Using user folder: {user_dir}")
    print("Chat started. /quit to exit. /reset to clear memory.\n")

    while True:
        user_query = input("> ").strip()
        if not user_query:
            continue

        if user_query.lower() in {"/quit", "/exit"}:
            save_state(state_path, state)
            print("Bye.")
            break

        if user_query.lower() == "/reset":
            state = default_state()
            state["user_dir"] = user_dir
            save_state(state_path, state)
            print("Memory cleared.")
            continue

        if state.get("form_session") and state["form_session"].get("active"):
            step = form_step(llm, state["form_session"], user_query)
            state["form_session"] = step["session"]

            if step["mode"] == "DELEGATE_NORMAL_CHAT":
                # do NOT exit form mode; just answer normally via router below
                pass
            else:
                assistant_text = json.dumps(step["assistant_json"], ensure_ascii=False, indent=2)
                print(assistant_text)
                state["recent"].append({"role": "assistant", "content": assistant_text})
                state = maybe_summarize(state)
                save_state(state_path, state)
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
        save_state(state_path, state)


if __name__ == "__main__":
    main()
