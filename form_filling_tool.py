# tools/form_filling_tool.py
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple

from pydantic import BaseModel, Field


DEFAULT_FORMS_SUBDIR = "forms"

# -----------------------------
# V0 form template (field set)
# -----------------------------
ECourtsCivilEfilingFields: List[Dict[str, Any]] = [
    {"key": "case_type", "question": "Case type (e.g., Civil Suit / Recovery / Contract dispute):", "required": True},
    {"key": "court_location", "question": "Which court/city are you filing in (State + City + Court name if known)?", "required": True},

    {"key": "plaintiff_name", "question": "Plaintiff full name:", "required": True},
    {"key": "plaintiff_gender", "question": "Plaintiff gender (or 'UNKNOWN'):", "required": False},
    {"key": "plaintiff_age", "question": "Plaintiff age (or 'UNKNOWN'):", "required": False},
    {"key": "plaintiff_mobile", "question": "Plaintiff mobile number (or 'UNKNOWN'):", "required": False},
    {"key": "plaintiff_email", "question": "Plaintiff email (or 'UNKNOWN'):", "required": False},

    {"key": "defendant_name", "question": "Defendant full name:", "required": True},
    {"key": "defendant_gender", "question": "Defendant gender (or 'UNKNOWN'):", "required": False},
    {"key": "defendant_age", "question": "Defendant age (or 'UNKNOWN'):", "required": False},
    {"key": "defendant_mobile", "question": "Defendant mobile number (or 'UNKNOWN'):", "required": False},
    {"key": "defendant_email", "question": "Defendant email (or 'UNKNOWN'):", "required": False},
    {"key": "extra_party_count", "question": "Any extra parties? Enter number (0 if none):", "required": False},

    {"key": "advocate_name", "question": "Advocate name (or type 'NONE' if self-representing):", "required": True},
    {"key": "advocate_bar_reg_no", "question": "Advocate Bar Registration Number (or 'NONE'/'UNKNOWN'):", "required": False},

    {"key": "suit_valuation", "question": "Suit valuation amount (numbers only if possible):", "required": True},
    {"key": "cause_of_action_date", "question": "Cause of action date (YYYY-MM-DD) (or 'UNKNOWN'):", "required": False},
    {"key": "cause_of_action_summary", "question": "Cause of action summary (1–3 sentences):", "required": True},
    {"key": "prayer", "question": "Prayer / relief sought (what you want the court to order):", "required": True},

    {"key": "acts_sections", "question": "Relevant Acts/Sections (if known). If unknown, type 'UNKNOWN':", "required": False},
]


@dataclass
class FormPlanItem:
    form_id: str
    title: str
    priority: int
    reason: str
    procedure_steps: List[str]
    fields: List[Dict[str, Any]]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_forms_dir(user_dir: str) -> str:
    forms_dir = os.path.join(user_dir, DEFAULT_FORMS_SUBDIR)
    os.makedirs(forms_dir, exist_ok=True)
    return forms_dir


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Planner (router already chose START_FORM_FILL)
# -----------------------------
def plan_forms_for_case(
    user_query: str,
    kanoon_evidence: Optional[Dict[str, Any]] = None,
    procedure_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    q = (user_query or "").strip()

    steps = [
        "Collect contract + proof of work completion + invoices/messages.",
        "Decide forum: civil court (recovery/breach) or arbitration (if contract has arbitration clause).",
        "Prepare plaint and annexures; decide suit valuation and relief/prayer.",
        "Fill filing details (party, advocate, valuation, cause of action, relief) in the filing process.",
        "Submit and keep acknowledgement/filing reference.",
    ]

    plan: List[FormPlanItem] = []

    if ECourtsCivilEfilingFields:
        plan.append(
            FormPlanItem(
                form_id="ecourts_civil_case_filing_v0",
                title="eCourts Civil Case Filing (V0 field set)",
                priority=1,
                reason="Router initiated form-filling; using the minimum civil filing field set currently supported.",
                procedure_steps=steps,
                fields=ECourtsCivilEfilingFields,
            )
        )

    if not plan:
        plan.append(
            FormPlanItem(
                form_id="generic_case_intake_v0",
                title="Generic Case Intake (fallback)",
                priority=1,
                reason="No form templates available; collecting core case facts to proceed.",
                procedure_steps=steps,
                fields=[
                    {"key": "problem_summary", "question": "Describe the problem in 3–6 sentences:", "required": True},
                    {"key": "where", "question": "Which state/city in India does this relate to?", "required": True},
                    {"key": "parties", "question": "Who are the parties (names + roles)?", "required": True},
                    {"key": "documents_available", "question": "Which documents do you have (contract, invoices, chats, etc.)?", "required": False},
                    {"key": "outcome_wanted", "question": "What outcome do you want?", "required": True},
                ],
            )
        )

    plan = sorted(plan, key=lambda x: x.priority)
    return {"query": q, "forms": [asdict(p) for p in plan]}


# -----------------------------
# LLM-driven extraction & validation (structured)
# -----------------------------

class PrefillResult(BaseModel):
    extracted: Dict[str, str] = Field(default_factory=dict)
    missing_keys: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class InterpretAction(BaseModel):
    """
    Decide what the user's message is doing relative to the form session.
    """
    action: Literal["UPDATE_FIELDS", "ASK_CLARIFY", "DELEGATE_NORMAL_CHAT", "CONFIRM_VALUE"]  # delegate keeps form session active
    updates: Dict[str, str] = Field(default_factory=dict)  # field_key -> value
    target_field: Optional[str] = None  # for CONFIRM_VALUE
    proposed_value: Optional[str] = None
    question: Optional[str] = None
    reason: str


class ValidateResult(BaseModel):
    ok: bool
    needs_confirmation: bool = False
    issue: Optional[str] = None
    explanation: Optional[str] = None


def _fields_by_key(fields: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {f["key"]: f for f in (fields or []) if "key" in f}


def _form_snapshot_for_llm(form_json: Dict[str, Any], max_chars: int = 2500) -> str:
    # keep short to avoid token bloat
    data = form_json.get("data", {})
    s = json.dumps(data, ensure_ascii=False)
    return s[:max_chars]


def prefill_from_context(llm, fields: List[Dict[str, Any]], context_text: str) -> PrefillResult:
    """
    Uses LLM to extract any field values it can from chat context.
    """
    keys = [f["key"] for f in fields]
    prompt = f"""
Extract any values for the following form fields from the conversation context.

Fields (keys): {keys}

Context:
{context_text}

Rules:
- Only extract if explicitly present in context.
- Return JSON only with keys: extracted (object), missing_keys (list), notes (list)
- extracted values must be strings.
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content or ""
    # robust parse: try json.loads; if fails, return empty
    try:
        obj = json.loads(raw)
        return PrefillResult.model_validate(obj)
    except Exception:
        return PrefillResult(extracted={}, missing_keys=keys, notes=["prefill_parse_failed"])


def interpret_user_message(llm, fields: List[Dict[str, Any]], form_json: Dict[str, Any], user_msg: str) -> InterpretAction:
    """
    Determine whether the user is:
    - providing info that maps to one or more fields
    - asking a question (delegate)
    - asking which info is needed (ask clarify)
    - responding to a confirmation (handled in main via pending state)
    """
    keys = [f["key"] for f in fields]
    field_questions = {f["key"]: f.get("question") for f in fields}

    prompt = f"""
You are helping fill a legal filing form while staying conversational.

Form fields (keys): {keys}
Field questions: {json.dumps(field_questions, ensure_ascii=False)}

Current filled data (JSON):
{_form_snapshot_for_llm(form_json)}

User message:
{user_msg}

Decide what to do next.

Actions:
- UPDATE_FIELDS: user provided info that should fill one or more fields (even if partial).
- ASK_CLARIFY: user is asking what info is needed or is unclear; ask 1 concise question that advances form completion.
- DELEGATE_NORMAL_CHAT: user asked a general question not meant as form data; do not update form.
- CONFIRM_VALUE: user provided a value but it likely doesn't make sense; ask confirmation.

Return ONLY JSON with schema:
{{
  "action": "UPDATE_FIELDS|ASK_CLARIFY|DELEGATE_NORMAL_CHAT|CONFIRM_VALUE",
  "updates": {{"field_key":"value", "...":"..."}},
  "target_field": "field_key or null",
  "proposed_value": "string or null",
  "question": "string or null",
  "reason": "short"
}}
Rules:
- If DELEGATE_NORMAL_CHAT, updates must be empty.
- If ASK_CLARIFY, provide question.
- If CONFIRM_VALUE, set target_field + proposed_value + question.
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content or ""
    try:
        obj = json.loads(raw)
        return InterpretAction.model_validate(obj)
    except Exception:
        # safest fallback: ask for the next missing required field
        return InterpretAction(
            action="ASK_CLARIFY",
            updates={},
            question="What is the defendant's full name (as on their ID/official records)?",
            reason="interpret_parse_failed",
        )


def validate_value_llm(llm, field: Dict[str, Any], proposed_value: str, form_json: Dict[str, Any]) -> ValidateResult:
    """
    LLM-based plausibility check. No hardcoded rules; it reasons using the field question + current form.
    """
    prompt = f"""
You are validating a user's proposed value for a form field.

Field key: {field.get("key")}
Field question/meaning: {field.get("question")}
Required: {bool(field.get("required"))}

Proposed value:
{proposed_value}

Current form data:
{_form_snapshot_for_llm(form_json)}

Decide if value is plausible. If not, require confirmation and explain briefly.

Return ONLY JSON:
{{
  "ok": true/false,
  "needs_confirmation": true/false,
  "issue": "short or null",
  "explanation": "short or null"
}}

Guidelines:
- Examples of suspicious: name is just numbers, email has no @, phone too short, age non-numeric, etc.
- If suspicious but maybe possible, set needs_confirmation=true.
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content or ""
    try:
        obj = json.loads(raw)
        return ValidateResult.model_validate(obj)
    except Exception:
        # conservative: ask confirmation if parse fails
        return ValidateResult(ok=False, needs_confirmation=True, issue="validation_parse_failed", explanation="I couldn't validate this value. Confirm if you want to keep it.")


# -----------------------------
# Session lifecycle
# -----------------------------
def start_form_session(
    llm,
    user_dir: str,
    form_item: Dict[str, Any],
    context_text_for_prefill: str,
) -> Dict[str, Any]:
    """
    Creates JSON file and returns a session.
    Also pre-fills from context and stores pending missing list.
    """
    forms_dir = ensure_forms_dir(user_dir)

    safe_id = str(form_item.get("form_id") or "form")
    filename = f"{safe_id}_{_now_ts()}.json"
    path = os.path.join(forms_dir, filename)

    fields = form_item.get("fields") or []
    field_map = _fields_by_key(fields)

    payload = {
        "form_id": form_item.get("form_id"),
        "title": form_item.get("title"),
        "created_at": _now_ts(),
        "data": {},
        "status": "in_progress",
    }

    # prefill
    pre = prefill_from_context(llm, fields, context_text_for_prefill)
    for k, v in (pre.extracted or {}).items():
        if k in field_map and isinstance(v, str) and v.strip():
            payload["data"][k] = v.strip()

    _write_json(path, payload)

    # compute missing required keys
    missing_required = []
    for f in fields:
        if f.get("required") and not payload["data"].get(f["key"]):
            missing_required.append(f["key"])

    return {
        "active": True,
        "form_path": path,
        "form_id": form_item.get("form_id"),
        "title": form_item.get("title"),
        "fields": fields,
        "missing_required": missing_required,
        "pending_confirmation": None,  # {"field":..., "value":..., "explanation":..., "question":...}
    }


def load_form(session: Dict[str, Any]) -> Dict[str, Any]:
    return _read_json(session["form_path"])


def save_form(session: Dict[str, Any], form_json: Dict[str, Any]) -> None:
    _write_json(session["form_path"], form_json)


def next_missing_question(fields: List[Dict[str, Any]], form_json: Dict[str, Any]) -> Optional[str]:
    for f in fields:
        if f.get("required") and not (form_json.get("data", {}) or {}).get(f["key"]):
            return f.get("question") or None
    return None


def apply_updates(session: Dict[str, Any], updates: Dict[str, str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Applies updates without validation. Returns (updated_form, applied_keys).
    Validation happens per-field before calling this.
    """
    form_json = load_form(session)
    form_json.setdefault("data", {})
    applied = []
    for k, v in (updates or {}).items():
        if isinstance(v, str):
            form_json["data"][k] = v.strip()
            applied.append(k)
    save_form(session, form_json)
    return form_json, applied


def maybe_complete(session: Dict[str, Any]) -> bool:
    form_json = load_form(session)
    fields = session.get("fields") or []
    for f in fields:
        if f.get("required") and not (form_json.get("data", {}) or {}).get(f["key"]):
            return False
    form_json["status"] = "completed"
    save_form(session, form_json)
    session["active"] = False
    return True


# -----------------------------
# Main step function (used by main.py when form session is active)
# -----------------------------
def form_step(
    llm,
    session: Dict[str, Any],
    user_msg: str,
) -> Dict[str, Any]:
    """
    Returns a dict telling main.py what to do next.
    This does NOT answer legal questions; it can request delegation.

    Output schema:
    {
      "mode": "DELEGATE_NORMAL_CHAT" | "FORM_REPLY",
      "assistant_json": {...},   # what to print as assistant message (already JSON serializable)
      "session": session         # updated session
    }
    """
    # Handle pending confirmation
    if session.get("pending_confirmation"):
        pc = session["pending_confirmation"]
        ans = (user_msg or "").strip().lower()
        if ans in {"yes", "y", "confirm", "ok", "okay", "correct"}:
            form_json, _ = apply_updates(session, {pc["field"]: pc["value"]})
            session["pending_confirmation"] = None
            done = maybe_complete(session)
            q = None if done else next_missing_question(session.get("fields") or [], form_json)
            return {
                "mode": "FORM_REPLY",
                "assistant_json": {
                    "argument": f"Saved {pc['field']}.",
                    "conclusion": ("Form completed and saved." if done else (q or "What’s the next detail you want to add?")),
                },
                "session": session,
            }
        if ans in {"no", "n", "change", "wrong"}:
            session["pending_confirmation"] = None
            # ask the same field again
            fields = _fields_by_key(session.get("fields") or [])
            fq = fields.get(pc["field"], {}).get("question") or f"Provide {pc['field']}:"
            return {
                "mode": "FORM_REPLY",
                "assistant_json": {
                    "argument": "Okay—let’s correct it.",
                    "conclusion": fq,
                },
                "session": session,
            }
        # otherwise ask for yes/no
        return {
            "mode": "FORM_REPLY",
            "assistant_json": {
                "argument": "I need a yes/no to proceed.",
                "conclusion": pc.get("question") or "Confirm? (yes/no)",
            },
            "session": session,
        }

    form_json = load_form(session)
    fields = session.get("fields") or []
    field_map = _fields_by_key(fields)

    decision = interpret_user_message(llm, fields, form_json, user_msg)

    if decision.action == "DELEGATE_NORMAL_CHAT":
        return {
            "mode": "DELEGATE_NORMAL_CHAT",
            "assistant_json": None,
            "session": session,
        }

    if decision.action == "ASK_CLARIFY":
        return {
            "mode": "FORM_REPLY",
            "assistant_json": {
                "argument": "To keep filling the form, I need one detail.",
                "conclusion": decision.question or (next_missing_question(fields, form_json) or "What detail do you want to add?"),
            },
            "session": session,
        }

    if decision.action == "CONFIRM_VALUE":
        tf = decision.target_field
        pv = decision.proposed_value
        if tf and pv and tf in field_map:
            vr = validate_value_llm(llm, field_map[tf], pv, form_json)
            # always ask confirmation here
            session["pending_confirmation"] = {
                "field": tf,
                "value": pv,
                "explanation": vr.explanation or "This value seems unusual.",
                "question": decision.question or f"Confirm you want to set {tf} = {pv!r}? (yes/no)",
            }
            return {
                "mode": "FORM_REPLY",
                "assistant_json": {
                    "argument": session["pending_confirmation"]["explanation"],
                    "conclusion": session["pending_confirmation"]["question"],
                },
                "session": session,
            }
        # fallback
        return {
            "mode": "FORM_REPLY",
            "assistant_json": {
                "argument": "I couldn't map that to a specific field.",
                "conclusion": next_missing_question(fields, form_json) or "What detail do you want to add?",
            },
            "session": session,
        }

    # UPDATE_FIELDS
    updates = decision.updates or {}
    valid_updates: Dict[str, str] = {}
    confirm_needed = None

    for k, v in updates.items():
        if k not in field_map:
            continue
        vr = validate_value_llm(llm, field_map[k], v, form_json)
        if vr.ok and not vr.needs_confirmation:
            valid_updates[k] = v
        else:
            # ask confirmation on first suspicious value; keep others for later
            confirm_needed = (k, v, vr.explanation or "This value seems unusual.")
            break

    if confirm_needed:
        k, v, expl = confirm_needed
        session["pending_confirmation"] = {
            "field": k,
            "value": v,
            "explanation": expl,
            "question": f"You entered {v!r} for {k}. Confirm this is correct? (yes/no)",
        }
        return {
            "mode": "FORM_REPLY",
            "assistant_json": {
                "argument": expl,
                "conclusion": session["pending_confirmation"]["question"],
            },
            "session": session,
        }

    if valid_updates:
        form_json, applied = apply_updates(session, valid_updates)
        done = maybe_complete(session)
        q = None if done else next_missing_question(fields, form_json)
        return {
            "mode": "FORM_REPLY",
            "assistant_json": {
                "argument": f"Saved fields: {applied}",
                "conclusion": ("Form completed and saved." if done else (q or "What else should we add?")),
            },
            "session": session,
        }

    # nothing applied
    return {
        "mode": "FORM_REPLY",
        "assistant_json": {
            "argument": "I couldn’t safely apply that to the form.",
            "conclusion": next_missing_question(fields, form_json) or "Tell me the detail you want to add.",
        },
        "session": session,
    }