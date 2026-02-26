# tools/router_tool.py
from __future__ import annotations

import json
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

Action = Literal[
    "RETRIEVE_KANOON",
    "RETRIEVE_PROCEDURE",   # NEW
    "REUSE_LAST_EVIDENCE",
    "NO_RETRIEVAL_EXPLAIN",
    "ASK_CLARIFY",
    "OUT_OF_SCOPE",
    "START_FORM_FILL",
]


class RoutingDecision(BaseModel):
    action: Action
    jurisdiction: Literal["IN", "US", "OTHER"] = "IN"
    search_query: Optional[str] = None
    needs_citations: bool = True
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_json_object(text: str) -> str:
    if not text:
        raise ValueError("Empty router output")
    t = re.sub(_CODE_FENCE_RE, "", text).strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    if start == -1:
        raise ValueError(f"No '{{' found in router output: {t[:200]}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    raise ValueError(f"Unbalanced braces in router output: {t[:200]}")


def route_query(
    llm: ChatOpenAI,
    user_query: str,
    memory_summary: str,
    recent_window_text: str,
    last_evidence_available: bool,
    last_evidence_hint: str,
) -> RoutingDecision:
    """
    last_evidence_hint: short string describing what last evidence was about (no huge JSON)
    """
    prompt = f"""
You are a routing controller for a legal assistant.

Choose whether to retrieve from Indian Kanoon OR the procedure/forms corpus, or not.

Available actions:
- RETRIEVE_KANOON: run retrieval now
- REUSE_LAST_EVIDENCE: do NOT retrieve; reuse evidence from the most recent retrieval
- NO_RETRIEVAL_EXPLAIN: do NOT retrieve; explain/simplify based on chat context only
- ASK_CLARIFY: ask 1–3 questions before doing anything else
- OUT_OF_SCOPE: non-India law or outside corpus
- RETRIEVE_PROCEDURE: retrieve from procedure/forms manuals corpus (filing steps, forms, e-filing guides)

Rules:
1) OUT_OF_SCOPE if user asks about non-India law (US/UK/etc).
2) RETRIEVE_PROCEDURE if the user asks about filing steps, e-filing process, required documents, court forms/templates/format, fees payment steps, annexures/affidavit formats, procedural checklists.
3) RETRIEVE_KANOON if the user asks for authoritative legal details OR new law/section/case/judgment/procedure/punishment/bail/FIR/petition/etc.
4) REUSE_LAST_EVIDENCE if:
   - last_evidence_available=true, AND
   - the user is continuing the SAME India-law topic, AND
   - the user is asking to apply/clarify/explain the SAME rule already in the last evidence,
   - and no new legal topic is introduced.
5) NO_RETRIEVAL_EXPLAIN if the user is only asking for explanation of what was already said and it does NOT require legal facts beyond what was retrieved.
6) ASK_CLARIFY if missing key facts and retrieval would likely be wasted.
7) If action=RETRIEVE_KANOON, set search_query to a short Indian Kanoon query (3–12 words). Otherwise search_query must be null.
8) Return ONLY a JSON object (no markdown, no code fences).
9) START_FORM_FILL if user asks to file, help me file, help me fill the form, which form do I need, prepare filing, etc.

Schema:
{{
  "action": "RETRIEVE_KANOON|RETRIEVE_PROCEDURE|START_FORM_FILL|REUSE_LAST_EVIDENCE|NO_RETRIEVAL_EXPLAIN|ASK_CLARIFY|OUT_OF_SCOPE",
  "jurisdiction": "IN|US|OTHER",
  "search_query": "string or null",
  "needs_citations": true/false,
  "confidence": 0.0-1.0,
  "reason": "short"
}}

Conversation memory summary:
{memory_summary if memory_summary else "(none)"}

Recent window:
{recent_window_text if recent_window_text else "(none)"}

Last evidence available: {str(bool(last_evidence_available)).lower()}
Last evidence hint:
{last_evidence_hint if last_evidence_hint else "(none)"}

User query:
{user_query}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content or ""
    obj = json.loads(_extract_json_object(raw))
    return RoutingDecision.model_validate(obj)