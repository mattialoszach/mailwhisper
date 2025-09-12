import re
import os
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from .schema import Plan

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Helpers
def make_llm():
    model = "qwen3:1.7b"
    return ChatOllama(model=model)

def make_structured_llm() -> Any:
    return make_llm().with_structured_output(Plan)

# Nodes
def transcribe_node(state: Dict) -> Dict:
    """Simulates STT"""
    print("\n🎤 Input:")
    print("   - subject: ... | to: ... | cc: ... | tone: friendly/formal/neutral")
    print("   - fix grammar | approve/finalize/finish | add text → gets appended")
    user_text = input("> ").strip()
    return {"transcript": user_text}

def intent_node(state: Dict) -> Dict:
    """
    Lets the LLM translate free/compact inputs into a structured update plan (Plan).
    Supports:
    - Free-form idea -> suggestion + extraction (subject/to/cc/tone/body)
    - Precise instructions -> targeted updates
    """
    transcript = state.get("transcript", "").strip()
    if not transcript:
        return {"last_op": "noop"}

    structured_llm = make_structured_llm()

    curr_state = {
        "subject": state.get("subject", ""),
        "to": state.get("to", []),
        "cc": state.get("cc", []),
        "tone": state.get("tone", "neutral"),
        "body": state.get("body", ""),
    }

    # System Prompt
    sys = SystemMessage(content=(
        "You are an assistant that converts user instructions into a STRICT JSON plan "
        "matching the provided Pydantic schema. "
        "Only extract information present; if missing, use null. "
        "Validate emails. No extra commentary—only the JSON structure."
    ))

    # User Prompt
    user = HumanMessage(content=(
        "Schema: Plan(approve: bool, updates: Updates("
        "subject?: str|null, to_set?: [EmailStr]|null, to_add?: [EmailStr]|null, "
        "cc_set?: [EmailStr]|null, cc_add?: [EmailStr]|null, tone?: {'friendly'|'formal'|'neutral'}|null, "
        "body?: BodyPlan(mode: {'replace'|'append'}|null, text?: str|null)"
        "))\n\n"
        f"Current Draft: {curr_state}\n\n"
        f"Input: {transcript}"
    ))

    plan: Plan = structured_llm.invoke([sys, user]) # Outputs Plan-object
    return {"last_op": "intent", "intent": plan.model_dump()}

def apply_node(state: Dict) -> Dict:
    """
    Applies the update plan to the draft state.
    - to/cc: *_set takes precedence over *_add
    - body: mode == replace/append
    """
    intent = state.get("intent", {}) or {}
    updates = intent.get("updates", {}) or {}

    out: Dict[str, Any] = {} # Patch for state

    # Subject
    if isinstance(updates.get("subject"), str):
        out["subject"] = updates["subject"]

    # to/cc
    curr_to = state.get("to", []) or []
    curr_cc = state.get("cc", []) or []

    to_set = updates.get("to_set")
    to_add = updates.get("to_add")
    cc_set = updates.get("cc_set")
    cc_add = updates.get("cc_add")

    if isinstance(to_set, list):
        out["to"] = list(dict.fromkeys(to_set))
    elif isinstance(to_add, list) and to_add:
        out["to"] = list(dict.fromkeys(curr_to + to_add))
    
    if isinstance(cc_set, list):
        out["cc"] = list(dict.fromkeys(cc_set))
    elif isinstance(cc_add, list) and cc_add:
        out["cc"] = list(dict.fromkeys(curr_cc + cc_add))

    # Tone
    tone = updates.get("tone")
    if tone in {"friendly", "formal", "neutral"}:
        out["tone"] = tone

    # Body
    body_plan = updates.get("body") or {}
    mode = body_plan.get("mode")
    text = body_plan.get("text")

    if isinstance(text, str) and text.strip():
        if mode == "replace":
            out["body"] = text.strip()
        elif mode == "append":
            body = state.get("body", "") or ""
            out["body"] = (body + ("\n\n" if body else "") + text.strip()).strip()
    
    return out

def critique_node(state: Dict) -> Dict:
    """
    Optional light smoothing—only when the body was just modified.
    (You can also disable it for now by returning {}.)
    """
    intent = state.get("intent", {}) or {}
    body_plan = (intent.get("updates", {}) or {}).get("body") or {}
    if not body_plan.get("text"):
        return {}
    
    body = state.get("body", "") or ""
    if not body.strip():
        return {}
    
    llm = make_llm()

    # System Prompt
    sys = SystemMessage(content=(
        "Lightly edit the following email for clarity and grammar. "
        "Preserve meaning and user's voice. Only return the improved body."
    ))

    # User Prompt
    user = HumanMessage(content=body)

    improved = llm.invoke([sys, user]).content.strip()
    return {"body": improved} if improved else {}

def decide_node(state: Dict) -> Dict:
    intent = state.get("intent", {}) or {}
    if bool(intent.get("approve", False)):
        return {"satisfied": True}
    return {}