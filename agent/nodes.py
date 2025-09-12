import re
import os
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from .schema import Plan
from utils.mic_mem import record_until_enter_mem
from utils.stt_whisper_mem import transcribe_array

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Helpers
def make_llm():
    model = "qwen3:8b"
    return ChatOllama(model=model, temperature=0.15, num_ctx=8192)

def make_structured_llm() -> Any:
    return make_llm().with_structured_output(Plan)

# Nodes
def transcribe_node(state: Dict) -> Dict:
    """Local STT directly in memory (no files)."""
    print("\n🎤 Recording...")
    print("   - Press ENTER to stop recording.")
    
    try:
        audio = record_until_enter_mem(samplerate=16000)
        text, detected_lang = transcribe_array(audio, samplerate=16000)
        if text:
            print(f"📝 Transcript ({detected_lang or 'auto'}): {text}")
            return {"transcript": text}
        else:
            print("⚠️ No transcription detected. Please try again or type.")
            user_text = input("> ").strip()
            return {"transcript": user_text}
        
    except KeyboardInterrupt:
        print("\n⏹️  Terminated. No recording.")
        return {"transcript": ""}
    except Exception as e:
        print(f"⚠️  STT-Error (offline): {e}")
        print("\n⌨️  Fallback: Please use Keyboard:")
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
        "You output a STRICT Plan (Pydantic schema). Rules:\n"
        "BODY\n"
        "• Always return a FULL updated body (replace) derived from CURRENT body.\n"
        "• Apply ONLY the specific changes the user asked for; keep everything else EXACTLY the same.\n"
        "• Do NOT add disclaimers, warnings, signatures, or boilerplate unless explicitly requested.\n"
        "• If the user says “reset / start over / rewrite completely”, you may discard the current body.\n"
        "RECIPIENTS\n"
        "• Change to/cc ONLY if user explicitly provides emails or uses “to:”/“send to …” or “cc:”/“add cc …”.\n"
        "• NEVER infer emails from names. NEVER set both to_set and to_add (same for cc).\n"
        "SUBJECT/TONE\n"
        "• Modify only if explicitly requested. Do NOT prefix “Re:” unless the user says it is a reply.\n"
        "NAME\n"
        "• If the user states their name, include it in the signature and replace “[Your Name]”.\n"
        "OUTPUT\n"
        "• Return ONLY the Plan JSON. Use null for fields you are NOT changing."
    ))

    # User Prompt
    user = HumanMessage(content=(
        "Schema: Plan(updates: Updates("
        "subject?: str|null, to_set?: [EmailStr]|null, to_add?: [EmailStr]|null, "
        "cc_set?: [EmailStr]|null, cc_add?: [EmailStr]|null, tone?: {'friendly'|'formal'|'neutral'}|null, "
        "body?: BodyPlan(mode: 'replace'|null, text?: str|null)"
        "))\n\n"
        "Apply requested changes MINIMALLY to the CURRENT draft; return the FULL updated body as 'replace'.\n"
        "If the user says 'reset/start over/rewrite completely', you may produce a fresh body.\n"
        "If no recipient change is requested, set to_* and cc_* to null.\n"
        "NEVER set both to_set and to_add (or cc_set and cc_add) at the same time.\n"
        "Do NOT introduce any disclaimers or boilerplate unless asked.\n\n"
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

    def clean_emails(lst):
        if not isinstance(lst, list):
            return []
        cleaned = []
        for e in lst:
            if isinstance(e, str):
                s = e.strip().lower()
                if EMAIL_RE.fullmatch(s):
                    cleaned.append(s)
        # dedupe preserving order
        return list(dict.fromkeys(cleaned))

    if isinstance(to_add, list) and to_add:
        cleaned = clean_emails(curr_to + to_add)
        if cleaned:
            out["to"] = cleaned
    elif isinstance(to_set, list):
        cleaned = clean_emails(to_set)
        if cleaned:
            out["to"] = cleaned
    
    if isinstance(cc_add, list) and cc_add:
        cleaned = clean_emails(curr_cc + cc_add)
        if cleaned:
            out["cc"] = cleaned
    elif isinstance(cc_set, list):
        cleaned = clean_emails(cc_set)
        if cleaned:
            out["cc"] = cleaned

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
    
    return out

def decide_node(state: Dict) -> Dict:
    """MVP: CLI/UX-Decision: 1=continue editing, 2=finished. Needs to be updated for UI."""
    print("\nAction: [1] continue   [2] exit")
    choice = input("Choose 1 or 2: ").strip()
    return {"done": choice == "2"}