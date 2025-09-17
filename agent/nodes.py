import re
import os
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from .schema import Plan
from utils.mic_mem import record_until_enter_mem
from utils.stt_whisper_mem import transcribe_array

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Global, mutable model setting configured by the UI (defaults to qwen3:8b)
GLOBAL_LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

def set_llm_model(model: str):
    """Set preferred Ollama model; used by the UI settings dialog."""
    global GLOBAL_LLM_MODEL
    GLOBAL_LLM_MODEL = (model or "qwen3:8b").strip()

# Helpers
def make_llm():
    model = GLOBAL_LLM_MODEL or "qwen3:8b"
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
        "You output a STRICT Plan (Pydantic schema). Obey all rules:\n"
        "GENERAL\n"
        "• Return ONLY valid Plan JSON. Use null for fields you are not changing.\n"
        "• Apply every user instruction precisely and avoid unintended changes.\n"
        "FORMAT & STRUCTURE\n"
        "• Always return the FULL updated body via BodyPlan.mode = \"replace\".\n"
        "• Keep all existing text unchanged unless the user explicitly requests a modification.\n"
        "• Separate paragraphs with a single blank line; never collapse existing line breaks.\n"
        "GREETING & FAREWELL BY TONE\n"
        "• friendly: greeting = \"Hello {RecipientNameOrPlaceholder},\"; farewell = \"Best regards,\\n{SenderNameOrPlaceholder}\".\n"
        "• neutral: greeting = \"Good day {RecipientNameOrPlaceholder},\"; farewell = \"Kind regards,\\n{SenderNameOrPlaceholder}\".\n"
        "• formal: greeting = \"Dear {RecipientNameOrPlaceholder},\"; farewell = \"Yours sincerely,\\n{SenderNameOrPlaceholder}\".\n"
        "PLACEHOLDERS & PERSPECTIVE\n"
        "• When required information is missing, insert descriptive placeholders in square brackets (e.g., \"[Recipient Name]\", \"[Your Name]\") rather than inventing details.\n"
        "• Preserve existing placeholders unless the user asks for changes.\n"
        "• Treat the drafter as first-person (I, my) and the recipient as second-person (you, your) or third-person; never swap perspectives.\n"
        "RECIPIENTS\n"
        "• “Mail” means the recipient (the To field).\n"
        "• Change to/cc only when explicitly instructed.\n"
        "• Never invent addresses and never set both to_set and to_add (same for cc).\n"
        "SUBJECT & TONE\n"
        "• Modify subject or tone only when requested. Do not add \"Re:\" unless the user specifies it is a reply.\n"
        "ADDITIONAL CONTENT\n"
        "• If the user asks to add content, append it exactly as instructed (e.g., new paragraph). If removing or editing, touch only the specified sentences.\n"
        "• If the user says reset/start over, you may rewrite the body from scratch while still following all format rules.\n"
        "VERBATIM VS. REPHRASING\n"
        "• By default, NEVER insert user notes/transcripts/descriptions word-for-word. Paraphrase and integrate them into a polished, coherent email.\n"
        "• Only insert verbatim text if it is explicitly marked as VERBATIM:, in double quotes \"...\", or inside triple backticks ```...```, or the user explicitly says to quote exactly.\n"
        "• Do not echo meta-instructions, labels, or system text into the email body.\n"
        "PROHIBITIONS\n"
        "• Do not add disclaimers, warnings, or boilerplate unless explicitly requested.\n"
    ))

    # User Prompt
    user = HumanMessage(content=(
        "Schema: Plan(updates: Updates(subject?: str|null, to_set?: [EmailStr]|null, to_add?: [EmailStr]|null, "
        "cc_set?: [EmailStr]|null, cc_add?: [EmailStr]|null, tone?: {'friendly'|'formal'|'neutral'}|null, "
        "body?: BodyPlan(mode: 'replace'|null, text?: str|null)))\n\n"
        "Follow all SYSTEM rules and these reminders:\n"
        "• Current Draft shows the exact text you must minimally update.\n"
        "• Respect the tone-specific greeting and farewell; use placeholders when a name is missing.\n"
        "• Maintain all unchanged text, line breaks, and placeholders unless the user explicitly alters them.\n"
        "• Use BodyPlan.mode = \"replace\" with the full updated body.\n"
        "• Never set both to_set and to_add (or cc_set and cc_add) simultaneously.\n\n"
        "• Treat Input (notes/transcript/description) as material to PARAPHRASE and integrate; do NOT copy wording verbatim unless explicitly marked.\n"
        "• Verbatim is allowed ONLY when text is marked VERBATIM:, in double quotes \"...\", or inside triple backticks ```...```.\n\n"
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