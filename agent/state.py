from typing import TypedDict, List, Optional, Dict, Any

class DraftState(TypedDict):
    """Central, minimal state for the mail draft.
    - body: current email text
    - subject: subject line
    - to, cc: recipient lists
    - tone: style hint ("neutral" | "friendly" | "formal")
    - transcript: last input (simulated STT)
    - last_op: most recently detected operation (debug/transparency)
    - satisfied: confirmed by the user?
    """
    body: str
    subject: str
    to: List[str]
    cc: List[str]
    tone: str
    transcript: str
    last_op: Optional[str]
    done: bool
    intent: Dict[str, Any]

def initial_state() -> DraftState:
    return DraftState(
        body="",
        subject="",
        to=[],
        cc=[],
        tone="neutral",
        transcript="",
        last_op=None,
        done=False,
        intent={}
    )