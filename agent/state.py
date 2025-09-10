from typing import TypedDict, List, Optional
#from pydantic import BaseModel, Field instead of TypedDict -> test later

class DraftState(TypedDict): # maybe add total=False
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
    satisfied: bool

def initial_state() -> DraftState:
    return DraftState(
        body="",
        subject="",
        to=[],
        cc=[],
        tone="neutral",
        transcript="",
        last_op=None,
        satisfied=False,
    )