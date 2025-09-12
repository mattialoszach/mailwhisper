from typing import List, Optional, Literal
from pydantic import BaseModel

Tone = Literal["friendly", "formal", "neutral"]
BodyMode = Literal["replace"] # For now only replace (experiment with other modes later)

class BodyPlan(BaseModel):
    mode: Optional[BodyMode]
    text: Optional[str]

class Updates(BaseModel):
    body: Optional[BodyPlan]
    subject: Optional[str]
    to_set: Optional[List[str]]
    to_add: Optional[List[str]]
    cc_set: Optional[List[str]]
    cc_add: Optional[List[str]]
    tone: Optional[Tone]

class Plan(BaseModel):
    updates: Updates