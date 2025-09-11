from typing import List, Optional, Literal
from pydantic import BaseModel, EmailStr

Tone = Literal["friendly", "formal", "neutral"]
BodyMode = Literal["replace", "append"]

class BodyPlan(BaseModel):
    mode: Optional[BodyMode] #= None?
    text: Optional[str]

class Updates(BaseModel):
    body: Optional[BodyPlan]
    subject: Optional[str]
    to_set: Optional[List[EmailStr]]
    to_add: Optional[List[EmailStr]]
    cc_set: Optional[List[EmailStr]]
    cc_add: Optional[List[EmailStr]]
    tone: Optional[Tone]

class Plan(BaseModel):
    approve: bool
    update: Updates