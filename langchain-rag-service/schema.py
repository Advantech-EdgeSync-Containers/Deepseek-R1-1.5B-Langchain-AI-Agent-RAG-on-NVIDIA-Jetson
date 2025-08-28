import json
import re
from typing import List, Literal, Optional, Union
from pydantic import BaseModel


# ---- Request and Response Models
class Message(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
