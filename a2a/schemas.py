from pydantic import BaseModel
from typing import Any, Dict
import uuid

class A2ATask(BaseModel):
    task_id: str
    sender: str
    recipient: str
    capability: str
    input: Dict[str, Any]

    @staticmethod
    def create(sender, recipient, capability, input):
        return A2ATask(
            task_id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            capability=capability,
            input=input
        )

class A2AResponse(BaseModel):
    task_id: str
    sender: str
    status: str
    output: Dict[str, Any]
