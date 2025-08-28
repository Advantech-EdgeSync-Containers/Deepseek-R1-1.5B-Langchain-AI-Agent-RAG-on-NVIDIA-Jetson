import json
import re
import uuid
from typing import Optional, Union


class ChatTokenUtils():
    def remove_think_block(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def extract_json_key(code_block: str, key: str) -> Optional[str]:
        """
        Extract the value of a specified key from a markdown-wrapped JSON code block.
        """
        try:
            # Remove triple backticks and optional language specifier like ```json
            cleaned = re.sub(r"```[a-zA-Z]*", "", code_block).replace("```", "").strip()
            parsed = json.loads(cleaned)
            return parsed.get(key)
        except (json.JSONDecodeError, TypeError):
            return None

    def build_chat_response(content: Union[str, dict], model: str, role: str = "assistant", is_json: bool = False):
        content_str = json.dumps(content) if is_json else content

        response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "message": {
                        "role": role,
                        "content": content_str
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }

        return response