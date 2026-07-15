"""
LLM integration layer.

This module is responsible for all communication
with the language model.
"""

import requests

try:
    from .config import MODEL_NAME, MODEL_ENDPOINT, API_KEY
except ImportError:
    from config import MODEL_NAME, MODEL_ENDPOINT, API_KEY

from tools.tool import Tool


class LLMClient:
    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if "azure.com" in MODEL_ENDPOINT:
            headers["api-key"] = API_KEY
        else:
            headers["Authorization"] = f"Bearer {API_KEY}"
        return headers
    def _tool_to_dict(self, tool: Tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }

    def generate_response(self, messages, tools=None):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        }

        if tools:
            payload["tools"] = [self._tool_to_dict(t) for t in tools]

        endpoint = MODEL_ENDPOINT
        if endpoint.endswith("/openai/v1"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"

        print("Endpoint:", endpoint)

        response = requests.post(endpoint, json=payload, headers=self._headers())
        response.raise_for_status()

        data = response.json()
        if "message" in data:
            return data

        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message")
            if message is not None:
                return {"message": message, "raw": data}

        raise ValueError(f"Unexpected model response format: {data}")
