from typing import Optional
from mistralai import Mistral
from config import MISTRAL_API_KEY, DEFAULT_MODEL, TEMPERATURE  

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None):
        self.api_key = api_key or MISTRAL_API_KEY
        self.model = model or DEFAULT_MODEL
        self.temperature = TEMPERATURE if temperature is None else temperature
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is missing. Set it in your environment.")
        self.client = Mistral(api_key=self.api_key)

    def chat(self, messages: list[dict], model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        model = model or self.model
        temperature = self.temperature if temperature is None else temperature
        resp = self.client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
