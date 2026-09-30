from dataclasses import dataclass

import requests

from core.model_selector import ModelChoice, select_model

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


@dataclass
class OllamaClient:
    base_url: str = DEFAULT_OLLAMA_URL
    timeout: int = 180

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", []) if model.get("name")]
        except (requests.RequestException, ValueError, KeyError) as error:
            raise OllamaError("Ollama is not responding. Start Ollama and try again.") from error

    def resolve_model(self) -> ModelChoice:
        return select_model(self.list_models())

    def chat(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
            if not answer:
                raise OllamaError("Ollama returned an empty response.")
            return answer
        except requests.Timeout as error:
            raise OllamaError("The local model timed out. Try a smaller model or a shorter question.") from error
        except requests.RequestException as error:
            raise OllamaError("Ollama could not generate a response. Check that the selected model is available.") from error
        except ValueError as error:
            raise OllamaError("Ollama returned an unreadable response.") from error
