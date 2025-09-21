from typing import Any, Optional

# Minimal local type stubs so Pylance recognizes these members.
# This file affects static analysis only; it does not change runtime behavior.

def configure(api_key: str, **kwargs: Any) -> None: ...

class GenerateContentResponse:
    text: Optional[str]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class GenerativeModel:
    model_name: str
    def __init__(self, model_name: str) -> None: ...
    def generate_content(self, content: Any, **kwargs: Any) -> GenerateContentResponse: ...
    def count_tokens(self, content: Any) -> Any: ...

__all__ = ["configure", "GenerativeModel"]