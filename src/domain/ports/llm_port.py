"""Abstract port for LLM operations (classical RAG)."""

from abc import ABC, abstractmethod


class LLMPort(ABC):
    """Port interface for LLM operations (multi-query generation + judge scoring)."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a text response from the LLM.

        Args:
            system_prompt: The system prompt to use.
            user_message: The user message to send.

        Returns:
            The generated text response.
        """
        pass

    @abstractmethod
    async def generate_chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The generated text response.
        """
        pass
