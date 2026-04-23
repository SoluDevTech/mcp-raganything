from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from domain.ports.llm_port import LLMPort

_MESSAGE_TYPES = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


class LangchainOpenAIAdapter(LLMPort):
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float):
        self._llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    async def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = await self._llm.ainvoke(messages)
        return response.content

    async def generate_chat(self, messages: list[dict[str, str]]) -> str:
        lc_messages = [
            _MESSAGE_TYPES[msg["role"]](content=msg["content"])
            for msg in messages
            if msg["role"] in _MESSAGE_TYPES
        ]
        response = await self._llm.ainvoke(lc_messages)
        return response.content
