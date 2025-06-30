from pydantic_ai.messages import ModelMessage

from pydantic_ai_history_processor.utils import fix_system_prompt


class SystemPromptPatcher:
    """Patch the static system prompt in the message history.

    If you want dynamic system prompt, use `instructions` instead.
    See https://ai.pydantic.dev/agents/#instructions
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def __call__(self, message_history: list[ModelMessage]) -> list[ModelMessage]:
        fixed_message_history = fix_system_prompt(message_history, self.system_prompt)
        return fixed_message_history
