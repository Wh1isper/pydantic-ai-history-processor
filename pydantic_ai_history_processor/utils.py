from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart


def fix_system_prompt(message_history: list[ModelMessage], system_prompt: str) -> list[ModelMessage]:
    if not message_history:
        return message_history

    message_history_without_system = []
    for msg in message_history:
        # Filter out system prompts
        if not isinstance(msg, ModelRequest):
            message_history_without_system.append(msg)
            continue
        message_history_without_system.append(
            ModelRequest(
                parts=[part for part in msg.parts if not isinstance(part, SystemPromptPart)],
                instructions=msg.instructions,
            )
        )
    if message_history_without_system and isinstance(message_history_without_system[0], ModelRequest):
        # inject system prompt
        message_history_without_system[0].parts.insert(0, SystemPromptPart(content=system_prompt))

    return message_history_without_system


def extract_system_prompts(message_history: list[ModelMessage]) -> list[str]:
    system_prompts = []
    for msg in message_history:
        if isinstance(msg, ModelRequest) and isinstance(msg.parts[0], SystemPromptPart):
            system_prompts.append(msg.parts[0].content)
    return system_prompts


def get_current_token_consumption(message_history: list[ModelMessage]) -> int | None:
    current_token_comsumption = None
    for msg in reversed(message_history):
        if isinstance(msg, ModelResponse) and msg.usage.total_tokens:
            current_token_comsumption = msg.usage.total_tokens
            break
    return current_token_comsumption
