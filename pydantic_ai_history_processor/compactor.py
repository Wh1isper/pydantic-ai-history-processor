import enum
import traceback
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, ToolOutput
from pydantic_ai.agent import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage

from pydantic_ai_history_processor.log import logger
from pydantic_ai_history_processor.utils import (
    extract_system_prompts,
    fix_system_prompt,
    get_current_token_consumption,
)

_HERE = Path(__file__).parent
SYSTEM_PROMPT = (_HERE / "compactor_system_prompt.md").read_text()

K_TOKENS_1000 = 1000
K_TOKENS = 1024


class CompactContext(BaseModel):
    compacted_messages: list[ModelMessage] | None = None
    compactor_usage: Usage = Field(default_factory=Usage)


class CondenseResult(BaseModel):
    analysis: str = Field(
        ...,
        description="""A summary of the conversation so far, capturing technical details, code patterns, and architectural decisions.""",
    )
    context: str = Field(
        ...,
        description="""The context to continue the conversation with. If applicable based on the current task, this should include:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
5. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
6. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
7. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
8. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.
""",
    )


class Feature(str, enum.Enum):
    refine_prompt = "refine_prompt"


class CompactStrategy(str, enum.Enum):
    in_conversation = "in_conversation"
    """Compact all message, including this round conversation"""

    none = "none"
    """Compact all previous messages"""

    last_two = "last_two"
    """Keeping the last two previous messages"""


class CompactorProcessor:
    def __init__(
        self,
        model: Model | KnownModelName,
        model_settings: ModelSettings | None = None,
        compactor_model_settings: ModelSettings | None = None,
        model_context_window: int = 200 * K_TOKENS_1000,
        compact_threshold: float = 0.5,
        in_conversation_compact_threshold: float = 0.8,
        system_prompt: str | None = None,
        *,
        compact_agent: Agent = None,
    ):
        self.model_context_window = model_context_window
        self.model_settings = model_settings or {}
        self.compact_threshold = compact_threshold
        self.in_conversation_compact_threshold = in_conversation_compact_threshold

        self.compact_strategy = CompactStrategy.last_two
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        if compact_agent:
            self.agent = compact_agent
        else:
            self.agent: Agent[None, CondenseResult] = Agent(
                model=model,
                model_settings=compactor_model_settings,
                system_prompt=self.system_prompt,
                output_type=ToolOutput(
                    type_=CondenseResult,
                    name="condense",
                    description="""
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions. This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing with the conversation and supporting any continuing tasks.
The user will be presented with a preview of your generated summary and can choose to use it to compact their context window or keep chatting in the current conversation.
Users may refer to this tool as 'smol' or 'compact' as well. You should consider these to be equivalent to 'condense' when used in a similar context.
""",
                    max_retries=5,
                ),
                retries=3,
            )

    def _split_history(
        self,
        message_history: list[ModelMessage],
        n: int,
    ) -> tuple[list[ModelMessage], list[ModelMessage]]:
        """
        Returns a tuple of (history, keep_messages)
        """
        if not n:
            # No keep messages
            return message_history, []

        user_prompt_indices = []
        for i, msg in enumerate(message_history):
            if not isinstance(msg, ModelRequest):
                continue
            if any(isinstance(p, UserPromptPart) for p in msg.parts) and not any(
                isinstance(p, ToolReturnPart) for p in msg.parts
            ):
                user_prompt_indices.append(i)
        if not user_prompt_indices:
            # No user prompt in history, keep all
            return [], message_history

        if len(user_prompt_indices) < n:
            # No enough history to keep
            return [], message_history
        return (
            message_history[: user_prompt_indices[-n]],
            message_history[user_prompt_indices[-n] :],
        )

    def need_compact(self, message_history: list[ModelMessage], threshold: float | None = None) -> bool:
        current_token_comsumption = get_current_token_consumption(message_history)

        token_threshold = (threshold or self.compact_threshold) * self.model_context_window
        will_overflow = (current_token_comsumption or 0) + self.model_settings.get(
            "max_tokens", 0
        ) >= self.model_context_window

        return (current_token_comsumption and current_token_comsumption >= token_threshold) or will_overflow

    def split_history(
        self,
        message_history: list[ModelMessage],
        compact_strategy: CompactStrategy | None = None,
    ) -> tuple[list[ModelMessage], list[ModelMessage]]:
        compact_strategy = compact_strategy or self.compact_strategy
        match compact_strategy:
            case CompactStrategy.none:
                # Only current 1
                history_messages, keep_messages = self._split_history(message_history, 1)
            case CompactStrategy.last_two:
                # Previous 2 + current 1
                history_messages, keep_messages = self._split_history(message_history, 3)
            case CompactStrategy.in_conversation:
                history_messages, keep_messages = self._split_history(message_history, 0)
            case _:
                raise NotImplementedError(f"Compact strategy {self.model_config.compact_strategy} not implemented")

        return history_messages, keep_messages

    async def __call__(
        self, ctx: RunContext[CompactContext], message_history: list[ModelMessage]
    ) -> list[ModelMessage]:
        try:
            message_history = await self._compact(ctx, message_history)
        except Exception as e:
            logger.error(f"Failed to compact history: {e} {traceback.format_exc()}")

        ctx.deps.compacted_messages = message_history
        return message_history

    async def _compact(
        self, ctx: RunContext[CompactContext], message_history: list[ModelMessage]
    ) -> list[ModelMessage]:
        ctx = ctx.deps
        if not self.need_compact(message_history):
            logger.info("No need to compact history.")
            return message_history
        original_system_prompts = extract_system_prompts(message_history)
        logger.info("Splitting history for compaction...")
        history_messages, keep_messages = self.split_history(message_history)
        if len(history_messages) <= 2:
            logger.info("No enough history to compact, try compacting all history.")
            history_messages, keep_messages = self.split_history(message_history, CompactStrategy.none)
            if len(history_messages) <= 2:
                if self.need_compact(message_history, 0.8):
                    logger.info("No enough history to compact, try compacting in conversation.")
                    history_messages, keep_messages = self.split_history(
                        message_history, CompactStrategy.in_conversation
                    )
                else:
                    logger.info("Already compacted all history, skipping.")
                    return keep_messages
        if not history_messages:
            logger.info("No history to compact, keeping all messages.")
            return keep_messages
        logger.info(
            f"Compacting history... {len(message_history)}({len(history_messages)}, {len(keep_messages)}) -> {len(keep_messages)}"
        )
        result = await self.agent.run(
            "The user has accepted the condensed conversation summary you generated. Use `condense` to generate a summary and context of the conversation so far. "
            "This summary covers important details of the historical conversation with the user which has been truncated. "
            "It's crucial that you respond by ONLY asking the user what you should work on next. "
            "You should NOT take any initiative or make any assumptions about continuing with work. "
            "Keep this response CONCISE and wrap your analysis in <analysis> and <context> tags to organize your thoughts and ensure you've covered all necessary points. ",
            message_history=fix_system_prompt(history_messages, self.system_prompt),
        )
        ctx.compactor_usage += result.usage()

        summary_prompt = f"""Condensed conversation summary(not in the history):
<condense>
<analysis>
{result.output.analysis}
</analysis>

<context>
{result.output.context}
</context>
</condense>
"""
        logger.info(
            f"""
{summary_prompt}

compact token usage: {result.usage()}
"""
        )

        return [
            ModelRequest(
                parts=[
                    *[SystemPromptPart(content=p) for p in original_system_prompts],
                    UserPromptPart(content="Please summary the conversation"),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=summary_prompt)],
            ),
            *keep_messages,
        ]
