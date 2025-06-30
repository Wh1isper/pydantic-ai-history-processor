import os

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from pydantic_ai_history_processor.compactor import (
    CompactContext,
    CompactorProcessor,
)

MODEL_NAME = os.getenv("MODEL_NAME", "openai:gpt-4.1")

# Actually, I recommend gemini-2.5 family for 1M context window
COMPACTOR_MODEL_NAME = os.getenv("COMPACTOR_MODEL_NAME", "openai:gpt-4.1")


class AgentContext(CompactContext):
    """Add your agent context here."""


async def main():
    async for _ in stream_agent():
        pass


async def stream_agent():
    agent = Agent(
        model=MODEL_NAME,
        history_processors=[
            CompactorProcessor(
                model=COMPACTOR_MODEL_NAME,
                # I know gpt-4.1 has 1M context window, but for triggering compactor, I use 1000 only
                model_context_window=1000,
                compact_threshold=0.1,
            ).__call__,
        ],
        deps_type=AgentContext,
    )
    ctx = AgentContext()
    async with agent.iter(
        "Please just tell me a joke",
        # Now we mock the history for testing compactor
        message_history=[
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(
                parts=[TextPart(content="Hello! May I help you?")],
            ),
            ModelRequest(parts=[UserPromptPart(content="Got some request for you")]),
            ModelResponse(
                parts=[TextPart(content="Please tell me")],
                usage=Usage(total_tokens=200),
            ),
        ],
        deps=ctx,
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue

            elif Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as request_stream:
                    run.ctx.state.message_history = ctx.compacted_messages if ctx.compacted_messages else ...
                    run.ctx.state.usage += ctx.compactor_usage
                    async for event in request_stream:
                        yield event
    print(run.result.all_messages())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
