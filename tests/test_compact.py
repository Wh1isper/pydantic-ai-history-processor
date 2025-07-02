from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from pydantic_ai_history_processor.compactor import (
    K_TOKENS,
    K_TOKENS_1000,
    CompactorProcessor,
    CompactStrategy,
)


@pytest.fixture
def compactor():
    return CompactorProcessor(
        model="test",
        model_settings={"max_tokens": 32 * K_TOKENS},
        model_context_window=200 * K_TOKENS_1000,
    )


def test_compactor_need_compact(compactor: CompactorProcessor):
    # Default value
    assert compactor.need_compact([ModelResponse(parts=[], usage=Usage(total_tokens=200 * K_TOKENS_1000))])
    assert not compactor.need_compact([ModelResponse(parts=[], usage=Usage(total_tokens=1))])

    # Reached threshold
    assert compactor.need_compact(
        [ModelResponse(parts=[], usage=Usage(total_tokens=50 * K_TOKENS_1000))],
        threshold=0.1,
    )

    # Not reached threshold
    assert not compactor.need_compact(
        [ModelResponse(parts=[], usage=Usage(total_tokens=120 * K_TOKENS_1000))],
        threshold=0.9,
    )

    # Overflow
    assert compactor.need_compact(
        [ModelResponse(parts=[], usage=Usage(total_tokens=180 * K_TOKENS_1000))],
        threshold=0.9,
    )


@pytest.mark.parametrize(
    "messages, compact_strategy, expected_length",
    [
        (
            [
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="World")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="New message")]),
            ],
            CompactStrategy.none,
            (4, 1),
        ),
        (
            [
                ModelRequest(parts=[UserPromptPart(content="我在")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="北京")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="World")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="New message")]),
            ],
            CompactStrategy.last_two,
            (4, 5),
        ),
        (
            [
                ModelRequest(parts=[UserPromptPart(content="我在")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="北京")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="World")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="New message")]),
            ],
            CompactStrategy.in_conversation,
            (9, 1),
        ),
        (
            [
                ModelRequest(parts=[UserPromptPart(content="我在")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="北京")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="World")]),
                ModelResponse(parts=[]),
                ModelRequest(parts=[UserPromptPart(content="New message")]),
                ModelResponse(parts=[ToolCallPart(tool_name="foo", tool_call_id="1")]),
                ModelRequest(parts=[ToolReturnPart(tool_name="foo", content="", tool_call_id="1")]),
                ModelResponse(parts=[ToolCallPart(tool_name="bar", tool_call_id="2")]),
                ModelRequest(parts=[ToolReturnPart(tool_name="bar", content="", tool_call_id="2")]),
            ],
            CompactStrategy.in_conversation,
            (13, 3),
        ),
    ],
)
def test_compactor_split_messages(
    compactor: CompactorProcessor,
    messages: list[ModelMessage],
    compact_strategy: CompactStrategy,
    expected_length: tuple[int, int],
):
    history_messages_length, keep_messages_length = expected_length
    history_messages, keep_messages = compactor.split_history(
        messages,
        compact_strategy=compact_strategy,
    )
    assert len(history_messages) == history_messages_length
    assert len(keep_messages) == keep_messages_length
