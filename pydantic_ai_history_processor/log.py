import os

USER_DEFINED_LOG_LEVEL = os.getenv("PYDANTIC_AI_HISTORY_PROCESSOR_LOG_LEVEL", "INFO")

os.environ["LOGURU_LEVEL"] = USER_DEFINED_LOG_LEVEL

from loguru import logger  # noqa: E402

__all__ = ["logger"]
