"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM query."""

    decision: str  # "evacuate" or "stay"
    reasoning: str  # Explanation of decision
    confidence: float  # 0.0 to 1.0
    prompt_tokens: int = 0  # Tokens in prompt
    completion_tokens: int = 0  # Tokens in completion
    total_tokens: int = 0  # Total tokens used

    # Phase 2: Communication actions
    broadcast_message: str | None = None  # Message to broadcast
    broadcast_radius: float | None = None  # Radius in meters


class LLMProvider(ABC):
    """
    Abstract base for LLM providers.

    Provides interface for querying language models with support for
    batch processing. Implementations handle specific providers
    (Azure, local, etc.).
    """

    @abstractmethod
    async def query(self, prompt: str) -> LLMResponse:
        """
        Query LLM with a single prompt.

        Args:
            prompt: The prompt string to send to the LLM

        Returns:
            LLMResponse with decision, reasoning, and confidence

        Raises:
            LLMError: If query fails
        """
        pass

    @abstractmethod
    async def batch_query(self, prompts: list[str]) -> list[LLMResponse]:
        """
        Query LLM with multiple prompts in a batch.

        This is the critical optimization - sends all prompts in one
        API call rather than making individual calls.

        Args:
            prompts: List of prompt strings

        Returns:
            List of LLMResponse objects (same order as prompts)

        Raises:
            LLMError: If batch query fails
        """
        pass


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass
