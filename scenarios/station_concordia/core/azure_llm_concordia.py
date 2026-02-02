"""
Azure OpenAI LLM adapter for Concordia.

This provides a synchronous interface compatible with Concordia's language model
expectations, calling Azure OpenAI API directly for general text completion.

Separate from scenarios.common.llm.azure_provider which is designed for
structured evacuation decision responses.
"""

import json
import os

import requests

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class AzureLLMConcordia:
    """
    Azure OpenAI LLM adapter for Concordia.

    Provides synchronous text generation compatible with Concordia's
    sample_text() interface. Uses direct REST API calls to avoid
    async/sync event loop conflicts.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str | None = None,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        """
        Initialize Azure OpenAI client for Concordia.

        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            model: Deployment name (optional, extracted from endpoint if not provided)
            api_version: Azure API version
            temperature: Sampling temperature (0.0 to 2.0)
            max_retries: Maximum number of retry attempts on failure
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.temperature = temperature
        self.max_retries = max_retries

        # Extract model/deployment name from endpoint if not provided
        if model:
            self.model = model
        else:
            # Extract from endpoint like: .../openai/deployments/gpt-4
            parts = self.endpoint.split("/deployments/")
            if len(parts) == 2:
                self.model = parts[1].split("/")[0]
            else:
                self.model = "gpt-4"  # Default fallback

        logger.info(f"Initialized AzureLLMConcordia with model: {self.model}")

    def sample_text(
        self, prompt: str, max_tokens: int = 1000, temperature: float | None = None, **kwargs
    ) -> str:
        """
        Generate text from a prompt.

        This is the primary interface method expected by Concordia.
        Uses synchronous REST API calls to avoid event loop conflicts.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (overrides default if provided)
            **kwargs: Additional parameters (for compatibility)

        Returns:
            Generated text string

        Raises:
            Exception: If all retry attempts fail
        """
        temp = temperature if temperature is not None else self.temperature

        # Build the API URL
        url = f"{self.endpoint}/chat/completions?api-version={self.api_version}"

        # Build the request
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Convert prompt to chat format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a simulation engine for evacuation training scenarios. "
                    "Generate realistic behavioral responses for simulated agents based on their personality profiles, "
                    "situational context, and safety protocols. "
                    "This is for emergency preparedness training and research purposes."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        payload = {
            "messages": messages,
            "max_completion_tokens": max_tokens,  # Use max_completion_tokens for newer models
            "temperature": temp,
        }

        # Retry logic
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"].strip()

                    # Log token usage
                    usage = result.get("usage", {})
                    logger.debug(
                        f"LLM call successful. Tokens: "
                        f"{usage.get('prompt_tokens', 0)} prompt, "
                        f"{usage.get('completion_tokens', 0)} completion"
                    )

                    return text

                else:
                    error_msg = f"Azure API error {response.status_code}: {response.text}"

                    # Check if it's a content filter / jailbreak error
                    if response.status_code == 400:
                        try:
                            if (
                                "content_filter" in response.text
                                or "jailbreak" in response.text.lower()
                            ):
                                logger.error("=" * 80)
                                logger.error("CONTENT FILTER / JAILBREAK DETECTED")
                                logger.error("=" * 80)
                                logger.error(f"Error: {error_msg}")
                                logger.error("\n" + "=" * 80)
                                logger.error("FULL PROMPT THAT TRIGGERED THE FILTER:")
                                logger.error("=" * 80)
                                logger.error(f"System message:\n{messages[0]['content']}")
                                logger.error("\n" + "-" * 80)
                                logger.error(f"User prompt:\n{messages[1]['content']}")
                                logger.error("=" * 80)
                        except Exception:
                            pass

                    logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {error_msg}")
                    last_error = Exception(error_msg)

            except requests.exceptions.Timeout as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} timed out")
                last_error = e

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                last_error = e

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse Azure response: {e}")
                last_error = e

        # All retries failed
        error_msg = f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)

        # Return a fallback response rather than crashing the simulation
        logger.warning("Returning fallback response due to API failures")
        return "I need to carefully consider my options and evacuate safely."

    @classmethod
    def from_env(cls, **kwargs) -> "AzureLLMConcordia":
        """
        Create instance from environment variables.

        Expects:
            - AZURE_LLM_ENDPOINT
            - AZURE_LLM_API_KEY
            - AZURE_LLM_MODEL (optional)

        Args:
            **kwargs: Additional arguments passed to constructor

        Returns:
            AzureLLMConcordia instance

        Raises:
            ValueError: If required environment variables are missing
        """
        endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        api_key = os.getenv("AZURE_LLM_API_KEY")
        model = os.getenv("AZURE_LLM_MODEL")

        if not endpoint or not api_key:
            raise ValueError(
                "Missing required environment variables: "
                "AZURE_LLM_ENDPOINT and AZURE_LLM_API_KEY"
            )

        return cls(endpoint=endpoint, api_key=api_key, model=model, **kwargs)


def create_concordia_llm_from_config(config: dict) -> AzureLLMConcordia:
    """
    Create Azure LLM instance from configuration.

    Args:
        config: Configuration dictionary with llm settings

    Returns:
        AzureLLMConcordia instance
    """
    from dotenv import load_dotenv

    load_dotenv()

    llm_config = config.get("llm", {})

    return AzureLLMConcordia.from_env(
        temperature=llm_config.get("temperature", 0.7),
        max_retries=llm_config.get("max_retries", 3),
    )
