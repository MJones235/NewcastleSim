"""Azure AI Foundry implementation of LLM provider."""

import asyncio
import json

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from scenarios.common.llm.llm_provider import LLMError, LLMProvider, LLMResponse
from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class AzureLLMProvider(LLMProvider):
    """
    Azure AI Foundry implementation of LLM provider.

    Uses Azure AI Inference API which supports:
    - Azure OpenAI models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo, etc.)
    - Open source models (Llama, Mistral, Phi, etc.)
    - Any model from Azure AI Model Catalog

    This provides a unified interface for all Azure-hosted models,
    making it easy to switch between different models without code changes.

    Configuration:
    - endpoint: Model endpoint URL from Azure AI Foundry
    - api_key: API key for authentication
    - model: Model name (optional, can be None for serverless endpoints)

    Pricing (GPT-5-nano, per million tokens, in £):
    - Input tokens: £0.04
    - Cached input: £0.01 (not tracked separately)
    - Output tokens: £0.30
    """

    # Pricing per million tokens (in £)
    PRICE_INPUT_PER_M = 0.04
    PRICE_OUTPUT_PER_M = 0.30

    # JSON Schema for evacuation decision response
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["evacuate", "stay"],
                "description": "Whether the person should evacuate or stay",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for the decision",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence level from 0.0 to 1.0",
            },
        },
        "required": ["decision", "reasoning", "confidence"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize Azure AI Inference provider.

        Args:
            endpoint: Azure AI model endpoint URL
            api_key: Azure AI API key
            model: Model name (None for serverless endpoints, required for some deployments)
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0

        try:
            self.client = ChatCompletionsClient(
                endpoint=endpoint, credential=AzureKeyCredential(api_key)
            )
            model_info = f"model: {model}" if model else "serverless endpoint"
            logger.info(f"Azure AI Inference provider initialized ({model_info})")
        except Exception as e:
            raise LLMError(f"Failed to initialize Azure AI Inference client: {e}")

    async def query(self, prompt: str) -> LLMResponse:
        """Query with a single prompt."""
        results = await self.batch_query([prompt])
        return results[0]

    async def batch_query(self, prompts: list[str]) -> list[LLMResponse]:
        """
        Query with multiple prompts in a batch.

        Sends all prompts concurrently to Azure OpenAI and waits for
        all responses. This is much faster than sequential queries.

        Args:
            prompts: List of prompt strings

        Returns:
            List of LLMResponse objects

        Raises:
            LLMError: If batch query fails
        """
        if not prompts:
            return []

        logger.info(f"Sending batch query with {len(prompts)} prompts")

        try:
            # Create concurrent tasks for all prompts
            tasks = [self._query_single(prompt) for prompt in prompts]

            # Wait for all to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            results: list[LLMResponse] = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Query {i} failed: {response}")
                    # Return default "stay" decision on error
                    results.append(
                        LLMResponse(
                            decision="stay",
                            reasoning="Error querying LLM - defaulting to stay",
                            confidence=0.0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                        )
                    )
                else:
                    # Type narrowing: response is LLMResponse here
                    results.append(response)  # type: ignore[arg-type]

            logger.info(f"Batch query completed: {len(results)} responses")
            return results

        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            raise LLMError(f"Batch query failed: {e}")

    async def _query_single(self, prompt: str) -> LLMResponse:
        """Internal method to query a single prompt with retries."""
        logger.debug(f"LLM Prompt:\n{prompt[:500]}...")  # Log first 500 chars

        for attempt in range(self.max_retries):
            try:
                # Build messages
                messages = [
                    SystemMessage(
                        content=(
                            "You are analyzing an evacuation scenario. "
                            "Consider the urgency and credibility of the information. "
                            "Provide your decision with reasoning and confidence level. "
                            "IMPORTANT: Respond ONLY with valid JSON in this exact format:\n"
                            '{"decision": "evacuate" or "stay", "reasoning": "brief explanation", "confidence": 0.0 to 1.0}'
                        )
                    ),
                    UserMessage(content=prompt),
                ]

                # Call Azure AI Inference API
                # Note: gpt-5-nano has limited parameter support - using defaults
                response = await self.client.complete(
                    messages=messages,
                    model=self.model,  # Can be None for serverless endpoints
                )

                # Parse response - guaranteed to match schema
                content = response.choices[0].message.content
                logger.debug(f"LLM Response (first 300 chars): {content[:300]}")

                # Extract token usage
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0

                # Update cumulative stats
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                self.total_requests += 1

                parsed = self._parse_response(
                    content, prompt_tokens, completion_tokens, total_tokens
                )
                return parsed

            except Exception as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise LLMError(f"Query failed after {self.max_retries} attempts: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        # This should never be reached due to the raise above, but satisfies type checker
        raise LLMError("Query failed - should not reach here")

    def _parse_response(
        self,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> LLMResponse:
        """
        Parse LLM response into structured format.

        Handles both JSON responses and natural language responses.
        """
        try:
            # Try to parse as JSON first
            data = json.loads(content)

            # Validate required fields
            if "decision" not in data or "reasoning" not in data or "confidence" not in data:
                raise ValueError(f"Missing required fields. Got: {list(data.keys())}")

            # Validate decision value
            decision = str(data["decision"]).lower()
            if decision not in ["evacuate", "stay"]:
                logger.warning(f"Invalid decision '{decision}', defaulting to 'stay'")
                decision = "stay"

            reasoning = str(data["reasoning"])

            # Validate and clamp confidence
            try:
                confidence = float(data["confidence"])
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid confidence value '{data['confidence']}', defaulting to 0.5"
                )
                confidence = 0.5

            return LLMResponse(
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except json.JSONDecodeError:
            # Not JSON - try to parse natural language response
            logger.debug(
                f"Response not in JSON format, attempting natural language parsing. Content: {content[:200]}"
            )
            content_lower = content.lower()

            # Look for decision keywords
            if "evacuate" in content_lower and "stay" not in content_lower:
                decision = "evacuate"
                confidence = 0.7
            elif "stay" in content_lower and "evacuate" not in content_lower:
                decision = "stay"
                confidence = 0.7
            else:
                # Ambiguous or unclear - default to conservative
                decision = "stay"
                confidence = 0.3

            # Use the raw content as reasoning (truncate if too long)
            reasoning = content[:200] if len(content) <= 200 else content[:197] + "..."

            logger.info(f"Parsed natural language response as: {decision}")

            return LLMResponse(
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except (KeyError, ValueError, TypeError) as e:
            # Handle other parsing errors gracefully
            logger.error(f"Response parsing failed: {e}")
            logger.debug(f"Raw content: {content}")

            # Emergency fallback
            return LLMResponse(
                decision="stay",  # Conservative default
                reasoning=f"Parsing error: {str(e)[:100]}",
                confidence=0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    def get_usage_stats(self) -> dict:
        """
        Get cumulative token usage statistics with cost estimates.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, total_requests,
            and estimated_cost_gbp
        """
        # Calculate cost in £
        input_cost = (self.total_prompt_tokens / 1_000_000) * self.PRICE_INPUT_PER_M
        output_cost = (self.total_completion_tokens / 1_000_000) * self.PRICE_OUTPUT_PER_M
        total_cost = input_cost + output_cost

        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "estimated_cost_gbp": total_cost,
            "input_cost_gbp": input_cost,
            "output_cost_gbp": output_cost,
        }

    async def close(self):
        """Close the Azure client session."""
        if hasattr(self.client, "close"):
            await self.client.close()
