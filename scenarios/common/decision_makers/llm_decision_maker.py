"""LLM-based decision maker for complex evacuation decisions."""

from typing import Any

from scenarios.base.decision_maker_base import Decision, DecisionMakerBase
from scenarios.common.llm import AzureLLMProvider, EvacuationPromptBuilder
from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class LLMDecisionMaker(DecisionMakerBase):
    """
    Decision maker that uses LLM for evacuation decisions.

    Only queries LLM when agent receives messages. For routine decisions
    (like continuing to wait), uses fast heuristics.

    This is designed for event-driven decisions, not continuous updates.
    """

    # Shared LLM provider instance (reused across all agents)
    _llm_provider: AzureLLMProvider | None = None

    @classmethod
    def initialize_llm(
        cls,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the shared LLM provider.

        Call this once at simulation start before creating agents.

        Args:
            endpoint: Azure AI model endpoint URL
            api_key: Azure AI API key
            model: Model name (optional - can be None for serverless endpoints)
        """
        if cls._llm_provider is None:
            cls._llm_provider = AzureLLMProvider(endpoint=endpoint, api_key=api_key, model=model)
            logger.info("LLM decision maker initialized")

    def __init__(self, agent: Any):
        """Initialize decision maker for an agent."""
        super().__init__(config=None)
        self.agent = agent
        self.pending_messages: list[dict | str] = []
        self.last_decision: Decision = Decision.IGNORE
        self.last_reasoning: str = ""

        # Message history tracking (Phase 1)
        self.message_history: list[dict[str, Any]] = []

        # Callback for structured message logging
        self._message_callback = None

        if self._llm_provider is None:
            logger.warning(
                "LLM provider not initialized - call LLMDecisionMaker.initialize_llm() first"
            )

    def set_message_callback(self, callback):
        """Set callback for logging messages: callback(agent_id, timestamp, message, from_agent)"""
        self._message_callback = callback

    def make_decision(
        self, message: str, agent_state: dict[str, Any], context: dict[str, Any]
    ) -> Decision:
        """
        Make a decision based on message (synchronous interface).

        For LLM decisions, this queues the message and returns IGNORE.
        Actual decisions are made in async batch processing.

        Args:
            message: The message received
            agent_state: Current agent state
            context: Simulation context

        Returns:
            Decision.IGNORE (actual decision made in batch processing)
        """
        # Extract sender from message if it's from another agent
        sender = "system"
        if message.startswith("[Agent "):
            # Format: "[Agent agent_5]: message"
            try:
                sender = message.split("]")[0].replace("[Agent ", "")
            except:
                pass

        # Track message with metadata
        sim_time = context.get("time", 0)
        self.message_history.append(
            {"time": sim_time, "type": "received", "from": sender, "message": message}
        )

        # Call structured logging callback if set
        if self._message_callback and self.agent:
            self._message_callback(self.agent.id, sim_time, message, sender)

        # Queue message for batch processing
        self.pending_messages.append(message)
        return Decision.IGNORE

    def get_decision_reasoning(self) -> str:
        """Get explanation for the last decision made."""
        return self.last_reasoning

    def decide(self, timestamp: float) -> dict:
        """
        Make decision for current timestep.

        Only queries LLM if there are pending messages to process.
        Otherwise returns quick "continue" decision.

        Args:
            timestamp: Current simulation time

        Returns:
            Decision dict with action
        """
        # If no pending messages, just continue current behavior
        if not self.pending_messages:
            return {"action": "continue"}

        # Has messages - need LLM decision (but process async in batch)
        # For now, mark as pending and process in batch
        return {"action": "pending_llm"}

    def on_message_received(self, message: dict | str):
        """
        Called when agent receives a message.

        Queues message for LLM processing.

        Args:
            message: The received message
        """
        # Avoid adding duplicate consecutive messages
        if self.pending_messages and self.pending_messages[-1] == message:
            logger.debug(f"Agent {self.agent.id} received duplicate message, ignoring")
            return

        self.pending_messages.append(message)
        logger.debug(f"Agent {self.agent.id} queued message for LLM processing")

    async def process_with_llm(self) -> dict:
        """
        Process pending messages with LLM.

        Returns:
            Decision dict with evacuation decision
        """
        if not self.pending_messages or self._llm_provider is None:
            return {"action": "continue"}

        # Consolidate multiple messages if present
        if len(self.pending_messages) == 1:
            message = self.pending_messages[0]
        else:
            # Summarize multiple messages
            unique_messages = list(dict.fromkeys(self.pending_messages))  # Remove duplicates
            if len(unique_messages) == 1:
                message = unique_messages[0]
            else:
                # Convert to strings before joining
                message_strs = [str(msg) for msg in unique_messages[-3:]]
                message = f"Multiple people nearby are saying: {'; '.join(message_strs)}"  # Last 3 unique messages

        # Build prompt
        prompt = EvacuationPromptBuilder.build_evacuation_prompt(
            self.agent, message, message_history=self.message_history
        )

        # Query LLM
        try:
            response = await self._llm_provider.query(prompt)

            logger.info(
                f"Agent {self.agent.id} LLM decision: {response.decision} "
                f"(confidence: {response.confidence:.2f})"
            )
            logger.debug(f"Reasoning: {response.reasoning}")

            # Clear processed messages
            self.pending_messages.clear()

            # Return decision
            if response.decision == "evacuate":
                self.last_decision = Decision.EVACUATE
                self.last_reasoning = response.reasoning
                return {
                    "action": "evacuate",
                    "reasoning": response.reasoning,
                    "confidence": response.confidence,
                }
            else:
                self.last_decision = Decision.IGNORE
                self.last_reasoning = response.reasoning
                return {
                    "action": "continue",
                    "reasoning": response.reasoning,
                    "confidence": response.confidence,
                }

        except Exception as e:
            logger.error(f"LLM query failed for agent {self.agent.id}: {e}")
            # Default to staying on error
            self.pending_messages.clear()
            return {"action": "continue"}

    @classmethod
    async def batch_process_agents(cls, agents: list[Any]) -> dict[int, dict]:
        """
        Process multiple agents with LLM in a single batch.

        This is the key optimization - sends all agent contexts to LLM
        in one API call rather than individual queries.

        Args:
            agents: List of agents with pending LLM decisions

        Returns:
            Dict mapping agent IDs to their decisions
        """
        if not agents or cls._llm_provider is None:
            return {}

        logger.info(f"Batch processing {len(agents)} agents with LLM")
        logger.debug(f"Agent IDs: {[agent.id for agent in agents]}")

        # Build prompts for all agents
        try:
            messages = {agent.id: agent.decision_maker.pending_messages[-1] for agent in agents}
            prompts = EvacuationPromptBuilder.build_batch_prompts(agents, messages)
            logger.debug(f"Built {len(prompts)} prompts for LLM")
        except Exception as e:
            logger.error(f"Failed to build prompts: {e}", exc_info=True)
            return {}

        # Query LLM in batch
        try:
            logger.debug("Sending batch query to LLM provider")
            responses = await cls._llm_provider.batch_query(prompts)
            logger.debug(f"Received {len(responses)} responses from LLM")

            # Map responses to agents
            decisions = {}
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0

            for agent, response in zip(agents, responses):
                # Track token usage
                total_prompt_tokens += response.prompt_tokens
                total_completion_tokens += response.completion_tokens
                total_tokens += response.total_tokens

                if response.decision == "evacuate":
                    decisions[agent.id] = {
                        "action": "evacuate",
                        "reasoning": response.reasoning,
                        "confidence": response.confidence,
                    }
                else:
                    decisions[agent.id] = {
                        "action": "continue",
                        "reasoning": response.reasoning,
                        "confidence": response.confidence,
                    }

                # Store full response for action processing (Phase 2)
                decisions[agent.id + "_response"] = response

                # Clear processed messages
                agent.decision_maker.pending_messages.clear()

                logger.info(
                    f"Agent {agent.id}: {response.decision} (confidence: {response.confidence:.2f}) - {response.reasoning[:80]}"
                )

            # Log token usage summary
            logger.info(
                f"Batch LLM usage: {total_tokens} tokens "
                f"(prompt: {total_prompt_tokens}, completion: {total_completion_tokens})"
            )

            # Count decisions (excluding _response entries)
            decision_count = sum(1 for k in decisions.keys() if not k.endswith("_response"))
            evacuate_count = sum(
                1
                for k, d in decisions.items()
                if not k.endswith("_response") and d.get("action") == "evacuate"
            )
            logger.info(f"Returning {decision_count} decisions to simulation")
            logger.info(f"  {evacuate_count} agents decided to evacuate")
            logger.info(f"  {decision_count - evacuate_count} agents decided to stay")

            return decisions

        except Exception as e:
            logger.error(f"Batch LLM processing failed: {e}", exc_info=True)
            # Return default "continue" for all agents
            return {agent.id: {"action": "continue"} for agent in agents}
