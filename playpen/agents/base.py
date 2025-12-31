import abc
from typing import TypeVar, Generic, Any

AgentObservation = TypeVar("AgentObservation")
AgentResponse = TypeVar("AgentResponse")


class BaseAgent(abc.ABC, Generic[AgentObservation, AgentResponse]):
    """
    Abstract base class for all agents.

    This class defines the standard interface for an agent that receives
    an observation from an environment and returns a response or action.
    """

    def __call__(self, observation: Any, *, memorize: bool = True) -> AgentResponse:
        """
        Calls the act method. Allows the agent to be used as a callable.

        Args:
            observation: The current state or observation from the environment.
            memorize: A flag indicating whether the observation should be stored in the agent's memory.
        Returns:
            The agent's calculated response or action.
        """
        return self.act(observation)

    @abc.abstractmethod
    def act(self, observation: AgentObservation) -> AgentResponse:
        """
        Main logic for the agent's decision-making process.

        Args:
            observation: The current state or observation from the environment.

        Returns:
            The agent's calculated response or action.
        """
        pass

    def reset(self):
        """ Asks the agent to resets any internal state.

        Usually called at the end of and episode. """
        pass


class BaseAgentWrapper(BaseAgent[AgentObservation, AgentResponse]):

    def __init__(self, wrapped_agent: BaseAgent):
        self.wrapped_agent = wrapped_agent

    def act(self, observation: AgentObservation) -> AgentResponse:
        self.wrapped_agent.act(observation)

    def reset(self):
        self.wrapped_agent.reset()
