import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Any, Optional

AgentObservation = TypeVar("AgentObservation")
AgentResponse = TypeVar("AgentResponse")


class BaseAgent(abc.ABC, Generic[AgentObservation, AgentResponse]):
    """
    Abstract base class for all agents.

    This class defines the standard interface for an agent that receives
    an observation from an environment and returns a response or action.
    """

    def __call__(self, observation: Any) -> AgentResponse:
        """
        Calls the act method. Allows the agent to be used as a callable.

        Args:
            observation: The current state or observation from the environment.

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


@dataclass(frozen=True)
class ClemObservation:
    """
    Attributes:
        raw: The original, unmodified observation dictionary received from
            the environment. This contains all keys, including those not
            explicitly modeled by this dataclass.
        role: The role associated with this turn (usually: "user").
        content: The main textual content of the observation, such as a prompt,
            message, or game description shown to the agent.
        image: Optional image payload associated with the observation. The
            concrete type depends on the environment (e.g. a PIL image,
            a NumPy array, or a framework-specific object). This field is
            ``None`` if no image is present.
    """
    raw: dict[str, Any]
    role: str
    content: str
    image: Optional[Any] = None


class ClemAgent(BaseAgent[ClemObservation, str], abc.ABC):
    """
    Abstract base class for agents that act in Clem games.

    Clem agents receive raw environment observations as dictionaries and
    internally convert them into :class:`ClemObservation` instances. This
    preserves the original payload in ``raw`` while exposing common fields
    (``role``, ``content``, and optional ``image``) in a structured form.

    Subclasses should implement the :meth:`act` method with the following
    signature:

        def act(self, observation: ClemObservation) -> str:
            ...

    and return a string response appropriate for the Clem game.
    """

    def __call__(self, observation: dict[str, Any]) -> str:
        """
        Convert a raw observation into a `ClemObservation` and delegate to `act`.

        The input dictionary is expected to contain at least the keys
        ``"role"`` and ``"content"``. If present, the value under the
        ``"image"`` key is passed through; all keys (including extras) are
        preserved in the ``raw`` field.

        Args:
            observation: Raw observation which is expected to be a dictionary.
        Returns:
            The string response produced by the agent for this observation.

        Raises:
            KeyError: If required keys such as ``"role"`` or ``"content"``
                are missing from the input dictionary.
        """
        assert isinstance(observation, dict), "Observation for ClemAgents must be a dictionary"
        clem_observation = ClemObservation(
            raw=observation,
            role=observation["role"],
            content=observation["content"],
            image=observation.get("image")
        )
        return self.act(clem_observation)
