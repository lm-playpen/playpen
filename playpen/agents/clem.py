import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional
from typing_extensions import TypedDict, NotRequired

from playpen.agents.base import BaseAgent


class MessageDict(TypedDict):
    role: str
    content: str
    image: NotRequired[Any]


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

    @cached_property
    def as_message(self) -> MessageDict:
        """
        Converts the observation into the exact dictionary format expected by OpenAI/LLM chat completion APIs.
        """
        message: MessageDict = {"role": self.role, "content": self.content}
        if self.image:
            message["image"] = self.image
        return message


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

    def __init__(self, *, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt
        self.observations: list[ClemObservation] = []

    def reset(self):
        self.observations.clear()

    @property
    def history(self) -> list[dict]:
        """
        Returns the conversation history as a list of message dictionaries.
        This includes the system prompt at the first index if present.

        Complexity is O(N) where N is a history length, but since individual
        messages are cached, this is just a list of pointer lookups.
        """
        messages: list[MessageDict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(obs.as_message for obs in self.observations)
        return messages  # TypedDict is a dict

    def observe(self, observation: dict[str, Any], *, memorize: bool = True) -> ClemObservation:
        """
        :param observation: The input dictionary expected to contain at least the keys
        ``"role"`` and ``"content"``. If present, the value under the
        ``"image"`` key is passed through; all keys (including extras) are
        preserved in the ``raw`` field.
        :param memorize:
        :return:
        """
        assert isinstance(observation, dict), "Observation for ClemAgents must be a dictionary"
        clem_observation = ClemObservation(
            raw=observation,
            role=observation["role"],
            content=observation["content"],
            image=observation.get("image")
        )
        if memorize:
            self.observations.append(clem_observation)
        return clem_observation

    def __call__(self, observation: dict[str, Any], *, memorize: bool = True) -> str:
        """
        Convert a raw observation into a `ClemObservation` and delegate to `act`.

        Args:
            observation: Raw observation which is expected to be a dictionary.
        Returns:
            The string response produced by the agent for this observation.

        Raises:
            KeyError: If required keys such as ``"role"`` or ``"content"``
                are missing from the input dictionary.
        """
        return self.act(self.observe(observation, memorize=memorize))

    @abc.abstractmethod
    def act(self, last: ClemObservation) -> str:
        """
        Implement the main logic for the clem agent decision-making process.

        * Use 'last' to access the immediate prompt/content.
        * Use 'self.observations' for the full conversation history.

        Note:
            The last observation has been observed and, if memorized,
            is already the final element in `self.observations`

        Args:
            last: The current observation from the environment.

        Returns:
            The agent's calculated response or action.
        """
        pass
