from openenv_core.client_types import StepResult
from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation

from playpen.agents.base import BaseAgentWrapper, AgentObservation, AgentResponse
from playpen.agents.clem import ClemAgent


class ClemGameEnvAgent(BaseAgentWrapper[StepResult[ClemGameObservation], ClemGameAction]):

    def __init__(self, wrapped_agent: ClemAgent):
        super().__init__(wrapped_agent)

    def __call__(self, result: StepResult[ClemGameObservation], *, memorize: bool = True) -> ClemGameAction:
        return ClemGameAction(response=self.wrapped_agent(result.observation.context, memorize=memorize))

    def act(self, observation: AgentObservation) -> AgentResponse:
        raise NotImplementedError("Calling act directly is not supported for this wrapper.")
