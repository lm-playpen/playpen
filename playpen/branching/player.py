from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Dict
from clemcore.clemgame.master import GameMaster, Player


@dataclass
class BranchingResponse:
    parent_master: "GameMaster"
    branch_master: "GameMaster"
    parent_context: Dict
    branch_response: str

    def step(self):
        return self.branch_master.step(self.branch_response)

    def __str__(self):
        return self.branch_response


class BranchingPlayer(Callable):
    """    Applies a player to a given context as many times as determined by the branching factor. """

    def __init__(self,
                 current_masters: List[GameMaster],
                 current_players: List[Player],
                 *,
                 branching_factor: int = 1,
                 branching_criteria: Callable[[GameMaster], bool] = None):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._branching_factor = branching_factor
        self._do_branch = branching_criteria or (lambda parent_master: True)  # always
        self._current_masters = current_masters
        self._current_players = current_players

    def __call__(self, contexts: List[str]) -> List[List[BranchingResponse]]:
        assert isinstance(contexts, List), "The context for TreePlayer must be a list of game environments"
        assert len(self._current_masters) == len(contexts), "There must be as many active branches as given contexts"
        context_responses = []
        for parent_master, parent_context in zip(self._current_masters, contexts):
            branch_responses = []
            branching_factor = self._branching_factor if self._do_branch(parent_master) else 1
            for _ in range(branching_factor):
                # We need to copy the env even with factor=1 (for the teacher) b.c. otherwise we run into problems
                # when adding the response to the tree, since we use the env identity as an id. If we do not copy,
                # then there will be two nodes with the same env which makes finding them via the env unpredictable.
                branch_master: GameMaster = deepcopy(parent_master)
                branch_player = branch_master.current_player  # we use the branch player as it keeps state
                # this already changes the player state in branch env
                branch_response_text = branch_player(parent_context)
                branch_response = BranchingResponse(parent_master, branch_master, parent_context, branch_response_text)
                branch_responses.append(branch_response)
            context_responses.append(branch_responses)
        return context_responses
