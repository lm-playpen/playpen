from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Dict
from clemcore.clemgame.master import GameMaster, Player


@dataclass
class BranchingStep:
    parent_master: "GameMaster"
    parent_player: Player  # observed in parent
    parent_context: Dict  # observed in parent
    branch_master: "GameMaster"
    branch_response: str  # advanced in branch

    def apply(self):
        return self.branch_master.step(self.branch_response)

    def __str__(self):
        return self.branch_response


class BranchingPlayer:

    @staticmethod
    def branching_response(
            parent_masters: List[GameMaster],
            *,
            branching_factor: int = 1,
            branching_criteria: Callable[[GameMaster], bool] = lambda game_master: True
    ) -> List[List[BranchingStep]]:
        assert isinstance(parent_masters, List), "The context must be a list"
        # For each parent master (leaf node of the interaction) we continue with possibly multiple branches
        all_branching_steps = []
        for parent_master in parent_masters:
            parent_player, parent_context = parent_master.observe()
            # We need to copy the env even with factor=1 (e.g. the teacher) b.c. otherwise we run into problems
            # when adding the response to the tree, since we use the env identity as an id. If we do not copy,
            # then there will be two nodes with the same env which makes finding them via the env unpredictable.
            branching_steps = []
            current_branching_factor = branching_factor if branching_criteria(parent_master) else 1
            for _ in range(current_branching_factor):  # todo we could use this to give ids like #turn.#branch
                # We detach the branch state from the parent state.
                branch_master = deepcopy(parent_master)
                # We use the branch player to advance its state (in the branch).
                # The player might generate a different response for each branch.
                branch_player, branch_context = branch_master.observe()
                # todo this fails after first step, why on earth?
                # todo parent master is wrong? branch should become new parent
                # if branch_context != parent_context:
                #    print(parent_context)
                #    print(branch_context)
                # assert branch_context == parent_context, "Context for parent and branch should be the same after copy"
                branch_response_text = branch_player(parent_context)
                branching_steps.append(
                    BranchingStep(
                        parent_master,
                        parent_player,
                        parent_context,
                        branch_master,
                        branch_response_text
                    )
                )
            all_branching_steps.append(branching_steps)
        return all_branching_steps
