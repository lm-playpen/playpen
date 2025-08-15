from copy import copy
from functools import partial
from typing import List, Dict, Callable, Tuple

from clemcore.clemgame import GameBenchmark, GameMaster
from clemcore.backends import Model
from clemcore.clemgame.events import GameEventLogger

from playpen.branching.player import BranchingStep, BranchingPlayer
from playpen.branching.tree import GameTree, ResponseTreeNode, GameTreeNode


class BranchingGameBenchmark(GameBenchmark):
    """
    A game benchmark decorator that ensures compatibility with the core framework
    while adding branching capabilities.

    Args:
        game_benchmark (GameBenchmark): The game benchmark to be extended with branching mechanisms.
        branching_factor (int): The number of branches created at each step when the branching_criteria is met.
        branching_criteria (Callable, optional): A function that determines when to branch at a step.
        Defaults to branching at every step.
    """

    def __init__(self,
                 game_benchmark: GameBenchmark,
                 *,
                 branching_factor: int = 1,
                 branching_criteria: Callable[[GameMaster], bool] = None):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        super().__init__(game_benchmark.game_spec)
        self.game_benchmark = game_benchmark
        self._branching_factor: int = branching_factor
        self._branching_criteria = branching_criteria

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        game_master = self.game_benchmark.create_game_master(experiment, player_models)
        return BranchingGameMaster(
            game_master,
            branching_factor=self._branching_factor,
            branching_criteria=self._branching_criteria
        )


class BranchingGameMaster(GameMaster):
    """
    A game master where an episode of gameplay evolves in a tree-like structure.
    The game master allows creating independently ongoing conversational branches at certain steps of the conversation.

    This enables collecting multiple responses for the same context at each step.
    In addition, each branch can itself be further branched, forming a full tree-like structure.

    Args:
        game_master (GameMaster): The game master that serves as the conversational root.
        branching_factor (int): The number of branches at each step when the branching_criteria is fulfilled.
        branching_criteria (Callable, optional): A function to determine when to branch at a step. Default: branches at every step.
    """

    def __init__(self,
                 game_master: GameMaster,
                 *,
                 branching_factor: int = 1,
                 branching_criteria: Callable[[GameMaster], bool] = lambda game_master: True):
        super().__init__(game_master.game_spec, game_master.experiment, game_master.player_models)
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._root: GameMaster = game_master
        self._done: bool = False
        self._game_tree = GameTree(GameTreeNode(self._root))
        self._active_parent_masters: List[GameMaster] = [self._root]
        self._branching_factor: int = branching_factor
        self._branching_criteria = branching_criteria

    def register_many(self, loggers: List[GameEventLogger]):
        self._root.register_many(loggers)

    def register(self, logger: GameEventLogger):
        self._root.register(logger)

    def setup(self, **game_instance):
        self._root.setup(**game_instance)

    def has_started(self) -> bool:
        return self._root.has_started()

    def observe(self) -> Tuple[Callable, List[GameMaster]]:
        # For simplicity, we use the active game masters as the context for branching
        return (partial(BranchingPlayer.branching_response,
                        branching_factor=self._branching_factor,
                        branching_criteria=self._branching_criteria),
                self._active_parent_masters)

    def step(self, all_branching_steps: List[List[BranchingStep]]) -> Tuple[bool, List[List[Dict]]]:
        assert isinstance(all_branching_steps, list), \
            f"Step expects a list of lists of branching steps and not {all_branching_steps.__class__}"

        all_infos = []
        candidates: List[BranchingCandidate] = []  # called candidates because we considered to apply a pruning function
        for branching_steps in all_branching_steps:
            infos = []
            for branching_step in branching_steps:  # each response represents a possible branch in the tree
                done, info = branching_step.apply()
                info["done"] = done  # store done in info, because runners only handle single bool return value
                infos.append(info)
                candidate = BranchingCandidate(branching_step, done, info)
                candidates.append(candidate)
            all_infos.append(infos)

        self._done = all([candidate.done for candidate in candidates])

        self._active_parent_masters = []  # memorize active leaves so that we do not have to find them again
        for candidate in candidates:
            candidate.add_branch_to(self._game_tree)
            self._active_parent_masters.append(candidate.response.branch_master)

        # return all dones and infos so that they match the quantity of the responses
        return self._done, all_infos

    def get_active_tree(self) -> "GameTree":
        """ Ad-hoc calculation of the tree containing only active branches """
        leaves = self._game_tree.find_leaves()
        active_leaves = []
        for leave in leaves:
            if leave.unwrap() in self._active_parent_masters:
                active_leaves.append(leave)

        def label_active_recursive(active_node):
            active_node.tag("active")
            if active_node.parent:  # root has no parent
                label_active_recursive(active_node.parent)

        for active_leave in active_leaves:
            label_active_recursive(active_leave)

        def copy_active_tree_recursive(active_node):
            _copy = copy(active_node)
            _copy.branches = [node for node in active_node if node.has_tag("active")]
            for branch in _copy:
                copy_active_tree_recursive(branch)

        active_root = copy(self._game_tree.root)  # we do not want to change the initial tree
        copy_active_tree_recursive(active_root)
        return GameTree(active_root)

    def is_done(self) -> bool:
        return self._done


class BranchingCandidate:

    def __init__(self, response: BranchingStep, done: bool, info: Dict):
        self.response = response
        self.done = done
        self.info = info

    def add_branch_to(self, game_tree):
        """ Find parent node and add child"""
        parent_node = game_tree.find_node(self.response.parent_master)
        assert parent_node is not None, "There must be a parent node that wraps the candidates parent env"
        branch_node = ResponseTreeNode(
            self.response.branch_master,  # the branched master (however, after step the current_player is wrong!)
            self.response.parent_player,  # the player creating the response
            self.response.parent_context,  # the context given to that player
            self.response.branch_response,  # one possible response given by that player
            self.done,
            self.info
        )
        parent_node.connect_to(branch_node)
