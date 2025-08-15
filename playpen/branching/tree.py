from typing import List, Set, Dict

from clemcore.clemgame import GameMaster, Player


class GameTreeNode:
    def __init__(self, game_master: GameMaster):
        self._game_master = game_master
        self._branches: List[GameTreeNode] = []
        self._parent: GameTreeNode = None  # root has no parent
        self._tags: Set = set()

    @property
    def branches(self):
        return self._branches

    @branches.setter
    def branches(self, branches):
        self._branches = branches

    @property
    def parent(self):
        return self._parent

    @property
    def tags(self):
        return self._tags

    def untag(self, tag: str):
        self._tags.remove(tag)

    def tag(self, tag: str):
        self._tags.add(tag)

    def has_tag(self, tag: str):
        return tag in self._tags

    def __iter__(self):
        return iter(self._branches)

    def __bool__(self):
        return bool(self._branches)

    def unwrap(self):
        return self._game_master

    def wraps(self, game_master: GameMaster) -> bool:
        is_wrapping = self._game_master is game_master
        return is_wrapping

    def connect_to(self, branch_node: "GameTreeNode"):
        if branch_node in self._branches:
            return
        self._branches.append(branch_node)
        branch_node._parent = self


class ResponseTreeNode(GameTreeNode):

    def __init__(self, game_master: GameMaster, player: Player, context: Dict, response: str, done: bool, info: Dict):
        super().__init__(game_master)
        self.player = player
        self.context = context
        self.response = response
        self.done = done
        self.info = info


class GameTree:

    def __init__(self, root: GameTreeNode):
        self._root: GameTreeNode = root

    @property
    def root(self):
        return self._root

    def find_node(self, target_master: GameMaster):
        def _find_node(node):
            if node.wraps(target_master):  # check for object identity
                return node
            for branch in node:
                target_node = _find_node(branch)
                if target_node is not None:
                    return target_node
            return None

        return _find_node(self._root)

    def find_leaves(self):
        def _find_leaves(node):
            if not node:
                return [node]
            leaves = []
            for branch in node:
                leaves.extend(_find_leaves(branch))
            return leaves

        return _find_leaves(self._root)
