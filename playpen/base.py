import abc

from clemcore.backends import Model, BatchGenerativeModel
from clemcore.clemgame import GameRegistry, Player


class BasePlayPen(abc.ABC):

    def __init__(self, learner: Model | BatchGenerativeModel, teacher: Model | BatchGenerativeModel = None):
        self.learner = learner
        self.teacher = teacher

    def is_learner(self, player: Player):
        return player.model is self.learner

    def is_teacher(self, player: Player):
        return player.model is self.teacher

    @abc.abstractmethod
    def learn(self, game_registry: GameRegistry):
        pass
