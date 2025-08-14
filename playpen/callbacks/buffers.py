from typing import TYPE_CHECKING, Dict

from clemcore.clemgame import GameBenchmarkCallback, GameStep

from playpen.buffers import EpisodeBuffer

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameMaster


class EpisodeBufferCallback(GameBenchmarkCallback):

    def __init__(self, episode_buffer: EpisodeBuffer):
        self.episode_buffer = episode_buffer

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        self.episode_buffer.next_episode()

    def on_game_step(self, game_master: "GameMaster", game_instance: Dict, game_step: GameStep):
        self.episode_buffer.add_step(game_step.context, game_step.response, game_step.done, game_step.info)
