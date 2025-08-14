from typing import TYPE_CHECKING, Dict

from clemcore.clemgame import GameBenchmarkCallback, GameStep, GameMaster

from playpen.branching.master import BranchingGameMaster
from playpen.buffers import EpisodeBuffer, BranchingEpisodeBuffer


class EpisodeBufferCallback(GameBenchmarkCallback):

    def __init__(self, episode_buffer: EpisodeBuffer):
        self.episode_buffer = episode_buffer

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        self.episode_buffer.next_episode()

    def on_game_step(self, game_master: "GameMaster", game_instance: Dict, game_step: GameStep):
        self.episode_buffer.add_step(game_step.context, game_step.response, game_step.done, game_step.info)


class BranchingEpisodeBufferCallback(GameBenchmarkCallback):

    def __init__(self, episode_buffer: BranchingEpisodeBuffer):
        self.episode_buffer = episode_buffer

    def on_game_end(self, game_master: "BranchingGameMaster", game_instance: Dict):
        episode_tree = game_master.get_active_tree()
        self.episode_buffer.add_episode_tree(episode_tree)
