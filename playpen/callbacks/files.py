from pathlib import Path
from typing import TYPE_CHECKING, List

from clemcore.clemgame import ResultsFolder, GameBenchmarkCallback, InteractionsFileSaver, GameInteractionsRecorder
from clemcore.clemgame.resources import store_json
from clemcore.backends import Model

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameBenchmark, GameMaster
    from playpen.branching.master import BranchingGameMaster


class EpisodeResultsFolder(ResultsFolder):
    """
    Episode-based results layout.

    Allows iterating over the same game instance multiple times.
    Each game run corresponds to a new episode directory, independent of the underlying instance identity.

    Note:
    This aligns with repeated exposure to an initial state, e.g., reinforcement learning or stochastic evaluation.
    """

    def __init__(self, result_dir_path: Path, run_dir: str):
        super().__init__(result_dir_path, run_dir)
        self.episode_id = 0  # reset per process; overwrites on rerun

    def increment_episode_id(self):
        self.episode_id += 1

    def to_episode_dir(self) -> str:
        return f"episode_{self.episode_id:05d}"

    def to_instance_dir(self, game_instance: dict) -> str:
        """
        In episode-based layouts, the 'instance directory' corresponds
        to an episode rather than a unique game instance.
        """
        return self.to_episode_dir()


class EpisodeResultsFolderCallback(GameBenchmarkCallback):

    def __init__(self, results_folder: EpisodeResultsFolder):
        self.results_folder = results_folder

    def on_game_start(self, game_master: "GameMaster", game_instance: dict):
        # One game execution == one episode
        self.results_folder.increment_episode_id()


class EpochResultsFolder(ResultsFolder):
    """
    Epoch-based results layout.

    Each benchmark run corresponds to a new epoch.
    Within an epoch, each game instance is evaluated exactly once.

    This aligns with dataset-style training loops, e.g., supervised learning.
    """

    def __init__(self, result_dir_path: Path, run_dir: str):
        super().__init__(result_dir_path, run_dir)
        self.epoch_id = 0  # reset per process; overwrites on rerun

    def increment_epoch_id(self):
        self.epoch_id += 1

    def to_run_dir_path(self):
        models_dir_path = super().to_run_dir_path() / f"epoch_{self.epoch_id:05d}"
        return models_dir_path


class EpochResultsFolderCallback(GameBenchmarkCallback):

    def __init__(self, results_folder: EpochResultsFolder):
        self.results_folder = results_folder

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        # assuming every benchmark run corresponds to an epoch
        self.results_folder.increment_epoch_id()


class BranchingInteractionsFileSaver(InteractionsFileSaver):

    def _store_files(self, recorder, game_master: "BranchingGameMaster", game_instance):

        def get_recorder(gm: "GameMaster"):
            # noinspection PyProtectedMember
            for logger in gm._loggers:
                if isinstance(logger, GameInteractionsRecorder):
                    return logger
            raise RuntimeError("Cannot find a GameInteractionsRecorder for the given game master")

        # Well this is quite hacky but should work: We know that on game start there is
        # initially only a single recorder registered, but it gets copied at each step.
        # So, on game end, we simply collect the recorder for all branched game masters that survived.
        # However, this is a bit delicate because the framework don't expect us to know the loggers.
        game_masters = [node.unwrap() for node in game_master.get_active_tree().find_leaves()]
        for branch_idx, game_master in enumerate(game_masters):
            recorder = get_recorder(game_master)
            instance_dir_path = self.results_folder.to_instance_dir_path(game_master, game_instance)
            branch_dir_path = instance_dir_path / f"branch_{branch_idx + 1:05d}"
            store_json(recorder.interactions, "interactions.json", branch_dir_path)
            store_json(recorder.requests, "requests.json", branch_dir_path)
