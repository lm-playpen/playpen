from pathlib import Path
from typing import TYPE_CHECKING, List

from clemcore.clemgame import ResultsFolder, GameBenchmarkCallback, InteractionsFileSaver, GameInteractionsRecorder
from clemcore.clemgame.resources import store_json
from clemcore.backends import Model

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameBenchmark, GameMaster
    from playpen.branching.master import BranchingGameMaster


class EpochResultsFolder(ResultsFolder):

    def __init__(self, result_dir_path: Path, player_models: List[Model]):
        super().__init__(result_dir_path, player_models)
        self.num_epoch = 0

    def increment_num_epoch(self):
        self.num_epoch += 1

    def to_models_dir_path(self):
        models_dir_path = super().to_models_dir_path() / f"epoch_{self.num_epoch:05d}"
        return models_dir_path


class EpochResultsFolderCallback(GameBenchmarkCallback):

    def __init__(self, results_folder: EpochResultsFolder):
        self.results_folder = results_folder

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        # assuming every benchmark run corresponds to an epoch
        self.results_folder.increment_num_epoch()


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
