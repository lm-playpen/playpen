from pathlib import Path
from typing import TYPE_CHECKING, List

from clemcore.clemgame import ResultsFolder, GameBenchmarkCallback
from clemcore.backends import Model

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameBenchmark


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
