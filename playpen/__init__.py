BANNER = \
    r"""
.--------------..--------------..--------------..--------------..--------------..--------------..--------------.
|   ______     ||   _____      ||      __      ||  ____  ____  ||   ______     ||  _________   || ____  _____  |
|  |_   __ \   ||  |_   _|     ||     /  \     || |_  _||_  _| ||  |_   __ \   || |_   ___  |  |||_   \|_   _| |
|    | |__) |  ||    | |       ||    / /\ \    ||   \ \  / /   ||    | |__) |  ||   | |_  \_|  ||  |   \ | |   |
|    |  ___/   ||    | |   _   ||   / ____ \   ||    \ \/ /    ||    |  ___/   ||   |  _|  _   ||  | |\ \| |   |
|   _| |_      ||   _| |__/ |  || _/ /    \ \_ ||    _|  |_    ||   _| |_      ||  _| |___/ |  || _| |_\   |_  |
|  |_____|     ||  |________|  |||____|  |____|||   |______|   ||  |_____|     || |_________|  |||_____|\____| |
'--------------''--------------''--------------''--------------''--------------''--------------''--------------'
"""  # Blocks font, thanks to http://patorjk.com/software/taag/
import os

if os.getenv("PLAYPEN_DISABLE_BANNER", "0") not in ("1", "true", "yes", "on"):
    print(BANNER)

from typing import Callable

from playpen.callbacks.buffers import EpisodeBufferCallback, BranchingEpisodeBufferCallback
from playpen.buffers import EpisodeBuffer, BranchingEpisodeBuffer
from playpen.base import BasePlaypenTrainer

__all__ = [
    "EpisodeBuffer",
    "EpisodeBufferCallback",
    "BranchingEpisodeBuffer",
    "BranchingEpisodeBufferCallback",
    "BasePlaypenTrainer",
    "to_instances_filter"
]


def to_instances_filter(dataset) -> Callable[[dict], bool]:
    """ Converts the given dataset into a filter condition for use with GameInstances.filter(). """

    def dataset_identifier(row: dict) -> tuple[str, str, int]:
        return row["game"], row["experiment"], int(row["task_id"])

    whitelist = set(dataset_identifier(row) for row in dataset)

    def instance_identifier(row: dict) -> tuple[str, str, int]:
        return row["game_name"], row["experiment"]["name"], int(row["game_instance"]["game_id"])

    return lambda row: instance_identifier(row) in whitelist
