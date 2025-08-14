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

from typing import List, Callable

from playpen.callbacks.buffers import EpisodeBufferCallback, BranchingEpisodeBufferCallback
from playpen.buffers import EpisodeBuffer, BranchingEpisodeBuffer
from playpen.base import BasePlayPen

__all__ = [
    "EpisodeBuffer",
    "EpisodeBufferCallback",
    "BranchingEpisodeBuffer",
    "BranchingEpisodeBufferCallback",
    "BasePlayPen",
    "to_sub_selector"
]


def to_sub_selector(dataset) -> Callable[[str, str], List[int]]:
    import collections
    tasks_by_group = collections.defaultdict(list)
    for row in dataset:  # a list of rows with game, experiment, task_id columns
        key = (row['game'], row['experiment'])
        tasks_by_group[key].append(int(row['task_id']))
    return lambda game, experiment: tasks_by_group[(game, experiment)]
