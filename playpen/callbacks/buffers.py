from copy import deepcopy
from typing import Dict, Callable

from clemcore.clemgame import GameBenchmarkCallback, GameStep, GameMaster, GameSnapshot

from playpen.buffers import EpisodeBuffer, BranchingEpisodeBuffer


class EpisodeBufferCallback(GameBenchmarkCallback):

    def __init__(self, episode_buffer: EpisodeBuffer):
        self.episode_buffer = episode_buffer

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        self.episode_buffer.next_episode()

    def on_game_step(self, game_master: "GameMaster", game_instance: Dict, game_step: GameStep):
        self.episode_buffer.add_step(game_step.context, game_step.response, game_step.done, game_step.info)


def average_rewards(rewards: dict):
    if not isinstance(rewards, dict):
        return 0.0
    return sum(rewards.values()) / len(rewards)


class BranchingEpisodeBufferCallback(GameBenchmarkCallback):
    """Callback that collects episode trajectories into a BranchingEpisodeBuffer for dataset creation.

    This callback is designed to work with BranchingRunner and will be deep copied at each
    branching point during the run. The deep copy mechanism is intentional and central to
    how trajectory collection works:

    Shared across all branches (same reference after deep copy):
        - episode_buffer: all branches write their branching points to the same buffer
        - score_func: a pure function, not copied anyway

    Independent per branch (deep copied):
        - _trajectory: each branch accumulates its own full trajectory independently
        - _trajectory_markers: each branch tracks its own path through the branching tree

    Design note - why we carry the full trajectory forward:
        At branch time we know the trajectory so far, but not the outcome (episode_score).
        At game end we know the outcome, but need to know where branching occurred.
        These two pieces of information are only available at different points in time,
        so we must carry the full trajectory forward to game end.

        _trajectory_markers is the bridge: a lightweight list of (branching_point_id, turn)
        bookmarks that record WHERE in the full trajectory each branch diverged, without
        storing a copy of the trajectory at each branch point. Only at game end, when the
        outcome is known, do we slice the trajectory at each marker and register it in the
        shared buffer — allowing to_preference_dataset to compare siblings at every level
        of the branching tree.

    Example:
        buffer = BranchingEpisodeBuffer()
        callback = BranchingEpisodeBufferCallback(buffer)
        # Pass to BranchingRunner via callbacks - the runner handles deep copying
        run(..., callbacks=[..., callback])
        # After run, buffer contains all branching points across all branches
        preference_dataset = buffer.to_preference_dataset(perspective=learner_model)
        conversational_dataset = buffer.to_conversational_dataset(perspective=learner_model)
    """

    def __init__(self,
                 episode_buffer: BranchingEpisodeBuffer,
                 score_func: Callable[[Dict[str, float]], float] = average_rewards):
        self.episode_buffer = episode_buffer
        self.score_func = score_func
        self._trajectory_markers: list[tuple[GameSnapshot, int]] = []  # (branching_point_id, turn index)
        self._trajectory: list[GameStep] = []

    def __deepcopy__(self, memo):
        # episode_buffer: passed as-is — all branches write to the same buffer
        # score_func: passed as-is — functions are not copied anyway
        # _trajectory_markers: deep copied — each branch tracks its own path through the tree
        # _trajectory: deep copied — each branch accumulates its own independent trajectory
        new = BranchingEpisodeBufferCallback(self.episode_buffer, self.score_func)
        new._trajectory_markers = deepcopy(self._trajectory_markers, memo)
        new._trajectory = deepcopy(self._trajectory, memo)
        return new

    def on_branching_point(self, game_master: "GameMaster", game_instance: Dict, snapshot: GameSnapshot):
        """Called by BranchingRunner for each new branch when a branching condition is met.

        Records a snapshot and the current trajectory length. In this way,
        we can later slice the full trajectory at the correct divergence point
        on game end. Does NOT store a copy of the trajectory at this point.

        Note:
            The snapshot identifies this branching point by its origin, which is
            shared across all siblings that diverged here,
            used to group them for preference pair creation.
        """
        self._trajectory_markers.append((snapshot, len(self._trajectory)))

    def on_game_step(self, game_master, game_instance, game_step: GameStep):
        """Accumulates the full trajectory for this branch, step by step."""
        self._trajectory.append(game_step)

    def on_game_end(self, game_master, game_instance, exception=None, rewards=None):
        """Registers this branch's trajectory under every branching point it passed through.

        At this point the episode_score is finally known, so we can slice the full
        trajectory at each marker and add the resulting BranchingPoints to the shared
        buffer. This allows to_preference_dataset to create preference pairs at every
        level of the branching tree, not just the final turn.
        """
        if exception is not None:
            return
        episode_score = self.score_func(rewards)
        for game_snapshot, turn in self._trajectory_markers:
            self.episode_buffer.add_branching_point(game_snapshot, turn, self._trajectory, episode_score)
        self._trajectory = []
        self._trajectory_markers = []
