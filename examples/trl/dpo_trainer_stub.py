from typing import List, Dict

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from playpen import BasePlayPen, RolloutProgressCallback, GameRecordCallback, make_tree_env, \
    BranchingRolloutBuffer, StepRolloutBuffer, make_env, GameEnv
from datasets import Dataset


class DPOPlayPenTrainer(BasePlayPen):
    """
    Then, fine-tuning a language model via DPO consists of two steps and is easier than PPO:
    (1) Data collection: Gather a preference dataset with positive and negative pairs of generation, given a prompt.
    (2) Optimization: Maximize the log-likelihood of the DPO loss directly.

    DPO requires a preference dataset. The DPOTrainer supports both conversational and standard dataset formats.
    When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

    See https://huggingface.co/docs/trl/dpo_trainer
    """

    def __init__(self, learner: Model):
        super().__init__(learner)
        self.rollout_steps = 16
        self.add_callback(RolloutProgressCallback(self.rollout_steps))
        self.add_callback(GameRecordCallback())

    def learn(self, game_registry: GameRegistry):
        # The tree env branches each step N times as specified by the branching_factor and branching_criteria
        def branch_on_guesser(game_master: "GameMaster"):
            # In this example, the learner plays both roles, so we also specify to branch only on a specific game role
            player = game_master.current_player
            return self.is_learner(player) and player.game_role == "WordGuesser"

        # For example, lets run the mock model on taboo with 16 rollout_steps.
        game_spec = game_registry.get_game_specs_that_unify_with("taboo")[0]
        # Given the branching criteria above there will be 16 conversations, because with the mock each game is played
        # for all 3 turns, so that there are 3 guesser turns, and on each guesser turn the game is multiplied 2 times
        # as specified by the branching factor. Hence, there will be 2^3=8 conversations per game.
        # Now, the rollout step is incremented each time a player does a turn, which is 2*3=6 times per game.
        # The third game will only be player for 4 turns (2 rounds), meaning that it will not be played until the end.
        # As a result you will only see episode_0 and episode_6 in the playpen-records directory.
        with make_tree_env(game_spec, [self.learner],
                           branching_factor=2,
                           branching_criteria=branch_on_guesser) as game_env:
            # (1) Step: Collect preference data
            rollout_buffer = DPORolloutBuffer(game_env)
            self._collect_rollouts(game_env, self.rollout_steps, rollout_buffer)
            # (2) Step: Train on preference data
            self._train(rollout_buffer)
            rollout_buffer.reset()

    def _train(self, rollout_buffer):
        # This is only to showcase the buffer use; you can also have a look into the playpen-records folder
        # Transform the buffer into a conversational dataset of the learner's perspective
        dataset = rollout_buffer.to_conversational_dataset(self.learner)
        if len(dataset) > 0:
            print(dataset[0])
        print(f"There are {len(dataset)} conversations in the dataset")
        # You need to implement this method to identify pairs of negative outcomes and positive outcomes
        rollout_buffer.to_preference_dataset(self.learner)
        # Do some serious training, setup DPOTrainer etc.
        ...
