from playpen.starters.branching_trainer import BranchingPlayPenTrainer
from clemcore.backends import Model
from clemcore.clemgame.runners import branching


class DPOPlayPenTrainer(BranchingPlayPenTrainer):
    """
    We use the same structure as defined by the BranchingPlayPenTrainer.

    Then, fine-tuning a language model via DPO consists of two steps and is easier than PPO:
    (1) Data collection: Gather a preference dataset with positive and negative pairs of generation, given a prompt.
    (2) Optimization: Maximize the log-likelihood of the DPO loss directly.

    DPO requires a preference dataset. The DPOTrainer supports both conversational and standard dataset formats.
    When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

    See https://huggingface.co/docs/trl/dpo_trainer
    """

    def __init__(self, learner: Model, teacher: Model):
        super().__init__(learner, teacher)
        # If necessary, customize values defined in the starter
        self.num_epochs = 1
        self.branching_factor = 2
        self.branching_criteria = branching.is_player_model(self.learner)
        self.teacher_role = "Player 1"  # teacher is describer
        self.learner_role = "Player 2"  # learner is guesser

    def _print_example_conversation(self, player_name: str):
        conversational_dataset = self.episode_buffer.to_conversational_dataset(player_name)
        print(f"Collected episodes (perspective={player_name}):", len(conversational_dataset))
        print("Example episode:")
        for conversation in conversational_dataset:
            for message in conversation["messages"]:
                print(message)
            break
        print()

    def _print_example_preferences(self, player_name: str):
        preference_dataset = self.episode_buffer.to_preference_dataset(player_name, require_different_scores=False)
        print(f"Collected preference samples (perspective={player_name}):", len(preference_dataset))
        print("Example preference sample:")
        preference_example = preference_dataset[0]
        print(preference_example["prompt"])
        print(preference_example["chosen"])
        print(preference_example["rejected"])
        print()

    def _train(self):
        # Convert the collected trajectories into conversational data format
        # Given a branching factor 2 and the criteria to branch only for the learner,
        # the resulting number of conversations should be 432, that is,
        # 8 branches for each of the 54 training episodes. Why 8 branches?
        # The mock player always play an episode to the end, so the guesser has always 3 turns.
        # At each of these turns all existing conversations branch:
        # - at first turn there are then 1*2=2 conversations,
        # - at the second turn there are then 2*2=4 conversations,
        # - and at the third turn there are then 4*2=8 conversations,
        # finally leading to 2^3=8 branches.
        self._print_example_conversation(self.learner_role)
        self._print_example_conversation(self.teacher_role)
        # Turn the collected interactions into a preference dataset
        self._print_example_preferences(self.learner_role)
        self._print_example_preferences(self.teacher_role)
        # Apply a training algorithm of your choice
        print("Training...")
