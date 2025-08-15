from playpen.starters.branching_trainer import BranchingPlayPenTrainer
from playpen.buffers import BranchingEpisodeBuffer
from clemcore.backends import Model
from datasets import Dataset


class DPOEpisodeBuffer(BranchingEpisodeBuffer):

    def to_preference_dataset(self, perspective: Model, data_format="conversational") -> Dataset:
        """
        Transform the branching rollout buffer to a preference dataset for, e.g., DPO learning.

        # Standard format
        preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}

        # Conversational format
        preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                              "chosen": [{"role": "assistant", "content": "It is blue."}],
                              "rejected": [{"role": "assistant", "content": "It is green."}]}

        :param perspective: of a model generating the responses
        :param data_format: conversational or standard
        :return: a preference dataset as described in https://huggingface.co/docs/trl/dataset_formats#preference
        """
        return Dataset.from_list([])


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
        self.num_epochs = 2
        self.branching_factor = 2
        self.branching_criteria = lambda gm: self.is_learner(gm.observe()[0])  # current player is learner

    def _train(self):
        # Convert the collected trajectories into conversational data format
        conversational_dataset = self.episode_buffer.to_conversational_dataset(self.learner)
        # Given a branching factor 2 and the criteria to branch only for the learner,
        # the resulting number of conversations should be 384, that is,
        # 8 branches for each of the 48 training episodes. Why 8 branches?
        # The mock player always play an episode to the end, so the guesser has always 3 turns.
        # At each of these turns all existing conversations branch:
        # - at first turn there are then 1*2=2 conversations,
        # - at the second turn there are then 2*2=4 conversations,
        # - and at the third turn there are then 4*2=8 conversations,
        # finally leading to 2^3=8 branches.
        print("Collected episodes (perspective=learner):", len(conversational_dataset))
        print("Example episode:")
        for conversation in conversational_dataset:
            for message in conversation["messages"]:
                print(message)
            break
        print()
        # Turn the collected interactions into a preference dataset
        preference_dataset = self.episode_buffer.to_preference_dataset(self.learner)
        print("Collected preference samples (perspective=learner):", len(preference_dataset))
        print("Example preference sample:")
        for preference_example in preference_dataset:
            print(preference_example["prompt"])
            print(preference_example["chosen"])
            print(preference_example["rejected"])
            break
        # Apply a training algorithm of your choice
        print("Training...")
