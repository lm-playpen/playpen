import time
from pathlib import Path

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry, GameInstances, GameBenchmarkCallbackList, GameBenchmark, \
    InstanceFileSaver, ExperimentFileSaver, EpochResultsFolder, EpochResultsFolderCallback, InteractionsFileSaver
from clemcore.clemgame.runners import branching
from playpen import BasePlayPen, to_instances_filter
from datasets import load_dataset

from playpen.buffers import BranchingEpisodeBuffer
from playpen.callbacks.buffers import BranchingEpisodeBufferCallback


class BranchingPlayPenTrainer(BasePlayPen):

    def __init__(self, learner: Model, teacher: Model):
        """Showcase using the game of Taboo, which requires two players.
        Therefore, the learner is supposed to be accompanied by a teacher.

        However, in contract to clembench, the roles played by each model are to be decided programmatically.
        This means that the results folder structure does not necessarily show which model played which role.

        Note:
            Both models will always be loaded into memory, even if they are the same.
            This is intentional: While we want to adjust the learner's parameters,
            the teacher model must remain fixed as part of the environment to allow proper convergence.

        Args:
            learner: The learner model instance to be trained or adapted.
            teacher: The teacher model instance that remains fixed and acts as part of the environment.
        """
        super().__init__(learner, teacher)
        self.num_epochs = 2
        self.branching_factor = 2
        self.branching_condition = branching.is_player_model(self.learner)
        self.episode_buffer = BranchingEpisodeBuffer()
        # setup callbacks for the clem benchmark run
        results_folder = EpochResultsFolder(Path("playpen-records-branching"), Model.to_identifier([learner, teacher]))
        model_infos = Model.to_infos([learner, teacher])
        self.callbacks = GameBenchmarkCallbackList([
            # a callback to collect episodes into the buffer during the benchmark run
            BranchingEpisodeBufferCallback(self.episode_buffer),
            # a callback to increase the epoch number in the result folder
            EpochResultsFolderCallback(results_folder),
            # a callback to save the instance.json using the epoch result folder structure
            InstanceFileSaver(results_folder),
            # a callback to save the experiment.json using the epoch result folder structure
            ExperimentFileSaver(results_folder, player_model_infos=model_infos),
            # a callback to save the interactions.json and requests.json for a specific branch of the conversation
            InteractionsFileSaver(results_folder, player_model_infos=model_infos, store_branches=True)
        ])

    def learn(self, game_registry: GameRegistry):
        # We use the taboo game to showcase the basic playpen flow
        game_spec = game_registry.get_game_specs_that_unify_with("taboo")[0]

        # We only use the training instances so that we can properly evaluate on the validation set later
        dataset_train = load_dataset("colab-potsdam/playpen-data", "instances", split="train")

        # We initialize the game benchmark which creates the game master for each game instance
        with GameBenchmark.load_from_spec(game_spec) as game_benchmark:
            # We run as many epochs over all game instances as specified
            for epoch in range(self.num_epochs):
                # We collect the episodes using the batchwise runner from clemcore
                self._collect_episodes(game_benchmark, dataset_train)
                # We use the collected episodes to adjust model parameters of the learner
                self._train()

    def _collect_episodes(self, game_benchmark, dataset_train):
        # We reset the iterator to play all game instances once again
        game_instances = GameInstances.from_game_spec(game_benchmark.game_spec)
        game_instances = game_instances.filter(to_instances_filter(dataset_train))

        # We reset the episode buffer before each epoch over game instances
        # Note: We could also collect episodes over multiple epochs by calling reset only later
        self.episode_buffer.reset()

        # We invoke the branching runner to collect the episode trajectories for the game instance,
        # so that all game instances are played one after the other, but each episode branches at
        # certain points in time. This mode is supported by all models.
        branching.run(
            game_benchmark,
            game_instances,
            # Note: Here the order is important! We assign the roles so that:
            # - the teacher plays as the word describer (player at index 0)
            # - the learner plays as the word guesser (player at index 1)
            [self.teacher, self.learner],
            callbacks=self.callbacks,
            branching_factor=self.branching_factor,
            branching_condition=self.branching_condition
        )

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
        # Apply a training algorithm of your choice
        print("Training...")
        time.sleep(1)
