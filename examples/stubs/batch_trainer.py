import time
from pathlib import Path

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry, GameInstanceIterator, GameBenchmarkCallbackList, GameBenchmark, \
    InstanceFileSaver, ExperimentFileSaver, InteractionsFileSaver
from clemcore.clemgame.runners import batchwise
from playpen import BasePlayPen, to_sub_selector
from datasets import load_dataset

from playpen.buffers import EpisodeBuffer
from playpen.callbacks.buffers import EpisodeBufferCallback
from playpen.callbacks.files import EpochResultsFolder, EpochResultsFolderCallback


class BatchwisePlayPenTrainer(BasePlayPen):

    def __init__(self, learner: Model):
        super().__init__(learner)
        self.batch_size = 8
        self.num_epochs = 4
        self.episode_buffer = EpisodeBuffer()
        # setup callbacks for the clem benchmark run
        results_folder = EpochResultsFolder(Path("playpen-records"), [learner])
        model_infos = Model.to_infos([learner])
        self.callbacks = GameBenchmarkCallbackList([
            # a callback to collect episodes into the buffer during the benchmark run
            EpisodeBufferCallback(self.episode_buffer),
            # a callback to increase the epoch number in the result folder
            EpochResultsFolderCallback(results_folder),
            # a callback to save the instance.json using the epoch result folder structure
            InstanceFileSaver(results_folder),
            # a callback to save the experiment.json using the epoch result folder structure
            ExperimentFileSaver(results_folder, model_infos),
            # a callback to save the interactions.json and requests.json using the epoch result folder structure
            InteractionsFileSaver(results_folder, model_infos)
        ])

    def learn(self, game_registry: GameRegistry):
        # We use the taboo game to showcase the basic playpen flow
        game_spec = game_registry.get_game_specs_that_unify_with("taboo")[0]

        # We only use the training instances so that we can properly evaluate on the validation set later
        dataset = load_dataset("colab-potsdam/playpen-data", "instances", split="train")
        game_instance_iterator = GameInstanceIterator.from_game_spec(game_spec, sub_selector=to_sub_selector(dataset))

        # We initialize the game benchmark which creates the game master for each game instance
        with GameBenchmark.load_from_spec(game_spec) as game_benchmark:
            # We run as many epochs over all game instances as specified
            for epoch in range(self.num_epochs):
                # We collect the episodes using the batchwise runner from clemcore
                self._collect_episodes(game_benchmark, game_instance_iterator)
                # We use the collected episodes to adjust model parameters of the learner
                self._train()

    def _collect_episodes(self, game_benchmark, game_instance_iterator):
        # We reset the iterator to play all game instances once again
        game_instance_iterator.reset(verbose=False)
        # We reset the episode buffer before each epoch over game instances
        # Note: We could also collect episodes over multiple epochs by calling reset only later
        self.episode_buffer.reset()
        # We invoke the batchwise runner to collect the episode trajectories for the game instance,
        # so that all game instances are prepared first and then processed in batches. Note that
        # for this to work, the model backends must support batching! Otherwise, fallback to sequential.
        batchwise.run(
            game_benchmark,
            game_instance_iterator,
            [self.learner],
            callbacks=self.callbacks,
            batch_size=self.batch_size
        )

    def _train(self):
        # Convert the collected trajectories into conversational data format
        conversational_dataset = self.episode_buffer.to_conversational_dataset(self.learner)
        print("Collected episodes:", len(conversational_dataset))
        # Apply a training algorithm of your choice
        print("Training...")
        time.sleep(1)
