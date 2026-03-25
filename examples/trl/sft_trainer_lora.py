import trl
from peft import LoraConfig
from datasets import load_dataset

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel

from playpen import BasePlaypenTrainer


class PeftSftTrainer(BasePlaypenTrainer):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self):
        # Load a conversational dataset for SFT, that is, a list of "messages" -- basically tuples of role and content.
        # The role can be "user" or "assistant" and typically alternates within the list.
        # During training, everything up to the last assistant message becomes the prefix for prediction.
        # The loss is calculated based on the differences to the last assistant message.
        # Here we load the canonical training split as available in the huggingface playpen-data repository.
        # By default, the dataset is stored in ~/.cache/huggingface/datasets/ on your machine. This might take a while.
        dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")

        # Only use the episodes we are interested to train on: here all episodes with successful outcome
        dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")

        # We shuffle and split the remaining filtered samples to receive a test split
        dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

        # Initialize training configuration
        config = trl.SFTConfig(  # inherits TrainingArguments
            max_length=300,
            output_dir=f"models/sft+lora/{self.learner.name}",
            eval_strategy="epoch",
            packing=False,
            completion_only_loss=True
        )

        # Initialize trainer context
        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=config,
            # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            peft_config=LoraConfig(
                r=16, lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_token"],
                task_type="CAUSAL_LM",
            )
        )

        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()

        # Optional: Uncomment these lines to merge and save directly
        # merged_model = trainer.model.merge_and_unload()
        # merged_model.save_pretrained(f"models/sft+lora/{self.learner.get_name()}")
