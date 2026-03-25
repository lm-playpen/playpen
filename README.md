![Playpen Logo](./playpen-logo.png)

The Playpen is an extension of the [clemcore](https://github.com/clp-research/clemcore) framework. It contains everything you need to get started **training** and **evaluating** models in - or from - interaction.

It was introduced in our [EMNLP 2025 paper](https://arxiv.org/abs/2504.08590). The code for the experiments in the paper can be found [here](https://github.com/lm-playpen/playpen-paper-2025).

# Get Started
## Set up the workspace

Clone the repository and switch into the workspace.
```bash
git clone https://github.com/lm-playpen/playpen.git && cd playpen
```

Set up the Python environment. Note: Playpen requires Python 3.10+.
```bash
python -m venv venv --system-site-packages && source venv/bin/activate
```

Install the requirements, which include the [clemcore](https://github.com/clp-research/clemcore) framework and the libraries to train and evaluate Huggingface models.
```bash
pip install -e .
```

Import then the [clembench](https://github.com/clp-research/clembench) repository in a directory of your choice and install its requirements. This will make the game environments available for training and evaluation.
```bash
git clone https://github.com/clp-research/clembench
pip install -r your/path/to/clembench/requirements.txt
```

In case the _clembench_ repository ha been cloned inside of the Playpen project root, games will be discovered automatically.
In case the cloning has been performed elsewhere, rename the file `game_registry.json.template` contained in the Playpen project root into `game_registry.json` and edit it to point to the _clembench_ path.

The file should be structured as follows:
```json
[
  {
    "benchmark_path": "your/path/to/clembench"
  }
]
```

To verify which games are available use:
```bash
clem list games
```

If you are interested in running the provided training examples, you should install TRL.
```bash
pip install '.[trl]'
```

Furthermore, if you want to run the prepared trainers in `examples/trl` with local huggingface models, you should install the `huggingface` extra. If you do not want this, you can still use the `trl` extra to run the trainers with remote models.
```bash
pip install 'clemcore[huggingface]'
```
Note: If you want to use the `transformers` library directly, you cannot use the playpen CLI to run the trainers, but you have to run your own scripts.

Now that everything is set up, you may follow the next steps to enter deeper into how to train and evaluate a model.
# Evaluate a model
## model_registry.json
In order to evaluate a model, the first thing we have to do is to register it into the `model_registry.json` file contained within this repository.
Mind that there is a file with the same name within the cloned _clembench_ folder. You may ignore that one.

Here is an example of model registry's entry for the Llama-3.1-Instruct model from Huggingface:
```json
{
  "model_name": "Llama-3.1-8B-Instruct",
  "backend": "huggingface_local",
  "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "release_date": "2024-07-23",
  "open_weight": true,
  "parameters": "8B",
  "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
  "context_size": "128k",
  "license": {
    "name": "Meta",
    "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
  },
  "model_config": {
    "requires_api_key": true,
    "premade_chat_template": true,
    "eos_to_cull": "<\\|eot_id\\|>"
  }
}
```
Here we are specifying fundamental information to help the engine to detect your model.
Most importantly, we are specifying a `model_name` for our model, the backend (in our case it is huggingface, indicated as `huggingface_local`) and the Huggingface model ID.
We are also specifying generation settings indicating whether a chat template associated to the model already exists (`premade_chat_template`: true), and the EOS token.

Note that specifying this character is of fundamental importance for the correct functioning of the benchmark.  
We also specify whether access to the model is gated and an API key is required under `requires_api_key`.

Further details regarding the other fields of a model entry are available in the [official documentation](https://github.com/clp-research/clemcore/blob/main/docs/model_backend_registry_readme.md).
## Registering an API key
There are two ways to register an API key. The first is within your environment as an environment variable. The second is within a file named `key.json`.
For the second case, copy or rename the file called `key.json.template` contained within the project's folder.
You may notice that there are several entries with the names of popular providers.
In the case you are interested in running Huggingface models, insert your key under `huggingface`:
```json
{
  ...
  "huggingface": {
    "api_key": "YOUR_API_KEY"
  },
  ...
 }
```
The mechanism is equivalent for other providers.
## Finally, the evaluation
To evaluate a model's gameplay performance on the `playpen-data` validation split, run the following command:

```bash
playpen eval <model-name>
```

where `<model-name>` should match your model's name as specified in the model registry. Connecting to the example above,
```bash
playpen eval Llama-3.1-8B-Instruct
```

This operation will produce a `<model-name>.val.json` file which contains two values:

1. **clemscore**: the average gameplay performance on the interactive benchmark games
2. **statscore**: the average performance on the static benchmark datasets

The file is by default located in a `playpen-eval/<timestamp>` folder.

Eventually, it is also possible to perform the evaluation separately on interactive and static benchmarks, or on single games.

To evaluate your model only on interactive games, run:
```bash
playpen eval Llama-3.1-8B-Instruct --suite clem
```
For static benchmarks:
```bash
playpen eval Llama-3.1-8B-Instruct --suite static
```

If you need to run the evaluation for specific games, e.g., for wordle, you can use the `-g` option of the eval command as follows:
```bash
playpen eval Llama-3.1-8B-Instruct -g wordle 
```

You may also eventually rescore episodes contained in a specific folder by indicating the folder under `-r` and setting `--skip-gameplay` to avoid rerunning the games from scratch:
```bash
playpen eval Llama-3.1-8B-Instruct --skip_gameplay -r playpen-eval/2026-03-05T09-37-23/
```
The existing scores will be overwritten.
# Train a model

The Playpen does not make many assumptions over the libraries or logic you will implement for training your models.
This means that you may implement your own training script as you see fit, and use either supervised fine-tuning or RL approaches.
The only constraint is that you have to inherit from the `BasePlaypenTrainer` class, which is the interface all training scripts are expected to follow.
It is defined [here](https://github.com/lm-playpen/playpen/blob/main/playpen/base.py) and it has a very simple structure.
It takes as arguments a `learner` model (i.e. the model to fine-tune) and optionally a `teacher` which may be helpful in an RL setup.

You may find all our examples in the [examples](https://github.com/lm-playpen/playpen/tree/main/examples) folder.

## Supervised Finetuning

Supervised fine-tuning (SFT) is known to help learning in interaction as it shifts the model's distribution towards the interactive data it will operate on.

In the context of clembench this means to let the model observe patterns of interaction which occur in various dialogue games.

### Training an Huggingface local model in the Playpen via TRL using Supervised Finetuning

Let's then run the basic trainer defined at `examples/trl/sft_trainer_simple.py` using the `Llama-3.1-8B-Instruct` model as a learning agent.
You may substitute this model with one of your choice.

```bash
playpen run examples/trl/sft_trainer_simple.py -l Llama-3.1-8B-Instruct 
```
> **Note:** a learning agent is defined using the argument -l <model_name>, with the <model_name> taken from the `model_registry.json` file.

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file.
After the execution is completed successfully, a directory named `models/sft/Llama-3.1-8B-Instruct` will appear
containing a sub-folder, e.g. `checkpoint-84` with the updated parameters of the model.

> **Note:** The trainer in the example above uses as a training set the train split available at the [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data/viewer/interactions) repository containing dialogues obtained from models playing the games available in the version 2.0 of clembench.
> You may check the `examples/trl/sft_trainer_simple.py` source code for more implementation details.

### Evaluate the fine-tuned model

To evaluate the effectiveness of our SFT approach, we can now run the trained model again on the clembench.
As we have seen above, we will have first to create a new entry for our model in our local `model_registry.json` file pointing to the checkpoint folder generated by the training script.

```json
{
  "model_name": "Llama-3.1-8B-Instruct-sft",
  "backend": "huggingface_local",
  "huggingface_id": "models/sft/Llama-3.1-8B-Instruct/checkpoint-84",
  "release_date": "2024-09-04",
  "open_weight": true,
  "parameters": "135M",
  "languages": ["en"],
  "context_size": "2048",
  "license": {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0"
  },
  "model_config": {
    "premade_chat_template": true,
    "eos_to_cull": "<\\|im_end\\|>"
  }
}
```
> **Note:** under 'huggingface_id' it is possible to specify the ID of a model from the Huggingface Hub rather than its local folder.

After this, we can run the benchmark, but this time specifying as model name that of our model:
```bash
playpen eval Llama-3.1-8B-Instruct-sft
```

> **Note:** You can look up baseline performances of other models on the leaderboard: https://clembench.github.io/leaderboard.html.

## Parameter Efficient Fine-tuning (PEFT)

One of the approaches you may use to train your model when under computational constraints is that of using Parameter-Efficient Fine-tuning strategies such as LoRA.
We show you how to do it in Playpen using TRL and Llama-3.1-8B-Instruct as model with [this](https://github.com/lm-playpen/playpen/blob/main/examples/trl/sft_trainer_lora.py) example.

You may notice that it is virtually equivalent to that introduced in the previous section, with the only difference in the specific settings for PEFT added to the TRL's SFTTrainer Object.

Similarly to above we may run the training, this time specifying a different training script and model for our new use-case.
```bash
playpen run examples/trl/sft_trainer_lora.py -l Llama-3.1-8B-Instruct 
```

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file.
When the command finished successfully, then there will be a `models/sft+lora/Llama-3.1-8B-Instruct` directory
containing a checkpoint folder, e.g. `checkpoint-78` **containing only the adapter's parameters**.

To evaluate the LoRA fine-tuned model we register it in the local `modal_registry.json`,
especially indicating the base model on top of which to put the trained adapters under `peft_model` in the `model_config` as follows:
```json
{
  "model_name": "Llama-3.1-8B-Instruct-sft-lora",
  "backend": "huggingface_local",
  "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "release_date": "2024-07-23",
  "open_weight": true,
  "parameters": "8B",
  "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
  "context_size": "128k",
  "license": {
    "name": "Meta",
    "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
  },
  "model_config": {
    "peft_model": "models/sft+lora/llama3-8b/checkpoint-78",
    "requires_api_key": true,
    "premade_chat_template": true,
    "eos_to_cull": "<\\|eot_id\\|>"
  }
}
```
> **Note:** If you want to evaluate quantized models, then you can simply add `load_in_8bit: true` or `load_in_4bit: true`
> in the model_config section of the model spec. Alternatively, you can also directly load a quantized model from the
> huggingface hub by specifying the according huggingface_id.

Now we run the usual command for running the evaluation:
```bash
playpen eval Llama-3.1-8B-Instruct-sft-lora
```

## Reinforcement Learning



# FOLLOWING SECTIONS TO BE REVIEWED - IGNORE FOR NOW
## Learning in Interaction (not yet available)

Having an SFT model ready, we can now turn to more interactive training algorithms.

### Running the GRPO+LoRA TRL example with self-play Llama3-8b (local)

The [clembench leaderboard](https://clembench.github.io/leaderboard.html) shows that the `Meta-Llama-3.1-8B-Instruct` model plays only 50% of the wordle game instances (v2.0) correctly and achieves only a quality score of 2.

Therefore, in this experiment we are interested in the performance gain of letting the model play the same instances multiple times, so that it eventually reaches better quality scores, but at least adheres more often to the game rules.

Hence, we use GRPO with a group size of 8, that is, we let the model play each instance (target word) of the wordle game `8` times, calculate the final reward for each gameplay and use LoRA to capture this learning signal in adapters:
```
trainer = trl.GRPOTrainer(
    peft_config=LoraConfig(
        r=8, lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM",
    )
)
```



### Running the GRPO+LoRA TRL example with Llama3-8b (local) and gpt4o-mini (remote)

Run the GRPO examples for a 2-player game.
In 2-player games, a teacher model plays the partner role.
In our case we use gpt4o-mini which is only accessible via a remote API.
Hence, we need to add credentials to the key.json to access the model.
```bash
echo '{
  "openai": {
    "organisation": "your_organisation",
    "api_key": "your_api_key"
  }
}' > key.json
```
> **Note:** An full template of the key.json for all supported remote backends is given in `key.json.template`.
> You can also manually insert the required information there and rename the file to `key.json`.

## Implement your own playpen trainer

tbd

# TL;DR

### Running the SFT+LoRA TRL example with Llama3-8b (local)

Run the SFT+LoRA TRL trainer example with a Llama3-8b learner (`-l`).
This doesn't require a teacher, because the model is optimized based on the examples given in the dataset (imitation learning).

```bash
playpen examples/trl/sft_trainer_lora.py -l llama3-8b
```

This saves the model checkpoint under a newly created folder at `models/sft+lora/llama3-8b`.

### Running the GRPO+LoRA TRL example with self-play Llama3-8b (local)

Run the GRPO+LoRA TRL trainer example with a Llama3-8b learner (`-l`)
using max token length (`-L`) 300 and temperature (`-T`) 0.75.

```bash
playpen examples/trl/grpo_trainer_lora_sp.py -l llama3-8b -L 300 -T 0.75
```

This creates a `playpen-records` directory containing the generated interactions
and saves the model checkpoint under a newly created folder at `models/grpo+lora/llama3-8b/selfplay`.

### Running the GRPO+LoRA TRL example with Llama3-8b (local) and gpt4o-mini (remote)

Run the GRPO+LoRA TRL trainer example with a Llama3-8b learner (`-l`)
and a GPT-4 teacher (`-t`) model (for 2-player games) using max token length (`-L`) 300 and temperature (`-T`) 0.75.

```bash
playpen examples/trl/grpo_trainer_lora_mp.py -l llama3-8b -t gpt4o-mini -L 300 -T 0.75
```

This creates a `playpen-records` directory containing the generated interactions
and saves the model checkpoint under a newly created folder at `models/grpo+lora/llama3-8b/gpt4o-mini`.

> **Note:** This only works when you added the proper `api_key` to the `key.json` for authentication.

### Create your own conversational dataset for SFT

The prepared examples make use of the canonical [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data) split
where we converted the interactions recorded during the v2.0 benchmark runs into a conversational dataset.
In HF, the main property of a conversational dataset is that it contains samples which specify a list of `messages`.
These messages usually iterate on roles, that is, between a `user` and an `assistant`, and carry textual content.

When you want to collect your own data samples, then run the benchmark with a model of your choice, for example:
```bash
clem run -g "{'benchmark':['2.0']}" -m llama3-8b
```

This will create a `results` directory with the model's gameplay recorded in `interaction.json` files.

To create a conversational dataset based on these interaction files, run the following command:
```bash
python3 examples/trl/data_utils.py <path-to>/results/
```

This will create in `examples/trl/results.jsonl` containing all interactions in form of a conversational dataset.
Furthermore, the script adds a `meta` annotation that informs about
`game`, `experiment`, `task_id`, `player_name`, `game_role`, `model` and `outcome`
which can be used for filtering the samples in the dataset.

Notably, the dataset contains samples of interaction from both perspectives of the 2-player games.
For example, for taboo the dataset contains the same episode, once from the perspective of the guesser and
once from the perspective of the clue giver.

> **Note:** The default implementation of TRL for SFT only trains the model to predict the last `assistant` messages.
> All other messages are handled as a prefix or context for the prediction.

### Using other existing models

Rename an already specified model or use another model by adding a custom model registry to the workspace.

Note: The entry with the renamed model is already prepared in the `model_registry.json` of this repository. The following code snippet exemplifies how this can be done.

Lookup existing (packaged) model specs.
```bash
playpen list models -v | grep Meta -A 6
...
Meta-Llama-3.1-8B-Instruct -> huggingface_local (packaged)
ModelSpec: {"model_name":"Meta-Llama-3.1-8B-Instruct" ... 
...
```

Note: You can also look up the packaged model specs in the [clemcore repository](https://github.com/clp-research/clemcore/blob/maintenance/2.x/clemcore/backends/model_registry.json).

Change model name from Meta-Llama-3.1-8B-Instruct to llama3-8b
```bash
echo '[{
  "model_name": "llama3-8b",
  "backend": "huggingface_local",
  "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "release_date": "2024-07-23",
  "open_weight": true,
  "parameters": "8B",
  "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
  "context_size": "128k",
  "license": {
    "name": "Meta",
    "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
  },
  "model_config": {
    "requires_api_key": true,
    "premade_chat_template": true,
    "eos_to_cull": "<\\|eot_id\\|>"
  }
}]' > model_registry.json 
```
The `llama3-8b` becomes available for model selection via the entry in the custom `model_registry.json`. Note that custom entries always precede packaged entries.
```bash
playpen list models | grep llama3
llama3-8b -> huggingface_local (.../playpen/model_registry.json)
```

If you want to make another existing Huggingface model available, then change here the `huggingface_id`, choose an appropriate `model_name` and set other relevant parameters.
