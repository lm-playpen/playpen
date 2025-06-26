![Playpen Logo](./playpen-logo.png)

[Pre-Print](https://arxiv.org/abs/2504.08590)

All you need to get started with the LM Playpen Environment for Learning in Interaction.

## Set up the playpen workspace

Clone the repository and switch into the workspace.
```bash
git clone https://github.com/lm-playpen/playpen.git && cd playpen
```

Set up the Python environment. Note: Playpen requires Python 3.10+.
```bash
python -m venv venv --system-site-packages && source venv/bin/activate
```

Install the clemcore framework to run games, backends and models. 
Note: The `huggingface` extra is (only) necessary to train with local hf models.
```bash
pip install clemcore[huggingface]==2.4.0
```

Make playpen available via CLI and install TRL enable running the examples.
```bash
pip install '.[trl]'
```

Make the clembench games, e.g. taboo, available for learning.
For this, clone the clembench repository to a directory of your choice. 
```bash
git clone https://github.com/clp-research/clembench
```

Furthermore, we must install the clembench game requirements in our venv so that all games can be run properly:   
```bash
pip install your/path/to/clembench/requirements.txt
```


Then, back in you playpen workspace, copy the `game_registry.json.template` to `game_registry.json` so that the `clem` CLI can find it in the current working directory.
Set the path to the directory which contains the clembench repository.
The following command has a similar effect:
```bash
echo '[{"benchmark_path": "your/path/to/clembench"}]' > game_registry.json
```

> **Note:** Adding the game registry file is not necessary, 
> when you clone the clembench repository directly in your playpen workspace. 
> In this case the clem CLI can directly find the games by looking into sub-directories.
 
In any case, check that games are available via:
```bash
clem list games
```

Now having everything set up, you can follow the experiment guide or jump to the [TLDR](#tldr) section for a quick overview.

## Evaluate a model

To evaluate a model's gameplay performance on the `playpen-data` validation split, run the following command:

```bash
playpen eval <model-name>
```

where `<model-name>` should match your model's name as specified in the model registry.

This will produce a `<model-name>.val.json` file which contains two numbers:

1. **clemscore**: the average gameplay performance on the interactive benchmark games
2. **statscore**: the average performance on the static benchmark datasets

# Examples

## Supervised Finetuning

Supervised fine-tuning (SFT) is known to help learning in interaction as it shifts the model's distribution towards the interactive data it will operate on.

In the context of clembench this means to let the model observe patterns of interaction which occur in various dialogue games. 

### Running the SFT example with local HF models

Now we are ready to run the simple SFT TRL trainer example with a `smol-135m` learner (`-l`).
Since the interactions were already performed by other models, we do not need a teacher model in this case.
The following commands runs the example training pipeline:
```bash
playpen run examples/trl/sft_trainer_simple.py -l smol-135m 
```

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file. 
When the command finished successfully, then there will be a `models/sft/smol-135m` directory 
containing a checkpoint folder, e.g. `checkpoint-84` with the updated parameters of the model.

> **Note:** The example trainer will use the interactions of the train split available at the [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data) repository. 
> Have a look at `examples/trl/sft_trainer_simple.py` for implementation details.

### Evaluate the fine-tuned model

To evaluate the effectiveness of our SFT approach, we run the trained model again on the clembench.
For this, we first register our trained model in our local `model_registry.json` 
by adding an entry that points to the checkpoint folder:
```json
{
  "model_name": "smol-135m-sft",
  "backend": "huggingface_local",
  "huggingface_id": "models/sft/smol-135m/checkpoint-84",
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

Then we can run the benchmark again, but this time with `-m smol-135m-sft`:  
```bash
clem run -g "{'benchmark':['2.0']}" -m smol-135m-sft
```

> **Note:** We choose `smol-135m` only to showcase the workflow. For real training you should more capable models e.g. `llama3-8b`.
> You can look up baseline performances of other models on the leaderboard: https://clembench.github.io/leaderboard.html.

### Parameter Efficient Fine-tuning (PEFT)

More capable models like `llama3-8b` usually do not fit into the RAM of a single GPU during training.
A common technique to circumvent this, is a technique called low-rank adapters (LoRA) 
where only a smaller set of parameters (adapters) is trained to improve the model's performance. 
To make use of the LoRA support in TRL, we have to install the `peft` package 
and provide the trainer with the following additional configuration argument:
```python
trainer = trl.SFTTrainer(
    peft_config=LoraConfig(
        r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM",
    )
)
```

Now we are ready to run the LoRA SFT TRL trainer example with a `llama3-8b` learner (`-l`).
Since the interactions were already performed by other models, we do not need a teacher model in this case.
The following commands runs the example training pipeline:
```bash
playpen run examples/trl/sft_trainer_lora.py -l llama3-8b 
```

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file. 
When the command finished successfully, then there will be a `models/sft+lora/llama3-8b` directory 
containing a checkpoint folder, e.g. `checkpoint-78` **containing only the adapter parameters**.

> **Note:** Have a look at `examples/trl/sft_trainer_lora.py` for implementation details.

To evaluate the LoRA fine-tuned model we register it in the local `modal_registry.json`, 
especially pointing to a `peft_model` in the `model_config`, as follows:
```json
{
  "model_name": "llama3-8b-sft",
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

With this addition to the local model registry, `clem` is able to load the peft model properly, when we run the benchmark:
```bash
clem run -g "{'benchmark':['2.0']}" -m llama3-8b-sft
```

> **Note:** This essentially evaluates the model on the same instances of gameplay that were seen during training. To properly measure generalization performance, you should use different (or create new) instances.

> **Note:** If you want to train quantized models, then you can simply add `load_in_8bit: True` or `load_in_4bit: True`
> in the model_config section of the model spec. Alternatively, you can also directly load a quantized model from the
> huggingface hub by specifying the according huggingface_id.

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
