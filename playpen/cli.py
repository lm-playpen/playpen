import argparse
import inspect
import importlib.util as importlib_util
import json
import os
from typing import Dict, Callable, List

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry
from playpen import BasePlayPen


def train(file_path: str, learner: ModelSpec, teacher: ModelSpec, temperature: float, max_tokens: int):
    def is_playpen(obj):
        return inspect.isclass(obj) and issubclass(obj, BasePlayPen) and obj is not BasePlayPen

    try:
        file_name = os.path.splitext(file_path)[0]
        spec = importlib_util.spec_from_file_location(file_name, file_path)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        playpen_subclasses = inspect.getmembers(module, predicate=is_playpen)
        if len(playpen_subclasses) == 0:
            raise ValueError(f"Cannot load playpen trainer, because no BasePlayPen found in {file_path}.\n"
                             f"Make sure that you have implemented a subclass of BasePlayPen and try again.")
        _, playpen_cls = playpen_subclasses[0]
    except Exception as e:
        raise RuntimeError(f"Cannot load playpen trainer, because {e}")

    game_registry = GameRegistry.from_directories_and_cwd_files()
    model_registry = ModelRegistry.from_packaged_and_cwd_files()

    learner_spec = model_registry.get_first_model_spec_that_unify_with(learner)
    print(f"Found registered model spec that unifies with {learner.to_string()} -> {learner_spec}")

    model_specs = [learner_spec]
    if teacher is not None:
        teacher_spec = model_registry.get_first_model_spec_that_unify_with(learner)
        print(f"Found registered model spec that unifies with {teacher.to_string()} -> {teacher_spec}")
        model_specs.append(teacher_spec)

    backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    for model_spec in model_specs:
        backend_selector = model_spec.backend
        if not backend_registry.is_supported(backend_selector):
            raise ValueError(f"Specified model backend '{backend_selector}' not found in backend registry.")
        print(f"Found registry entry for backend {backend_selector} "
              f"-> {backend_registry.get_first_file_matching(backend_selector)}")

    models = []
    for model_spec in model_specs:  # only now since model loading might take long
        print(f"Dynamically import backend {model_spec.backend}")
        backend = backend_registry.get_backend_for(model_spec.backend)
        model = backend.get_model_for(model_spec)
        model.set_gen_args(max_tokens=max_tokens, temperature=temperature)
        print(f"Successfully loaded {model_spec.model_name} model")
        models.append(model)

    learner_model = models[0]
    if len(models) == 1:
        playpen_cls(learner_model).learn(game_registry)
    else:
        teacher_model = models[1]
        playpen_cls(learner_model, teacher_model).learn(game_registry)


def store_eval_score(file_path, name, value):
    try:  # first, try to load file to not overwrite already written eval scores
        with open(file_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"Update {file_path}")
    except FileNotFoundError:
        print(f"Create {file_path}")
        scores = {}
    new_scores = {**scores, **{name: value}}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(new_scores, f)
    print(json.dumps(new_scores, indent=2))
    return new_scores


def to_task_selector(dataset) -> Callable[[str, str], List[int]]:
    import collections
    tasks_by_group = collections.defaultdict(list)
    for row in dataset:  # a list of rows with game, experiment, task_id columns
        key = (row['game'], row['experiment'])
        tasks_by_group[key].append(int(row['task_id']))
    return lambda game, experiment: tasks_by_group[(game, experiment)]


def evaluate(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: str, game_selector: str,
             skip_gameplay: bool):
    results_file = os.path.join(results_dir, f"{model_spec.model_name}.val.json")
    if suite in ["all", "clem"]:
        if not skip_gameplay:
            from datasets import load_dataset
            dataset = load_dataset("colab-potsdam/playpen-data", "instances", split="validation")
            task_selector = to_task_selector(dataset)
            clem.run(game_selector, [model_spec], gen_args=gen_args, results_dir=results_dir,
                     instances_filename="instances", task_selector=task_selector)
        # clem.score(game_selector, results_dir) # already done during run in clemcore 2.x
        clem.transcripts("all", results_dir)  # these will contain only the played games anyway
        df = clem.clemeval.perform_evaluation(results_dir, return_dataframe=True)
        clem_score = df["-, clemscore"][0]
        store_eval_score(results_file, "clemscore", clem_score)
    if suite in ["all", "stat"]:
        ...#todo: continue here with FMs


def cli(args: argparse.Namespace):
    if args.command_name == "list":
        if args.mode == "games":
            clem.list_games(args.selector, args.verbose)
        elif args.mode == "models":
            clem.list_models(args.verbose)
        elif args.mode == "backends":
            clem.list_backends(args.verbose)
        else:
            print(f"Cannot list {args.mode}. Choose an option documented at 'list -h'.")
    if args.command_name == "run":
        learner_spec = ModelSpec.from_string(args.learner)
        teacher_spec = ModelSpec.from_string(args.teacher) if args.teacher is not None else None
        train(args.file_path, learner_spec, teacher_spec, args.temperature, args.max_tokens)

    if args.command_name == "eval":
        model_spec = ModelSpec.from_string(args.model)
        gen_args = dict(temperature=args.temperature, max_tokens=args.max_tokens)
        results_dir = args.results_dir
        if results_dir is None:  # default
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            results_dir = f"playpen-eval/{timestamp}"
        evaluate(args.suite, model_spec, gen_args, results_dir, args.game, args.skip_gameplay)


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends. Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    train_parser = sub_parsers.add_parser("run")
    train_parser.add_argument("file_path", type=str,
                              help="The path to the trainer file to use for learning.")
    train_parser.add_argument("-l", "--learner", type=str,
                              help="The model name of the learner model (as listed by 'playpen list models').")
    train_parser.add_argument("-t", "--teacher", type=str, default=None,
                              help="The model name of the partner model (as listed by 'playpen list models')."
                                   "Optional, since non-interactive methods (like SFT) may not require a teacher model.",
                              required=False)
    train_parser.add_argument("-T", "--temperature", type=float, required=False, default=0.0,
                              help="The temperature used for generation. Should be the same as during training. "
                                   "Default: 0.0.")
    train_parser.add_argument("-L", "--max_tokens", type=int, required=False, default=300,
                              help="The token limit for generated responses. Should be the same as during training. "
                                   "Default: 300.")

    # Note: For now, we directly bound the eval to the playpen-data validate split.
    eval_parser = sub_parsers.add_parser("eval",
                                         description="Run the playpen eval pipelines to compute clem- and statscore.")
    eval_parser.add_argument("model", type=str,
                             help="The model name of the model to be evaluated (as listed by 'playpen list models').")
    eval_parser.add_argument("--suite", choices=["clem", "static", "all"],
                             default="all", nargs="?", type=str,
                             help="Choose which eval suites to run. Default: all")
    eval_parser.add_argument("-g", "--game", type=str, default="{'benchmark':['2.0']}",
                             help="A game selector e.g. a game name or a GameSpec-like JSON object given as a string."
                                  "Default: \"{'benchmark':['2.0']}\". Only relevant for 'clem'.")
    eval_parser.add_argument("-r", "--results_dir", type=str, default=None,
                             help="A relative or absolute path to a results directory. "
                                  "Default: playpen-eval/<timestamp>. Only relevant for 'clem'.")
    eval_parser.add_argument("--skip_gameplay", action="store_true",
                             help="Flag to skip gameplay and only calculate the clemscore for a given 'results_dir'."
                                  "Default: False. Only relevant for 'clem'.")
    eval_parser.add_argument("-T", "--temperature", type=float, default=0.0,
                             help="The temperature used for generation. Should be the same as during training. "
                                  "Default: 0.0.")
    eval_parser.add_argument("-L", "--max_tokens", type=int, default=300,
                             help="The token limit for generated responses. Should be the same as during training. "
                                  "Default: 300.")

    # todo: add a 'playpen play' option to allow collection of new interaction data on the train split

    cli(parser.parse_args())


if __name__ == "__main__":
    main()
