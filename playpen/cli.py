import argparse
import inspect
import importlib.util as importlib_util
import json
import os
from pathlib import Path
from typing import Dict
from datetime import datetime

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry, GameSpec
from playpen import BasePlayPen, to_sub_selector


def train(file_path: str, learner: ModelSpec, teacher: ModelSpec, temperature: float, max_tokens: int):
    def is_playpen(obj):
        return (inspect.isclass(obj)
                and issubclass(obj, BasePlayPen)
                and obj is not BasePlayPen
                and obj.__module__ == module.__name__  # defined in this file
                )

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


def store_eval_score(file_path: Path, name: str, value):
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



def get_default_results_dir():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_dir = Path("playpen-eval") / timestamp
    return results_dir


def evaluate_suite(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
                   dataset_name: str):
    suite_results_dir = str(results_dir / suite)
    if dataset_name is not None:
        from datasets import load_dataset
        dataset = load_dataset("colab-potsdam/playpen-data", dataset_name, split="validation")
        clem.run(game_selector, [model_spec],
                 gen_args=gen_args, results_dir=suite_results_dir, sub_selector=to_sub_selector(dataset))
    clem.score(game_selector, suite_results_dir)
    clem.transcripts(game_selector, suite_results_dir)
    df = clem.clemeval.perform_evaluation(suite_results_dir, return_dataframe=True)
    clem_score = df["-, clemscore"][0]
    return clem_score


def evaluate(suite: str, model_spec: ModelSpec, gen_args: Dict, results_dir: Path, game_selector: str,
             skip_gameplay: bool):
    overall_results_file = results_dir / f"{model_spec.model_name}.val.json"
    if suite in ["all", "clem"]:
        dataset_name = None if skip_gameplay else "instances"
        _game_selector = GameSpec.from_dict({"benchmark": ["2.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        clem_score = evaluate_suite("clem", model_spec, gen_args, results_dir, _game_selector, dataset_name)
        store_eval_score(overall_results_file, "clemscore", clem_score)
    if suite in ["all", "static"]:
        dataset_name = None if skip_gameplay else "instances-static"
        _game_selector = GameSpec.from_dict({"benchmark": ["static_1.0"]}, allow_underspecified=True) \
            if game_selector is None else game_selector
        stat_score = evaluate_suite("static", model_spec, gen_args, results_dir, _game_selector, dataset_name)
        store_eval_score(overall_results_file, "statscore", stat_score)


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
        evaluate(args.suite, model_spec, gen_args, args.results_dir, args.game, args.skip_gameplay)


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
    eval_parser.add_argument("--suite", choices=["clem", "static", "all"], default="all",
                             nargs="?", type=str,
                             help="Choose which eval suites to run. Default: all")
    eval_parser.add_argument("-g", "--game", type=str,
                             help="A game selector e.g. a game name or a GameSpec-like JSON object given as a string.")
    eval_parser.add_argument("-r", "--results_dir", type=Path, default=get_default_results_dir(),
                             help="A relative or absolute path to a playpen-eval results directory. "
                                  "This is expected to be one level above 'clem' or 'static' results."
                                  "Default: playpen-eval/<timestamp>.")
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
