"""Score both baselines and five abbreviation detection techniques on the same split."""

import argparse
import hashlib
import json
import sys
from itertools import count
from pathlib import Path
from pickle import UnpicklingError

import baseline
import regex_based
import rule_based
from crf_model import CRFModel, crfsuite_available
from evaluate import SPAN_METRICS, evaluate, score_values
from features import ALL_GROUPS, PROPOSAL_GROUPS
from pos import get_pos_tags
from terminal_table import choose_dataset, format_table
from train_crf import DEFAULT_CRF_ARGS, add_crf_args, train_and_eval
from utils import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SAVED_CRF_FILENAMES,
    default_data_path,
    load_jsonl,
    load_lexicon,
    reject_evaluation_overlap,
    split_records,
)

HYBRID_NAME = "CRF + Schwartz–Hearst features"
HYBRID_LABEL = "CRF + S&H features"
HYBRID_GROUPS = (*PROPOSAL_GROUPS, "rules")
SAVED_CRF_PATHS = {
    "CRF": MODEL_DIR / SAVED_CRF_FILENAMES["plain"],
    HYBRID_LABEL: MODEL_DIR / SAVED_CRF_FILENAMES["schwartz_hearst"],
    "CRF + Kaggle lexicon": MODEL_DIR / SAVED_CRF_FILENAMES["kaggle_lexicon"],
}
# Training data and settings used for the three saved CRFs.
SAVED_CRF_TRAINING_HASH = "44b73f14f0f780a2b2a1b0c7041325dc89185b3a76f994c3f42ef197ec900cd3"
SAVED_CRF_SETTINGS = {"c1": 0.2, "c2": 0.1, "max_iterations": 150}


def feature_groups(value):
    groups = tuple(group.strip() for group in value.split(","))
    if len(set(groups)) != len(groups) or set(groups) - set(ALL_GROUPS):
        raise argparse.ArgumentTypeError(f"expected unique comma-separated names from {ALL_GROUPS}")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Run without arguments to open the dataset menu.")
    parser.add_argument("--data", type=Path, default=default_data_path(),
                        help="training/development JSONL (defaults to the processed SDU data)")
    parser.add_argument("--extra-train", type=Path, action="append", default=[],
                        help="JSONL to append only to training after splitting (repeatable)")
    parser.add_argument("--test", type=Path,
                        help="held-out JSONL to score (loads saved CRFs for the bundled test when compatible)")
    parser.add_argument("--crf-groups", type=feature_groups, default=PROPOSAL_GROUPS,
                        help="comma-separated feature groups shared by the plain and lexicon CRFs")
    parser.add_argument("--hybrid-groups", type=feature_groups, default=HYBRID_GROUPS,
                        help="comma-separated feature groups for the hybrid CRF (must include rules)")
    parser.add_argument("--with-csit", action="store_true",
                        help="also compare a Schwartz–Hearst CRF with the Kaggle acronym lexicon")
    metric_selection = parser.add_mutually_exclusive_group()
    metric_selection.add_argument("--pairs-only", action="store_true",
                                  help="report only definition-pair metrics")
    metric_selection.add_argument("--spans-only", action="store_true",
                                  help="report only abbreviation and long-form span metrics")
    parser.add_argument("--output", type=Path, help="write metrics and aligned predictions as JSON")
    add_crf_args(parser)
    parser.set_defaults(**dict.fromkeys(DEFAULT_CRF_ARGS))

    if not sys.argv[1:]:
        try:
            run_menu(parser)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
        return

    args = parser.parse_args()
    try:
        run_comparison(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


def run_menu(parser):
    actions = {
        "1": "Compare on the bundled SDU development split",
        "2": "Split your dataset into training and development sets",
        "3": "Test on your dataset with bundled SDU training data",
        "4": "Choose separate training and test datasets",
        "0": "Exit",
    }
    menu = format_table(("Option", "Action"), actions.items(), title="NLP System Comparison", right=(0,))
    while True:
        print("\n" + menu)
        action = input("\nChoose an action: ").strip()
        if action == "0":
            return
        if action not in actions:
            print("Invalid action.")
            continue

        args = parser.parse_args([])
        if action in ("2", "4"):
            args.data = choose_dataset(DATA_DIR, (".jsonl",), title="Select a training dataset")
            if args.data is None:
                continue
        if action in ("3", "4"):
            args.test = choose_dataset(DATA_DIR, (".jsonl",), title="Select a test dataset")
            if args.test is None:
                continue

        try:
            run_comparison(args, sdu_test_menu=action == "3")
        except (ValueError, OSError) as error:
            print(f"\nError: {error}")
        input("\nPress Enter to continue...")


def load_saved_crfs(args, configurations, *, reuse_saved_for_custom_test=False):
    """Load the matched set of three CRFs for a test with compatible training settings."""
    if (args.test is None
            or (not reuse_saved_for_custom_test and args.test.resolve() != (DATA_DIR / "held-out-cs.jsonl").resolve())
            or args.data.resolve() != (DATA_DIR / "SDU-train-data.partial.jsonl").resolve()
            or args.extra_train or len(configurations) != len(SAVED_CRF_PATHS)):
        return {}

    if (args.backend not in (None, "auto", "crfsuite")
            or args.c1 not in (None, SAVED_CRF_SETTINGS["c1"])
            or args.c2 not in (None, SAVED_CRF_SETTINGS["c2"])
            or args.max_iter not in (None, SAVED_CRF_SETTINGS["max_iterations"])):
        return {}

    if hashlib.sha256(args.data.read_bytes()).hexdigest() != SAVED_CRF_TRAINING_HASH:
        print("The SDU training file has changed. Training the CRFs for this comparison.")
        return {}

    models = {}
    try:
        if not crfsuite_available():
            raise ValueError("CRFsuite is unavailable")
        for name, _, groups in configurations:
            model = CRFModel.load(SAVED_CRF_PATHS[name])
            if (not isinstance(model, CRFModel) or model.backend != "crfsuite"
                    or model.hparams != SAVED_CRF_SETTINGS or not model.use_pos
                    or model.propagation is not None):
                raise ValueError(f"{SAVED_CRF_PATHS[name].name} does not match the saved configuration")
            if model.groups != groups:
                return {}
            models[name] = model
        if get_pos_tags([], use_cache=False) is None:
            raise ValueError("the spaCy English model is unavailable")
    except (OSError, ValueError, EOFError, UnpicklingError, ImportError, AttributeError) as error:
        print(f"Can't use the saved CRFs ({error}). Training the CRFs for this comparison.")
        return {}
    return models


def save_test_report(test_path, contents):
    """Save a menu report to a new file without overwriting an earlier run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for run_number in count(1):
        suffix = "" if run_number == 1 else f"-{run_number}"
        path = RESULTS_DIR / f"{test_path.stem}-comparison{suffix}.json"
        try:
            output = path.open("x", encoding="utf-8")
        except FileExistsError:
            continue
        try:
            with output:
                output.write(contents)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path


def run_comparison(args, *, sdu_test_menu=False):
    if "rules" in args.crf_groups or "rules" not in args.hybrid_groups:
        raise ValueError("the rules feature group must appear only in --hybrid-groups")
    if args.test and args.test.resolve() == args.data.resolve():
        raise ValueError("training and test data must be separate files")
    if args.output:
        input_paths = (args.data, args.test, *args.extra_train, *SAVED_CRF_PATHS.values())
        output_path = args.output.resolve()
        if any(output_path == path.resolve()
               or (output_path.exists() and path.exists() and output_path.samefile(path))
               for path in input_paths if path is not None):
            raise ValueError("--output must be separate from dataset inputs and saved models")
    data = load_jsonl(args.data)
    if args.test:
        train, evaluation = data, load_jsonl(args.test)
        split_name = f"test ({args.test})"
    else:
        train, evaluation = split_records(data)
        split_name = "Dev"

    if args.test:
        reject_evaluation_overlap(train, evaluation, args.data)
    for path in args.extra_train:
        extra = load_jsonl(path)
        reject_evaluation_overlap(extra, evaluation, path)
        train.extend(extra)
        print(f"Added {len(extra):,} training sentences from {path}")

    if not train or not evaluation:
        raise ValueError("training and evaluation must each contain at least one sentence")

    print(format_table(
        ("Dataset", "Sentences"), [("Train", f"{len(train):,}"), (split_name, f"{len(evaluation):,}")],
        title="Comparison datasets", right=(1,), wrap=0,
    ), flush=True)
    lexicon = load_lexicon()
    predictions = {
        "Majority baseline": [baseline.majority_baseline(r["tokens"]) for r in evaluation],
        "Heuristic baseline": [baseline.rule_detector_baseline(r["tokens"], lexicon)
                               for r in evaluation],
        "Regex": regex_based.predict(evaluation),
        "S&H + propagation": rule_based.predict(evaluation),
    }
    systems = {name: evaluate(evaluation, labels) for name, labels in predictions.items()}

    crf_groups = tuple(group for group in args.crf_groups if group != "csit")
    hybrid_groups = tuple(group for group in args.hybrid_groups if group != "csit")
    configurations = [
        ("CRF", "CRF", crf_groups),
        (HYBRID_LABEL, HYBRID_NAME, hybrid_groups),
        ("CRF + Kaggle lexicon", "CRF + Kaggle lexicon", (*crf_groups, "csit")),
    ]
    if args.with_csit or "csit" in args.hybrid_groups:
        configurations.append(
            ("CRF + S&H + Kaggle lexicon", f"{HYBRID_NAME} + Kaggle lexicon", (*hybrid_groups, "csit")),
        )

    print()
    saved_models = load_saved_crfs(args, configurations, reuse_saved_for_custom_test=sdu_test_menu)
    options = {name: value for name, value in vars(args).items() if name in DEFAULT_CRF_ARGS and value is not None}
    training_args = argparse.Namespace(**(DEFAULT_CRF_ARGS | options))
    models = {}
    for name, log_name, groups in configurations:
        if saved_models:
            print(f"Using saved {log_name} ({SAVED_CRF_PATHS[name].name})…", flush=True)
            model = saved_models[name]
            labels = model.predict(evaluation)
            scores = evaluate(evaluation, labels)
        else:
            print(f"Training {log_name}…", flush=True)
            model, labels, scores = train_and_eval(train, evaluation, groups, training_args)
        models[name], predictions[name], systems[name] = model, labels, scores

    metrics = SPAN_METRICS
    if args.pairs_only:
        metrics = SPAN_METRICS[-1:]
    elif args.spans_only:
        metrics = SPAN_METRICS[:-1]
    rows = []
    for name, scores in systems.items():
        cells = ["/".join(score_values(scores[key])) for key, _ in metrics]
        rows.append((name, *cells))
    print("\n" + format_table(
        ("System", *(label for _, label in metrics)), rows,
        title="System comparison (precision / recall / F1)", right=range(1, len(metrics) + 1), wrap=0,
    ))
    print("S&H refers to Schwartz–Hearst.")
    if not args.spans_only:
        print("Pair scores use explicit gold links when available and heuristic links for predictions.")

    if args.output or sdu_test_menu:
        report = {
            "training_sentences": len(train),
            "evaluation_sentences": len(evaluation),
            "explicit_pair_sentences": sum("pairs" in record for record in evaluation),
            "models": {name: {"groups": model.groups, "backend": model.backend,
                              "use_pos": model.use_pos, **model.hparams} for name, model in models.items()},
            "scores": (
                {name: {key: scores[key] for key, _ in metrics} for name, scores in systems.items()}
                if args.pairs_only or args.spans_only else systems
            ),
            "predictions": [
                {"id": record["id"],
                 "labels": {name: labels[index] for name, labels in predictions.items()}}
                for index, record in enumerate(evaluation)
            ],
        }
        contents = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(contents, encoding="utf-8")
        else:
            path = save_test_report(args.test, contents)
            print(f"\nSaved scores and predictions to {path}")


if __name__ == "__main__":
    main()
