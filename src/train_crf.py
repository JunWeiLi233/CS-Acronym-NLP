"""Train a CRF to label abbreviations and long forms together.

Usage:
    python src/train_crf.py                    # train on train split, score dev, save model
    python src/train_crf.py --with-rules       # add Schwartz–Hearst features
    python src/train_crf.py --propagate later  # + document-level propagation at prediction
    python src/train_crf.py --ablation         # remove one feature group at a time (dev)
    python src/train_crf.py --search           # small c1/c2 grid on dev
    python src/train_crf.py --final --test data/processed/test.jsonl
                                               # train on train+dev, one final test run

The model is saved to models/crf.pkl (or --model-out) and can be used by
predict.py. Dev = 20% of the SDU data, split as in utils.split_records.
"""

import argparse
import json
import time

from crf_model import CRFModel, crfsuite_available
from evaluate import (
    SPAN_F1_HEADERS,
    SPAN_METRICS,
    dump_errors,
    evaluate,
    format_report,
    span_f1_values,
)
from features import ALL_GROUPS, PROPOSAL_GROUPS
from terminal_table import format_table
from utils import (
    MODEL_DIR,
    RESULTS_DIR,
    load_split,
    reject_evaluation_overlap,
    save_jsonl,
    split_records,
)


def train_and_eval(train, dev, groups, args, propagation=None, verbose=True):
    """Fit a CRF and return its model, labels, and evaluation scores."""
    started = time.time()
    model = CRFModel(
        groups=groups,
        backend=args.backend,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter,
        propagation=propagation,
    )
    model.fit(train)
    predictions = model.predict(dev)
    scores = evaluate(dev, predictions)
    if verbose:
        print(f"  [{model.backend}] trained on {len(train)} sentences in {time.time() - started:.0f}s"
              f" (POS features: {'on' if model.use_pos else 'off'})")
    return model, predictions, scores


def run_ablation(train, dev, groups, args):
    """Measure the effect of removing one feature group at a time."""
    print("\nTraining with all feature groups…", flush=True)
    _, _, full_scores = train_and_eval(train, dev, groups, args)
    rows = [("all groups", full_scores)]
    for omitted_group in groups:
        remaining_groups = [group for group in groups if group != omitted_group]
        print(f"Training without {omitted_group}…", flush=True)
        _, _, scores = train_and_eval(train, dev, remaining_groups, args, verbose=False)
        rows.append((f"without {omitted_group}", scores))
    full_pair_f1 = full_scores["pair"]["f1"]
    print("\n" + format_table(
        ("Configuration", *SPAN_F1_HEADERS, "Pair F1 change"),
        ((name, *span_f1_values(scores), f"{scores['pair']['f1'] - full_pair_f1:+.3f}")
         for name, scores in rows),
        title="Feature ablation on dev", right=range(1, 5), wrap=0,
    ))
    return rows


def run_search(train, dev, groups, args):
    """Return the regularization settings with the highest dev pair F1."""
    if args.backend == "simple" or (args.backend == "auto" and not crfsuite_available()):
        grid = [(0.0, c2) for c2 in (0.01, 0.05, 0.2)]
    else:
        grid = [(c1, c2) for c1 in (0.0, 0.05, 0.2) for c2 in (0.01, 0.05, 0.1)]
    print(f"\nSearching {len(grid)} hyperparameter settings on dev", flush=True)
    results = []
    for index, (c1, c2) in enumerate(grid, start=1):
        print(f"[{index}/{len(grid)}] Training with c1={c1}, c2={c2}…", flush=True)
        args.c1, args.c2 = c1, c2
        _, _, scores = train_and_eval(train, dev, groups, args, verbose=False)
        results.append((c1, c2, scores))
    best = max(results, key=lambda result: result[2]["pair"]["f1"])
    print("\n" + format_table(
        ("c1", "c2", *SPAN_F1_HEADERS, "Selected"),
        ((c1, c2, *span_f1_values(scores), "Yes" if (c1, c2) == best[:2] else "")
         for c1, c2, scores in results),
        title="Hyperparameter search on dev", right=range(5),
    ))
    return best


DEFAULT_CRF_ARGS = {"backend": "auto", "c1": 0.05, "c2": 0.05, "max_iter": 150}


def add_crf_args(parser):
    parser.add_argument("--backend", choices=["auto", "crfsuite", "simple"],
                        help="training backend (default: auto)")
    parser.add_argument("--c1", type=float, help="L1 (crfsuite only)")
    parser.add_argument("--c2", type=float, help="L2")
    parser.add_argument("--max-iter", type=int, help="maximum training iterations (default: 150)")
    parser.set_defaults(**DEFAULT_CRF_ARGS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="SDU jsonl used for the train/dev split")
    add_crf_args(ap)
    ap.add_argument("--groups", default=",".join(PROPOSAL_GROUPS),
                    help=f"comma-separated feature groups from {ALL_GROUPS}")
    ap.add_argument("--with-rules", action="store_true",
                    help="add Schwartz–Hearst and glossary matching features")
    ap.add_argument("--with-csit", action="store_true", help="add the Kaggle acronym lexicon feature")
    ap.add_argument("--propagate", choices=["later", "all"], default=None,
                    help="apply document-level propagation to CRF output")
    ap.add_argument("--ablation", action="store_true", help="score each feature-group omission on dev")
    ap.add_argument("--search", action="store_true", help="select c1 and c2 by pair F1 on dev before training")
    ap.add_argument("--final", action="store_true", help="train on train+dev before testing")
    ap.add_argument("--test", default=None, help="gold test .jsonl for the final evaluation")
    ap.add_argument("--model-out", default=str(MODEL_DIR / "crf.pkl"), help="model output file (default: models/crf.pkl)")
    ap.add_argument("--save-preds", action="store_true", help="write dev and any test predictions to results/")
    ap.add_argument("--errors", default=None, help="write dev error examples to this file")
    args = ap.parse_args()

    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    unknown_groups = sorted(set(groups) - set(ALL_GROUPS))
    if unknown_groups:
        ap.error(f"unknown feature groups: {', '.join(unknown_groups)}")
    if args.with_rules and "rules" not in groups:
        groups.append("rules")
    if args.with_csit and "csit" not in groups:
        groups.append("csit")

    train, dev = split_records(load_split("all", args.data))
    test = load_split(args.test) if args.test else None
    if test is not None:
        if not test:
            ap.error("--test must contain at least one sentence")
        try:
            reject_evaluation_overlap(train + dev, test, args.data or "bundled SDU data")
        except ValueError as error:
            ap.error(str(error))
    print(format_table(
        ("Setting", "Value"), (("Train sentences", f"{len(train):,}"),
                               ("Dev sentences", f"{len(dev):,}"),
                               ("Feature groups", ", ".join(groups))),
        title="Training configuration", wrap=1,
    ), flush=True)
    if not crfsuite_available() and args.backend != "simple":
        print("[info] python-crfsuite not installed -> using the numpy fallback CRF "
              "(pip install python-crfsuite for the faster backend)")

    if args.ablation:
        rows = run_ablation(train, dev, groups, args)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / "crf_ablation.json"
        ablation_scores = {
            name: {metric: scores[metric] for metric, _ in SPAN_METRICS}
            for name, scores in rows
        }
        with output_path.open("w", encoding="utf-8") as output:
            json.dump(ablation_scores, output, indent=2)
        print(f"\nAblation results saved to {output_path}")
        return
    if args.search:
        args.c1, args.c2, _ = run_search(train, dev, groups, args)

    fit_on = train + dev if args.final else train
    print(f"\nTraining on {len(fit_on):,} sentences…", flush=True)
    model, preds, scores = train_and_eval(fit_on, dev, groups, args, propagation=args.propagate)
    name = "CRF" + (" + rules" if "rules" in groups else "")
    if "csit" in groups:
        name += " + Kaggle lexicon"
    if args.propagate:
        name += " + propagation"
    if args.final:
        print("[note] --final: model was trained on train+dev, so dev scores below are not held-out")
    print(format_report(f"{name} on dev", scores, token_level=True))
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")

    if args.save_preds:
        out = RESULTS_DIR / "crf.dev.jsonl"
        save_jsonl([{"id": r["id"], "labels": p} for r, p in zip(dev, preds)], out)
        print(f"Dev predictions saved to {out}")
    if args.errors:
        dump_errors(dev, preds, args.errors)

    if test is not None:
        test_preds = model.predict(test)
        print(format_report(f"{name} on TEST ({args.test})", evaluate(test, test_preds), token_level=True))
        if args.save_preds:
            save_jsonl([{"id": r["id"], "labels": p} for r, p in zip(test, test_preds)],
                       RESULTS_DIR / "crf.test.jsonl")


if __name__ == "__main__":
    main()
