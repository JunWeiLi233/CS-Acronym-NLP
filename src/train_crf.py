"""Technique 2: feature-based CRF that labels abbreviations and long forms
jointly (B-AC / B-LF / I-LF / B-O) in one pass.

Usage:
    python src/train_crf.py                    # train on train split, score dev, save model
    python src/train_crf.py --with-rules       # add Schwartz-Hearst output as features
    python src/train_crf.py --propagate        # + document-level propagation at prediction
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
from evaluate import dump_errors, evaluate, format_report, summary_row
from features import ALL_GROUPS, PROPOSAL_GROUPS
from utils import MODEL_DIR, RESULTS_DIR, load_split, save_jsonl


def train_and_eval(train, dev, groups, args, propagation=None, verbose=True):
    t0 = time.time()
    model = CRFModel(groups=groups, backend=args.backend, c1=args.c1, c2=args.c2,
                     max_iterations=args.max_iter, propagation=propagation)
    model.fit(train)
    preds = model.predict(dev)
    scores = evaluate(dev, preds)
    if verbose:
        print(f"  [{model.backend}] trained on {len(train)} sentences in {time.time() - t0:.0f}s"
              f" (POS features: {'on' if model.use_pos else 'off'})")
    return model, preds, scores


def run_ablation(train, dev, groups, args):
    print(f"\nFeature ablation on dev (groups: {', '.join(groups)})")
    rows = []
    _, _, full = train_and_eval(train, dev, groups, args)
    rows.append(("all groups", full))
    for g in groups:
        kept = [x for x in groups if x != g]
        print(f"  - without {g} ...")
        _, _, s = train_and_eval(train, dev, kept, args, verbose=False)
        rows.append((f"without {g}", s))
    print(f"\n{'configuration':34}{'AC F1':>8}{'LF F1':>8}{'Pair F1':>8}   (delta pair F1)")
    base = full["pair"]["f1"]
    for name, s in rows:
        print(summary_row(name, s) + f"   {s['pair']['f1'] - base:+.3f}")
    return rows


def run_search(train, dev, groups, args):
    if args.backend == "simple" or (args.backend == "auto" and not crfsuite_available()):
        grid = [(0.0, c2) for c2 in (0.01, 0.05, 0.2)]
    else:
        grid = [(c1, c2) for c1 in (0.0, 0.05, 0.2) for c2 in (0.01, 0.05, 0.1)]
    print(f"\nHyper-parameter search on dev ({len(grid)} settings)")
    best = None
    for c1, c2 in grid:
        args.c1, args.c2 = c1, c2
        _, _, s = train_and_eval(train, dev, groups, args, verbose=False)
        print(summary_row(f"c1={c1} c2={c2}", s))
        if best is None or s["pair"]["f1"] > best[2]["pair"]["f1"]:
            best = (c1, c2, s)
    print(f"Best: c1={best[0]} c2={best[1]} (pair F1 {best[2]['pair']['f1']:.3f})")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="SDU jsonl used for the train/dev split")
    ap.add_argument("--backend", choices=["auto", "crfsuite", "simple"], default="auto")
    ap.add_argument("--c1", type=float, default=0.05, help="L1 (crfsuite only)")
    ap.add_argument("--c2", type=float, default=0.05, help="L2")
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--groups", default=",".join(PROPOSAL_GROUPS),
                    help=f"comma-separated feature groups from {ALL_GROUPS}")
    ap.add_argument("--with-rules", action="store_true",
                    help="add the 'rules' group (Schwartz-Hearst output as features)")
    ap.add_argument("--propagate", choices=["later", "all"], default=None,
                    help="apply document-level propagation to CRF output")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--final", action="store_true", help="train on train+dev before testing")
    ap.add_argument("--test", default=None, help="gold test .jsonl for the final evaluation")
    ap.add_argument("--model-out", default=str(MODEL_DIR / "crf.pkl"))
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--errors", default=None, help="write dev error examples to this file")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if args.with_rules and "rules" not in groups:
        groups.append("rules")

    train, dev = load_split("train", args.data), load_split("dev", args.data)
    print(f"Train: {len(train)} sentences | Dev: {len(dev)} sentences | groups: {', '.join(groups)}")
    if not crfsuite_available() and args.backend != "simple":
        print("[info] python-crfsuite not installed -> using the numpy fallback CRF "
              "(pip install python-crfsuite for the faster backend)")

    if args.ablation:
        rows = run_ablation(train, dev, groups, args)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "crf_ablation.json", "w") as f:
            json.dump({name: {k: s[k] for k in ("abbreviation", "long_form", "pair")}
                       for name, s in rows}, f, indent=2)
        return
    if args.search:
        args.c1, args.c2, _ = run_search(train, dev, groups, args)

    fit_on = train + dev if args.final else train
    model, preds, scores = train_and_eval(fit_on, dev, groups, args, propagation=args.propagate)
    name = "CRF" + (" + rules" if "rules" in groups else "") + (" + propagation" if args.propagate else "")
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

    if args.test:
        test = load_split(args.test)
        test_preds = model.predict(test)
        print(format_report(f"{name} on TEST ({args.test})", evaluate(test, test_preds), token_level=True))
        if args.save_preds:
            save_jsonl([{"id": r["id"], "labels": p} for r, p in zip(test, test_preds)],
                       RESULTS_DIR / "crf.test.jsonl")


if __name__ == "__main__":
    main()
