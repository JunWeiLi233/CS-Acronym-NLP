"""Span-level evaluation, as defined in the proposal (Part 2).

* Abbreviation detection: exact match of the single B-AC token.
* Long-form detection: exact match of the B-LF I-LF* span boundaries.
* Pairing: each distinct (abbreviation, long form) pair is counted once per
  document and is correct only if both strings and their link match gold.
  Links are produced by utils.link_pairs for both gold and predictions.
* Token-level per-label scores are reported too, for error analysis.

Usage (compare a predictions file against gold):
    python src/evaluate.py --gold dev --pred results/rule_based.dev.jsonl
where the predictions file has one {"id": ..., "labels": [...]} per line.
"""

import argparse
import json
from collections import Counter, defaultdict

from utils import LABELS, OUTSIDE, doc_key, load_jsonl, load_split, pair_strings, spans_from_labels


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _score_sets(gold, pred):
    tp = len(gold & pred)
    return prf(tp, len(pred) - tp, len(gold) - tp)


def evaluate(records, predictions):
    """records: gold records (with tokens/labels). predictions: list of label
    lists aligned with records. Returns a dict of scores."""
    assert len(records) == len(predictions), "predictions must align with records"
    gold_ac, pred_ac, gold_lf, pred_lf = set(), set(), set(), set()
    gold_pairs, pred_pairs = defaultdict(set), defaultdict(set)
    tok = Counter()

    for idx, (rec, pred) in enumerate(zip(records, predictions)):
        gold = rec["labels"]
        assert len(gold) == len(pred), f"length mismatch in record {rec['id']}"
        g_ac, g_lf = spans_from_labels(gold)
        p_ac, p_lf = spans_from_labels(pred)
        gold_ac |= {(idx, s) for s in g_ac}
        pred_ac |= {(idx, s) for s in p_ac}
        gold_lf |= {(idx, s) for s in g_lf}
        pred_lf |= {(idx, s) for s in p_lf}

        d = doc_key(rec)
        gold_pairs[d] |= pair_strings(rec["tokens"], gold)
        pred_pairs[d] |= pair_strings(rec["tokens"], pred)

        for g, p in zip(gold, pred):
            if g == p:
                tok[("tp", g)] += 1
            else:
                tok[("fp", p)] += 1
                tok[("fn", g)] += 1

    gp = {(d, x) for d, xs in gold_pairs.items() for x in xs}
    pp = {(d, x) for d, xs in pred_pairs.items() for x in xs}

    token_scores = {lab: prf(tok[("tp", lab)], tok[("fp", lab)], tok[("fn", lab)])
                    for lab in LABELS}
    return {
        "abbreviation": _score_sets(gold_ac, pred_ac),
        "long_form": _score_sets(gold_lf, pred_lf),
        "pair": _score_sets(gp, pp),
        "token": token_scores,
        "n_sentences": len(records),
    }


def format_report(name, scores, token_level=False):
    lines = [f"\n=== {name} ({scores['n_sentences']} sentences) ===",
             f"{'':14}{'P':>8}{'R':>8}{'F1':>8}{'TP':>7}{'FP':>7}{'FN':>7}"]
    for key, label in (("abbreviation", "Abbreviation"), ("long_form", "Long form"), ("pair", "Pair")):
        s = scores[key]
        lines.append(f"{label:14}{s['precision']:8.3f}{s['recall']:8.3f}{s['f1']:8.3f}"
                     f"{s['tp']:7d}{s['fp']:7d}{s['fn']:7d}")
    if token_level:
        lines.append("  token level:")
        for lab in LABELS:
            s = scores["token"][lab]
            lines.append(f"  {lab:12}{s['precision']:8.3f}{s['recall']:8.3f}{s['f1']:8.3f}")
    return "\n".join(lines)


def summary_row(name, scores):
    return (f"{name:34}"
            f"{scores['abbreviation']['f1']:8.3f}{scores['long_form']['f1']:8.3f}{scores['pair']['f1']:8.3f}")


def dump_errors(records, predictions, path, limit=300):
    """Write sentences where the prediction differs from gold, with
    abbreviations/long forms marked, for error analysis on the dev set."""
    def render(tokens, labels):
        out = []
        for t, l in zip(tokens, labels):
            out.append(t if l == OUTSIDE else f"[{t}|{l[2:]}]")
        return " ".join(out)

    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec, pred in zip(records, predictions):
            if rec["labels"] == list(pred):
                continue
            f.write(f"id={rec['id']}\n  GOLD: {render(rec['tokens'], rec['labels'])}\n"
                    f"  PRED: {render(rec['tokens'], pred)}\n\n")
            n += 1
            if n >= limit:
                break
    print(f"Wrote {n} error examples to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="dev", help="train | dev | all | path to gold .jsonl")
    ap.add_argument("--pred", required=True, help="predictions .jsonl with id + labels")
    ap.add_argument("--data", default=None, help="dataset used for train/dev split")
    args = ap.parse_args()

    gold = load_split(args.gold, args.data)
    preds = {str(r["id"]): r["labels"] for r in load_jsonl(args.pred)}
    missing = [r["id"] for r in gold if str(r["id"]) not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} gold sentences have no prediction (e.g. {missing[:5]})")
    scores = evaluate(gold, [preds[str(r["id"])] for r in gold])
    print(format_report(args.pred, scores, token_level=True))
    print(json.dumps({k: scores[k] for k in ("abbreviation", "long_form", "pair")}, indent=2))


if __name__ == "__main__":
    main()
