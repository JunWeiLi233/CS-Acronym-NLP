"""Span-level evaluation, as defined in the proposal (Part 2).

* Abbreviation detection: exact match of the single B-AC token.
* Long-form detection: exact match of the B-LF I-LF* span boundaries.
* Pairing: each distinct (abbreviation, long form) pair is counted once per
  document and is correct only if both strings and their link match gold.
  Held-out records can carry explicit gold ``pairs``. Existing records use
  utils.link_pairs for gold and predictions.
* Token-level per-label scores are reported too, for error analysis.

Usage (compare a predictions file against gold):
    python src/evaluate.py --gold dev --pred results/rule_based.dev.jsonl
where the predictions file has one {"id": ..., "labels": [...]} per line.
"""

import argparse
import json
from collections import Counter

from terminal_table import format_table
from utils import (
    LABELS,
    doc_key,
    format_labeled_tokens,
    load_jsonl,
    load_split,
    pair_strings,
    spans_from_labels,
)

SPAN_METRICS = (("abbreviation", "Abbreviation"), ("long_form", "Long form"), ("pair", "Pair"))
SCORE_FIELDS = ("precision", "recall", "f1")
SCORE_HEADERS = ("Precision", "Recall", "F1")
SPAN_F1_HEADERS = tuple(f"{label} F1" for _, label in SPAN_METRICS)


def prf(tp, fp, fn):
    """Return micro-averaged scores and the counts behind them."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _score_sets(gold, pred):
    tp = len(gold & pred)
    return prf(tp, len(pred) - tp, len(gold) - tp)


def gold_pair_strings(record):
    """Return explicit gold links when supplied, otherwise infer legacy links."""
    if "pairs" not in record:
        return pair_strings(record["tokens"], record["labels"])

    pairs = record["pairs"]
    if not isinstance(pairs, list):
        raise ValueError(f"pairs for record {record['id']} must be a list")
    normalized = set()
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(text, str) for text in pair)):
            raise ValueError(f"Invalid pair for record {record['id']}: {pair}")
        abbreviation, long_form = pair
        normalized.add((abbreviation, long_form.lower()))
    return normalized


def evaluate(records, predictions):
    """Score exact token spans and distinct document-level abbreviation pairs."""
    assert len(records) == len(predictions), "predictions must align with records"
    gold_abbreviations, predicted_abbreviations = set(), set()
    gold_long_forms, predicted_long_forms = set(), set()
    gold_document_pairs, predicted_document_pairs = set(), set()
    token_counts = Counter()

    for sentence_index, (record, predicted_labels) in enumerate(zip(records, predictions)):
        gold_labels = record["labels"]
        assert len(gold_labels) == len(predicted_labels), f"length mismatch in record {record['id']}"
        gold_short, gold_long = spans_from_labels(gold_labels)
        predicted_short, predicted_long = spans_from_labels(predicted_labels)
        gold_abbreviations.update((sentence_index, span) for span in gold_short)
        predicted_abbreviations.update((sentence_index, span) for span in predicted_short)
        gold_long_forms.update((sentence_index, span) for span in gold_long)
        predicted_long_forms.update((sentence_index, span) for span in predicted_long)

        document = doc_key(record)
        gold_document_pairs.update((document, pair) for pair in gold_pair_strings(record))
        predicted_pairs = pair_strings(record["tokens"], predicted_labels)
        predicted_document_pairs.update((document, pair) for pair in predicted_pairs)

        for gold, predicted in zip(gold_labels, predicted_labels):
            if gold == predicted:
                token_counts["tp", gold] += 1
            else:
                token_counts["fp", predicted] += 1
                token_counts["fn", gold] += 1

    token_scores = {
        label: prf(token_counts["tp", label], token_counts["fp", label], token_counts["fn", label])
        for label in LABELS
    }
    return {
        "abbreviation": _score_sets(gold_abbreviations, predicted_abbreviations),
        "long_form": _score_sets(gold_long_forms, predicted_long_forms),
        "pair": _score_sets(gold_document_pairs, predicted_document_pairs),
        "token": token_scores,
        "n_sentences": len(records),
    }


def format_report(name, scores, token_level=False):
    rows = []
    for key, label in SPAN_METRICS:
        metric = scores[key]
        rows.append((label, *score_values(metric), metric["tp"], metric["fp"], metric["fn"]))
    tables = [format_table(
        ("Metric", *SCORE_HEADERS, "TP", "FP", "FN"), rows,
        title=f"{name} ({scores['n_sentences']:,} sentences)", right=range(1, 7),
    )]
    if token_level:
        tables.append(format_table(
            ("Token label", *SCORE_HEADERS),
            ((label, *score_values(scores["token"][label])) for label in LABELS),
            title="Token-level scores", right=range(1, 4),
        ))
    return "\n" + "\n\n".join(tables)


def score_values(metrics):
    return tuple(f"{metrics[field]:.3f}" for field in SCORE_FIELDS)


def span_f1_values(scores):
    return tuple(f"{scores[key]['f1']:.3f}" for key, _ in SPAN_METRICS)


def dump_errors(records, predictions, path, limit=300):
    """Write mismatched sentences with their gold and predicted spans marked."""
    written = 0
    with open(path, "w", encoding="utf-8") as output:
        for record, predicted_labels in zip(records, predictions):
            gold_labels = record["labels"]
            if gold_labels == list(predicted_labels):
                continue
            tokens = record["tokens"]
            output.write(
                f"id={record['id']}\n"
                f"  GOLD: {format_labeled_tokens(tokens, gold_labels)}\n"
                f"  PRED: {format_labeled_tokens(tokens, predicted_labels)}\n\n"
            )
            written += 1
            if written >= limit:
                break
    print(f"Wrote {written} error examples to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="dev", help="train | dev | all | path to gold .jsonl")
    ap.add_argument("--pred", required=True, help="predictions .jsonl with id + labels")
    ap.add_argument("--data", default=None, help="dataset used for train/dev split")
    ap.add_argument("--json", action="store_true", help="output span and pair scores as JSON")
    args = ap.parse_args()

    gold = load_split(args.gold, args.data)
    preds = {str(r["id"]): r["labels"] for r in load_jsonl(args.pred)}
    missing = [r["id"] for r in gold if str(r["id"]) not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} gold sentences have no prediction (e.g. {missing[:5]})")
    scores = evaluate(gold, [preds[str(r["id"])] for r in gold])
    if args.json:
        print(json.dumps({key: scores[key] for key, _ in SPAN_METRICS}, indent=2))
    else:
        print(format_report(f"Evaluation of {args.pred} on {args.gold}", scores, token_level=True))


if __name__ == "__main__":
    main()
