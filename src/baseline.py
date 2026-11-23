"""Baselines from the proposal ("Every system we build has the goal of
beating two baselines").

1. Majority class: every token is B-O.
2. Rule-based detector: a token is B-AC if it
     (a) is 2-10 characters long and has at least two uppercase letters, or
     (b) is a short token alone inside parentheses after a candidate phrase
         (a preceding word starts with the token's first letter), or
     (c) is not in an English dictionary and looks abbreviation-like
         (an internal capital or a letter+digit mix), following
         Kreuzthaler et al. (2016).
   Filters for the false positives reported by Vanopstal et al. (2010):
   all-caps titles, name initials, Roman numerals, and single letters used
   as labels ("types A, B and C").
   The detector only finds abbreviations; it never labels long forms.

Usage:
    python src/baseline.py                 # evaluate on the dev split
    python src/baseline.py --split path/to/test.jsonl
"""

import argparse

from evaluate import evaluate, format_report
from utils import (OUTSIDE, ROMAN_IVX, in_lexicon, is_word, load_lexicon,
                   load_split, save_jsonl, upper_count, RESULTS_DIR)


def majority_baseline(tokens):
    return [OUTSIDE] * len(tokens)


def _is_all_caps_title(tokens):
    """True for headings such as 'I. CONSTRUCT THE PROPOSED ANCHORS'."""
    words = [t for t in tokens if any(c.isalpha() for c in t)]
    if len(words) < 4:
        return False
    caps = [t for t in words if t.isupper() and len(t) > 1]
    return len(caps) >= 4 and len(caps) / len(words) >= 0.6


def _is_name_initials(tokens, i):
    """'J . R . Smith', 'Smith , JR .' style initials."""
    t = tokens[i]
    if not (t.isalpha() and t.isupper() and len(t) <= 2):
        return False
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    after = tokens[i + 2] if i + 2 < len(tokens) else ""
    prev = tokens[i - 1] if i > 0 else ""
    prev2 = tokens[i - 2] if i > 1 else ""
    followed_by_name = nxt == "." and after[:1].isupper() and after[1:].islower()
    after_surname = prev == "," and prev2[:1].isupper() and prev2[1:].islower() and nxt == "."
    return followed_by_name or after_surname


def _inside_parens_alone(tokens, i):
    return 0 < i < len(tokens) - 1 and tokens[i - 1] == "(" and tokens[i + 1] == ")"


def _has_candidate_phrase(tokens, i, max_back=None):
    """Some word before the '(' starts with the token's first letter."""
    first = tokens[i][0].lower()
    n = max_back or min(len(tokens[i]) + 5, 2 * len(tokens[i]))
    words = [t for t in tokens[: i - 1] if is_word(t)][-n:]
    return any(w[0].lower() == first for w in words)


def detect_abbreviations(tokens, lexicon=None):
    """Return the set of token indices the heuristic detector marks B-AC."""
    lexicon = load_lexicon() if lexicon is None else lexicon
    title = _is_all_caps_title(tokens)
    found = set()
    for i, t in enumerate(tokens):
        if not (2 <= len(t) <= 10) or not any(c.isalpha() for c in t):
            continue
        # filters
        if ROMAN_IVX.match(t) or _is_name_initials(tokens, i):
            continue
        in_parens = _inside_parens_alone(tokens, i)
        if title and not in_parens:
            continue

        shape_rule = upper_count(t) >= 2
        paren_rule = (in_parens and t[0].isalnum() and not in_lexicon(t, lexicon)
                      and _has_candidate_phrase(tokens, i))
        oov_rule = (not in_lexicon(t, lexicon)
                    and (any(c.isupper() for c in t[1:])
                         or (any(c.isdigit() for c in t) and any(c.isupper() for c in t))))
        if shape_rule or paren_rule or oov_rule:
            found.add(i)
    return found


def rule_detector_baseline(tokens, lexicon=None):
    labels = [OUTSIDE] * len(tokens)
    for i in detect_abbreviations(tokens, lexicon):
        labels[i] = "B-AC"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", help="train | dev | all | path to .jsonl")
    ap.add_argument("--data", default=None, help="dataset used for train/dev split")
    ap.add_argument("--save-preds", action="store_true", help="write predictions to results/")
    args = ap.parse_args()

    records = load_split(args.split, args.data)
    lexicon = load_lexicon()
    systems = {
        "majority": lambda toks: majority_baseline(toks),
        "rule_detector": lambda toks: rule_detector_baseline(toks, lexicon),
    }
    for name, fn in systems.items():
        preds = [fn(r["tokens"]) for r in records]
        print(format_report(f"Baseline: {name}", evaluate(records, preds), token_level=False))
        if args.save_preds:
            tag = args.split if args.split in ("train", "dev", "all") else "custom"
            save_jsonl([{"id": r["id"], "labels": p} for r, p in zip(records, preds)],
                       RESULTS_DIR / f"baseline_{name}.{tag}.jsonl")


if __name__ == "__main__":
    main()
