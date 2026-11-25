"""Baseline systems for abbreviation detection.

The majority baseline labels every token B-O. The heuristic baseline looks for
short tokens with capital letters, definition-like parentheses, or an unusual
spelling. It filters common name initials, Roman numerals, and all-caps headings.
"""

import argparse

from evaluate import evaluate, format_report
from utils import (
    OUTSIDE,
    RESULTS_DIR,
    ROMAN_IVX,
    in_lexicon,
    is_word,
    load_lexicon,
    load_split,
    save_jsonl,
    upper_count,
)


def majority_baseline(tokens):
    """Label every token as outside an abbreviation or long form."""
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
    """Check whether a word before the opening parenthesis has the same initial."""
    abbreviation = tokens[i]
    initial = abbreviation[0].lower()
    window = max_back or min(len(abbreviation) + 5, 2 * len(abbreviation))
    words = [token for token in tokens[: i - 1] if is_word(token)]
    return any(word[0].lower() == initial for word in words[-window:])


def detect_abbreviations(tokens, lexicon=None):
    """Return the set of token indices the heuristic detector marks B-AC."""
    lexicon = load_lexicon() if lexicon is None else lexicon
    all_caps_title = _is_all_caps_title(tokens)
    found = set()
    for index, token in enumerate(tokens):
        if not (2 <= len(token) <= 10) or not any(char.isalpha() for char in token):
            continue
        if ROMAN_IVX.match(token) or _is_name_initials(tokens, index):
            continue
        in_parens = _inside_parens_alone(tokens, index)
        if all_caps_title and not in_parens:
            continue

        if upper_count(token) >= 2:
            found.add(index)
            continue
        if in_lexicon(token, lexicon):
            continue

        follows_candidate_phrase = (
            in_parens and token[0].isalnum() and _has_candidate_phrase(tokens, index)
        )
        has_internal_capital = any(char.isupper() for char in token[1:])
        mixes_digits_and_capitals = (
            any(char.isdigit() for char in token) and any(char.isupper() for char in token)
        )
        if follows_candidate_phrase or has_internal_capital or mixes_digits_and_capitals:
            found.add(index)
    return found


def rule_detector_baseline(tokens, lexicon=None):
    labels = [OUTSIDE] * len(tokens)
    for index in detect_abbreviations(tokens, lexicon):
        labels[index] = "B-AC"
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
        "majority": majority_baseline,
        "rule_detector": lambda tokens: rule_detector_baseline(tokens, lexicon),
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
