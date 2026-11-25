"""Extract abbreviation definitions with regex patterns over tokenized text.

The initials of consecutive words or hyphenated components must spell the
abbreviation. Common connecting words may be skipped between initials.
"""

import argparse
import re
from functools import lru_cache

from evaluate import evaluate, format_report
from rule_based import propagate
from utils import OUTSIDE, ROMAN_IVX, labels_from_spans, load_split, strip_plural

PARENTHESES = re.compile(r"(?<!\S)\( (?P<inside>[^()]+?) \)(?!\S)")
GLOSSARY = re.compile(r"(?<!\S)(?P<short>\S+) [:=](?= )")
SHORT_FORM = re.compile(r"(?=[A-Za-z0-9-]{2,10}$)(?=[A-Za-z0-9-]*[A-Za-z])"
                        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
UNDEFINED_FORM = re.compile(
    r"(?:[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)*s?"
    r"|[a-z]-?[A-Z]{2,}s?"
    r"|(?=[A-Za-z0-9-]*[A-Z])(?=[A-Za-z0-9-]*[0-9])[A-Za-z0-9-]+)"
)
UPPERCASE_HEADING = re.compile(r"(?:[IVX]+ \. )?(?:[A-Z][A-Z-]* ){3,}[A-Z][A-Z-]*(?: [.!?])?")

CONNECTOR = r"[ -](?:(?:a|an|and|by|for|in|of|on|or|the|to|with)[ -]){0,2}"
WORD_TAIL = r"[^\W_]*(?:'[^\W_]+)*"
MAX_EXPANSION_TOKENS = 16


@lru_cache(maxsize=1024)
def _expansion_pattern(short, left_anchor, right_anchor):
    initials = strip_plural(short.replace("-", ""))
    words = CONNECTOR.join(re.escape(initial) + WORD_TAIL for initial in initials)
    return re.compile(left_anchor + r"(?<!\S)" + words + r"(?!\S)" + right_anchor, re.IGNORECASE)


def _match_expansion(short, tokens, start, end, left_anchor="", right_anchor=""):
    if not SHORT_FORM.fullmatch(short):
        return None
    text = " ".join(tokens[start:end])
    match = _expansion_pattern(short, left_anchor, right_anchor).search(text)
    if match is None:
        return None
    # Each space in the joined text marks a token boundary.
    first_token = start + text.count(" ", 0, match.start())
    after_last_token = start + text.count(" ", 0, match.end()) + 1
    return first_token, after_last_token


def _definitions(tokens, text):
    token_at_offset = {
        match.start(): index for index, match in enumerate(re.finditer(r"\S+", text))
    }
    for match in PARENTHESES.finditer(text):
        opening = token_at_offset[match.start()]
        inside = match["inside"]
        closing = opening + inside.count(" ") + 2
        expansion = _match_expansion(
            inside, tokens, max(0, opening - MAX_EXPANSION_TOKENS), opening, right_anchor="$",
        )
        if expansion is not None:
            yield opening + 1, expansion
        elif opening > 0 and closing - opening <= MAX_EXPANSION_TOKENS + 1:
            expansion = _match_expansion(
                tokens[opening - 1], tokens, opening + 1, closing, left_anchor="^", right_anchor="$",
            )
            if expansion is not None:
                yield opening - 1, expansion

    for match in GLOSSARY.finditer(text):
        short_idx = token_at_offset[match.start()]
        start = short_idx + 2
        expansion = _match_expansion(
            match["short"], tokens, start, start + MAX_EXPANSION_TOKENS, left_anchor="^",
        )
        if expansion is not None:
            yield short_idx, expansion


def _definition_labels(tokens, text):
    abbreviations, long_forms = [], []
    used_positions = set()
    for short_index, (start, end) in _definitions(tokens, text):
        positions = set(range(start, end))
        if (
            short_index in positions
            or short_index in used_positions
            or not used_positions.isdisjoint(positions)
        ):
            continue
        abbreviations.append((short_index, short_index + 1))
        long_forms.append((start, end))
        used_positions.update(positions)
        used_positions.add(short_index)
    return labels_from_spans(len(tokens), abbreviations, long_forms)


def predict(records):
    """Return BIO labels in the same order as the input records and tokens."""
    sentences = [record["tokens"] for record in records]
    texts = [" ".join(tokens) for tokens in sentences]
    labels = [_definition_labels(tokens, text) for tokens, text in zip(sentences, texts)]
    predictions = propagate(records, labels, scope="later")
    for tokens, text, predicted in zip(sentences, texts, predictions):
        if UPPERCASE_HEADING.fullmatch(text):
            continue
        for index, token in enumerate(tokens):
            if predicted[index] != OUTSIDE:
                continue
            if (
                SHORT_FORM.fullmatch(token)
                and UNDEFINED_FORM.fullmatch(token)
                and not ROMAN_IVX.fullmatch(token)
            ):
                predicted[index] = "B-AC"
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", help="train | dev | all | path to gold JSONL")
    parser.add_argument("--data", default=None, help="dataset used for the train/dev split")
    args = parser.parse_args()

    records = load_split(args.split, args.data)
    print(format_report(f"Regex on {args.split}", evaluate(records, predict(records))))


if __name__ == "__main__":
    main()
