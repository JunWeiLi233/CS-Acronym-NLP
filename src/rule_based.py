"""Find parenthetical abbreviation definitions with Schwartz–Hearst matching.

The matcher works from right to left and requires the first character of the
short form to start a word. Short forms occupy one token in our label scheme.
The detector can also label later mentions within the same document.
"""

import argparse
from collections import defaultdict

from baseline import detect_abbreviations
from evaluate import (
    SPAN_F1_HEADERS,
    dump_errors,
    evaluate,
    format_report,
    span_f1_values,
)
from terminal_table import format_table
from utils import (
    OUTSIDE,
    RESULTS_DIR,
    doc_key,
    is_word,
    load_lexicon,
    load_split,
    save_jsonl,
    sh_match_tokens,
    strip_plural,
)

MAX_PAREN_TOKENS = 15


def valid_short_form(sf_tokens):
    if len(sf_tokens) != 1:
        return False
    sf = sf_tokens[0]
    return (2 <= len(sf) <= 10
            and any(c.isalpha() for c in sf)
            and sf[0].isalnum())


def _closing_paren(tokens, i):
    depth = 0
    for j in range(i, min(len(tokens), i + MAX_PAREN_TOKENS + 2)):
        if tokens[j] == "(":
            depth += 1
        elif tokens[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return None


def _trim(tokens, start, end):
    """Drop leading/trailing punctuation tokens from a long-form span."""
    while start < end and not is_word(tokens[start]):
        start += 1
    while end > start and not is_word(tokens[end - 1]):
        end -= 1
    return start, end


def _window_start(tokens, end, n_words):
    """Index where a window of `n_words` word tokens ending before `end` starts,
    without crossing a parenthesis."""
    i, words = end, 0
    while i > 0 and words < n_words:
        if tokens[i - 1] in ("(", ")"):
            break
        i -= 1
        if is_word(tokens[i]):
            words += 1
    return i


def _accept(tokens, sf, lf_start, lf_end):
    if lf_end <= lf_start:
        return False
    lf_tokens = tokens[lf_start:lf_end]
    if len(" ".join(lf_tokens)) < len(sf):
        return False
    lower = {t.lower() for t in lf_tokens}
    return sf.lower() not in lower and strip_plural(sf).lower() not in lower


def find_definitions(tokens):
    """Find definitions in a tokenized sentence.

    Each result contains the short-form index and a half-open long-form span.
    """
    pairs = []
    for opening, token in enumerate(tokens):
        if token != "(":
            continue
        closing = _closing_paren(tokens, opening)
        if closing is None or closing == opening + 1:
            continue
        inner_end = closing
        for index in range(opening + 1, closing):
            if tokens[index] in (",", ";"):
                inner_end = index
                break
        inner = tokens[opening + 1:inner_end]
        if not inner:
            continue

        if sum(is_word(token) for token in inner) > 2:
            # Pattern "short form ( long form )"
            short_index = opening - 1
            if short_index < 0 or not valid_short_form([tokens[short_index]]):
                continue
            window_start, window_end = opening + 1, inner_end
        else:
            # Pattern "long form ( short form )"
            if not valid_short_form(inner):
                continue
            short_index = opening + 1
            short_length = len(tokens[short_index])
            max_words = min(short_length + 5, 2 * short_length)
            window_start, window_end = _window_start(tokens, opening, max_words), opening

        short_form = tokens[short_index]
        match_start = sh_match_tokens(short_form, tokens[window_start:window_end])
        if match_start is None:
            continue
        long_start, long_end = _trim(tokens, window_start + match_start, window_end)
        if _accept(tokens, short_form, long_start, long_end):
            pairs.append((short_index, (long_start, long_end)))
    return pairs


GLOSSARY_SEPARATORS = ("=", ":")
GLOSSARY_STOP = {",", ";", ".", "(", ")", "=", ":", "?"}


def find_glossary_definitions(tokens):
    """Extension: "SF = long form" and "SF : long form".

    The long form runs from the token after the separator up to the next
    delimiter (or the next "X =" entry), capped at min(|SF|+5, 2|SF|) words.
    It must match the short form with the S&H matcher starting at its first
    word; trailing words whose first letter is not in the short form are
    dropped ("NLP = Natural Language Processing researcher").
    """
    pairs = []
    for i, t in enumerate(tokens):
        if t not in GLOSSARY_SEPARATORS or i == 0:
            continue
        sf = tokens[i - 1]
        if not valid_short_form([sf]) or not any(c.isupper() for c in sf):
            continue
        n = min(len(sf) + 5, len(sf) * 2)
        end, words = i + 1, 0
        while end < len(tokens) and tokens[end] not in GLOSSARY_STOP and words < n:
            if end + 1 < len(tokens) and tokens[end + 1] in GLOSSARY_SEPARATORS and end > i + 1:
                break
            words += is_word(tokens[end])
            end += 1
        start, end = _trim(tokens, i + 1, end)
        if sh_match_tokens(sf, tokens[start:end]) != 0:
            continue
        letters = {c.lower() for c in sf if c.isalnum()}
        while end - start > 1 and tokens[end - 1][0].lower() not in letters:
            end -= 1
        start, end = _trim(tokens, start, end)
        if sh_match_tokens(sf, tokens[start:end]) == 0 and _accept(tokens, sf, start, end):
            pairs.append((i - 1, (start, end)))
    return pairs


def label_sentence(tokens, glossary=False):
    """S&H labels for one sentence, plus the defined short forms."""
    labels = [OUTSIDE] * len(tokens)
    defined = []
    candidates = find_definitions(tokens)
    if glossary:
        candidates += find_glossary_definitions(tokens)
    for sf_idx, (s, e) in candidates:
        span = range(s, e)
        if labels[sf_idx] != OUTSIDE or any(labels[k] != OUTSIDE for k in span) or sf_idx in span:
            continue
        labels[sf_idx] = "B-AC"
        labels[s] = "B-LF"
        for k in range(s + 1, e):
            labels[k] = "I-LF"
        defined.append((sf_idx, tokens[sf_idx]))
    return labels, defined


def propagate(records, predictions, scope="later"):
    """Copy B-AC labels to other mentions of the same form within a document.

    The first labeled mention sets the starting point for scope="later".
    Scope "all" also reaches earlier mentions. Only B-O tokens are relabeled.
    """
    document_indices = defaultdict(list)
    for index, record in enumerate(records):
        document_indices[doc_key(record)].append(index)

    propagated = [list(labels) for labels in predictions]
    for indices in document_indices.values():
        first_mention = {}
        for sentence_index in indices:
            tokens = records[sentence_index]["tokens"]
            for token_index, (token, label) in enumerate(zip(tokens, propagated[sentence_index])):
                if label == "B-AC":
                    for form in (token, strip_plural(token), token + "s"):
                        first_mention.setdefault(form, (sentence_index, token_index))

        for sentence_index in indices:
            labels = propagated[sentence_index]
            for token_index, token in enumerate(records[sentence_index]["tokens"]):
                first = first_mention.get(token)
                if (
                    labels[token_index] == OUTSIDE
                    and first is not None
                    and (scope == "all" or (sentence_index, token_index) > first)
                ):
                    labels[token_index] = "B-AC"
    return propagated


def predict(records, propagation="later", with_detector=False, glossary=False, lexicon=None):
    """Label a list of records (dicts with "tokens", "id", optional "doc_id")."""
    preds = [label_sentence(r["tokens"], glossary=glossary)[0] for r in records]
    if propagation:
        preds = propagate(records, preds, scope=propagation)
    if with_detector:
        lexicon = load_lexicon() if lexicon is None else lexicon
        for r, p in zip(records, preds):
            for k in detect_abbreviations(r["tokens"], lexicon):
                if p[k] == OUTSIDE:
                    p[k] = "B-AC"
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", help="train | dev | all | path to .jsonl")
    ap.add_argument("--data", default=None, help="dataset used for train/dev split")
    ap.add_argument("--no-propagate", action="store_true", help="plain Schwartz & Hearst")
    ap.add_argument("--propagate-scope", choices=["later", "all"], default="later")
    ap.add_argument("--with-detector", action="store_true",
                    help="also label undefined abbreviations with the heuristic detector")
    ap.add_argument("--glossary", action="store_true", help='also use "SF = long form" definitions')
    ap.add_argument("--compare", action="store_true", help="evaluate all variants")
    ap.add_argument("--save-preds", action="store_true", help="write predictions to results/")
    ap.add_argument("--errors", default=None, help="write error examples to this file")
    args = ap.parse_args()

    records = load_split(args.split, args.data)

    if args.compare:
        variants = [
            ("S&H", {"propagation": None}),
            ("S&H + propagation", {"propagation": "later"}),
            ("S&H + propagation(all)", {"propagation": "all"}),
            ("S&H + propagation + detector", {"propagation": "later", "with_detector": True}),
            ("S&H + propagation + glossary", {"propagation": "later", "glossary": True}),
            (
                "S&H + prop. + glossary + detector",
                {"propagation": "later", "glossary": True, "with_detector": True},
            ),
        ]
        rows = ((name, *span_f1_values(evaluate(records, predict(records, **kw))))
                for name, kw in variants)
        print(format_table(
            ("Variant", *SPAN_F1_HEADERS), rows,
            title=f"Rule-based comparison on {args.split} ({len(records):,} sentences)",
            right=range(1, 4), wrap=0,
        ))
        return

    propagation = None if args.no_propagate else args.propagate_scope
    preds = predict(records, propagation=propagation, with_detector=args.with_detector,
                    glossary=args.glossary)
    name = ("S&H" + (" + propagation" if propagation else "") + (" + glossary" if args.glossary else "")
            + (" + detector" if args.with_detector else ""))
    print(format_report(f"Rule-based: {name}", evaluate(records, preds), token_level=True))

    tag = args.split if args.split in ("train", "dev", "all") else "custom"
    if args.save_preds:
        out = RESULTS_DIR / f"rule_based.{tag}.jsonl"
        save_jsonl([{"id": r["id"], "labels": p} for r, p in zip(records, preds)], out)
        print(f"Predictions saved to {out}")
    if args.errors:
        dump_errors(records, preds, args.errors)


if __name__ == "__main__":
    main()
