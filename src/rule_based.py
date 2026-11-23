"""Technique 1: rule-based definition extraction.

Re-implements the training-free algorithm of Schwartz & Hearst (2003) on
tokenised sentences, plus our extension, document-level propagation.

  1. Find candidates: "long form ( short form )" or "short form ( long form )".
     The second pattern is used when the parentheses hold more than two words.
     Text inside the parentheses is cut at the first ',' or ';'.
  2. Filter short forms: at most two words, 2-10 characters, contains a
     letter, starts with a letter or digit. Because B-AC is a single token in
     our label scheme, a short form must also be exactly one token.
  3. Limit the long-form window to min(|SF| + 5, |SF| * 2) words before the
     short form (|SF| = number of characters). The window does not cross
     another parenthesis.
  4. Match right to left: every short-form character, in order, must match
     a long-form character; the first one must start a word.
  5. Reject long forms shorter than the short form or containing it as a word.

Extension: once a short form is defined, later uses in the same document
are labelled B-AC too (plural forms such as CNNs included).

Optional extras (off by default, reported separately):
  --glossary       also read "SF = long form" / "SF : long form" definitions,
                   common in CS tables and legends (~12% of SDU long forms).
                   The same character-matching rules are applied.
  --with-detector  also label abbreviations that are never defined, using
                   the heuristic detector from baseline.py.

Usage:
    python src/rule_based.py                          # dev split, with propagation
    python src/rule_based.py --no-propagate           # plain Schwartz & Hearst
    python src/rule_based.py --with-detector          # + undefined abbreviations
    python src/rule_based.py --glossary --with-detector
    python src/rule_based.py --split data/processed/test.jsonl --save-preds
    python src/rule_based.py --compare                # all variants side by side
"""

import argparse
from collections import defaultdict

from baseline import detect_abbreviations
from evaluate import dump_errors, evaluate, format_report, summary_row
from utils import (OUTSIDE, RESULTS_DIR, doc_key, is_word, load_lexicon, load_split,
                   save_jsonl, sh_match_tokens, strip_plural)

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
    """Schwartz & Hearst on one tokenised sentence.

    Returns a list of (short_form_index, (lf_start, lf_end)) token spans.
    """
    pairs = []
    for i, t in enumerate(tokens):
        if t != "(":
            continue
        j = _closing_paren(tokens, i)
        if j is None or j == i + 1:
            continue
        inner_end = j
        for k in range(i + 1, j):
            if tokens[k] in (",", ";"):
                inner_end = k
                break
        inner = tokens[i + 1:inner_end]
        if not inner:
            continue
        n_inner_words = sum(is_word(x) for x in inner)

        if n_inner_words > 2:
            # Pattern "short form ( long form )"
            sf_idx = i - 1
            if sf_idx < 0 or not valid_short_form([tokens[sf_idx]]):
                continue
            sf = tokens[sf_idx]
            m = sh_match_tokens(sf, inner)
            if m is None:
                continue
            lf_start, lf_end = _trim(tokens, i + 1 + m, inner_end)
        else:
            # Pattern "long form ( short form )"
            if not valid_short_form(inner):
                continue
            sf_idx = i + 1
            sf = tokens[sf_idx]
            n = min(len(sf) + 5, len(sf) * 2)
            w_start = _window_start(tokens, i, n)
            window = tokens[w_start:i]
            m = sh_match_tokens(sf, window)
            if m is None:
                continue
            lf_start, lf_end = _trim(tokens, w_start + m, i)

        if _accept(tokens, sf, lf_start, lf_end):
            pairs.append((sf_idx, (lf_start, lf_end)))
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
    """Document-level propagation: label other uses of a defined short form
    (and its plural / singular) as B-AC.

    scope="later": only uses after the definition (as in the proposal).
    scope="all":   every use in the document.
    """
    by_doc = defaultdict(list)
    for idx, rec in enumerate(records):
        by_doc[doc_key(rec)].append(idx)

    out = [list(p) for p in predictions]
    for idxs in by_doc.values():
        # (record position, token position) of the first definition of each form
        first_def = {}
        for order, idx in enumerate(idxs):
            toks, labs = records[idx]["tokens"], out[idx]
            for k, (t, l) in enumerate(zip(toks, labs)):
                if l == "B-AC" and t not in first_def:
                    first_def[t] = (order, k)
        forms = {}
        for sf, pos in first_def.items():
            for variant in {sf, strip_plural(sf), sf + "s"}:
                if variant not in forms or pos < forms[variant]:
                    forms[variant] = pos
        for order, idx in enumerate(idxs):
            toks, labs = records[idx]["tokens"], out[idx]
            for k, t in enumerate(toks):
                if labs[k] == OUTSIDE and t in forms and (scope == "all" or (order, k) > forms[t]):
                    labs[k] = "B-AC"
    return out


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
            ("S&H", dict(propagation=None)),
            ("S&H + propagation", dict(propagation="later")),
            ("S&H + propagation(all)", dict(propagation="all")),
            ("S&H + propagation + detector", dict(propagation="later", with_detector=True)),
            ("S&H + propagation + glossary", dict(propagation="later", glossary=True)),
            ("S&H + prop. + glossary + detector", dict(propagation="later", glossary=True, with_detector=True)),
        ]
        print(f"\n{'system':34}{'AC F1':>8}{'LF F1':>8}{'Pair F1':>8}")
        for name, kw in variants:
            print(summary_row(name, evaluate(records, predict(records, **kw))))
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
