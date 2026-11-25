"""Shared data, BIO-label, abbreviation-linking, and lexicon helpers.

Models use ``B-AC`` for a one-token abbreviation, ``B-LF`` and ``I-LF``
for a long form, and ``B-O`` for every remaining token.
"""

import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
LEXICON_PATH = ROOT / "data" / "lexicon" / "english_words.txt"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SAVED_CRF_FILENAMES = {
    "plain": "crf-plain.pkl",
    "schwartz_hearst": "crf-schwartz-hearst.pkl",
    "kaggle_lexicon": "crf-kaggle-lexicon.pkl",
}

LABELS = ["B-AC", "B-LF", "I-LF", "B-O"]
OUTSIDE = "B-O"

_WORD_RE = re.compile(r"\w", re.UNICODE)


def default_data_path():
    """The clean dataset if prepare_data.py produced one, else the partial one."""
    clean = DATA_DIR / "SDU-train-data.jsonl"
    return clean if clean.exists() else DATA_DIR / "SDU-train-data.partial.jsonl"


def load_jsonl(path):
    """Read nonblank JSONL records from ``path``."""
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def save_jsonl(rows, path):
    """Write an iterable of records as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def doc_key(record):
    """Document a sentence belongs to.

    SDU has no paper ids, so each sentence is its own document. If a record
    carries a "doc_id" (e.g. our own annotated test papers), sentences from
    the same paper are grouped together.
    """
    return str(record.get("doc_id", record["id"]))


def reject_evaluation_overlap(train, evaluation, train_source):
    """Reject a training source that shares an ID, document, or tokenized sentence."""
    evaluation_ids = {str(record["id"]) for record in evaluation}
    evaluation_docs = {doc_key(record) for record in evaluation}
    evaluation_tokens = {
        tuple(token.casefold() for token in record["tokens"]) for record in evaluation
    }
    for record in train:
        if str(record["id"]) in evaluation_ids or doc_key(record) in evaluation_docs:
            raise ValueError(f"{train_source} overlaps the evaluation IDs (record {record['id']})")
        if tuple(token.casefold() for token in record["tokens"]) in evaluation_tokens:
            raise ValueError(
                f"{train_source} repeats an evaluation sentence (record {record['id']})"
            )


def split_records(records, dev_frac=0.2, seed=13):
    """Deterministic 80/20 train/dev split, grouped by document so no
    document appears in both splits."""
    groups = sorted({doc_key(r) for r in records})
    random.Random(seed).shuffle(groups)
    n_dev = math.ceil(len(groups) * dev_frac)
    dev_groups = set(groups[:n_dev])
    train = [r for r in records if doc_key(r) not in dev_groups]
    dev = [r for r in records if doc_key(r) in dev_groups]
    return train, dev


def load_split(split, data_path=None, dev_frac=0.2, seed=13):
    """split: 'train', 'dev', 'all', or a path to a .jsonl file (e.g. test set)."""
    if split not in ("train", "dev", "all"):
        return load_jsonl(split)
    records = load_jsonl(data_path or default_data_path())
    if split == "all":
        return records
    train, dev = split_records(records, dev_frac, seed)
    return train if split == "train" else dev


def is_word(token):
    """Return whether ``token`` contains a Unicode word character."""
    return bool(_WORD_RE.search(token))


def upper_count(token):
    """Count uppercase characters in ``token``."""
    return sum(c.isupper() for c in token)


# Roman numerals made only of I, V, X (section numbers, "Table IV", ...).
# We deliberately leave out C, D, L, M because CC, CD, DC, LC, MD are real
# abbreviations in CS text (e.g. the POS tag CC).
ROMAN_IVX = re.compile(r"^(?=[IVX])(X{0,3})(IX|IV|V?I{0,3})$")
ROMAN_FULL = re.compile(
    r"^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def spans_from_labels(labels):
    """Return (abbreviation spans, long-form spans) as (start, end) token
    index pairs, end exclusive. An I-LF without a preceding B-LF/I-LF starts
    a new long form (conlleval convention)."""
    acs, lfs = [], []
    start = None
    for i, label in enumerate(list(labels) + [OUTSIDE]):
        if label == "I-LF" and start is not None:
            continue
        if start is not None:
            lfs.append((start, i))
            start = None
        if label in ("B-LF", "I-LF"):
            start = i
        elif label == "B-AC":
            acs.append((i, i + 1))
    return acs, lfs


def labels_from_spans(n, acs, lfs):
    """Create BIO labels for abbreviation and long-form token spans."""
    labels = [OUTSIDE] * n
    for s, e in lfs:
        labels[s] = "B-LF"
        for i in range(s + 1, e):
            labels[i] = "I-LF"
    for s, e in acs:
        labels[s] = "B-AC"
    return labels


def format_labeled_tokens(tokens, labels):
    """Mark the span label next to each labeled token."""
    return " ".join(
        token if label == OUTSIDE else f"[{token}|{label[2:]}]"
        for token, label in zip(tokens, labels)
    )


def sh_best_long_form_start(short_form, long_form):
    """Schwartz & Hearst's matcher.

    Walk the short form right to left and find each character, in order, in
    the long-form candidate (also right to left). The first short-form
    character must be at the start of a word. Returns the character offset
    in `long_form` where the long form starts, or None if there is no match.
    """
    s_index = len(short_form) - 1
    l_index = len(long_form) - 1
    while s_index >= 0:
        c = short_form[s_index].lower()
        if not c.isalnum():
            s_index -= 1
            continue
        while l_index >= 0 and (
            long_form[l_index].lower() != c
            or (s_index == 0 and l_index > 0 and long_form[l_index - 1].isalnum())
        ):
            l_index -= 1
        if l_index < 0:
            return None
        l_index -= 1
        s_index -= 1
    # back up to the start of the word containing the first match
    return long_form.rfind(" ", 0, l_index + 1) + 1


def strip_plural(short_form):
    """'CNNs' -> 'CNN' (plural abbreviations), otherwise unchanged."""
    if len(short_form) > 2 and short_form.endswith("s") and upper_count(short_form[:-1]) >= 1:
        return short_form[:-1]
    return short_form


def sh_match_tokens(short_form, tokens):
    """Run the S&H matcher on a token window. Returns the index of the first
    long-form token inside `tokens`, or None."""
    if not tokens:
        return None
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t) + 1
    text = " ".join(tokens)
    for sf in dict.fromkeys([short_form, strip_plural(short_form)]):
        char_start = sh_best_long_form_start(sf, text)
        if char_start is not None and char_start in starts:
            return starts.index(char_start)
    return None


def chars_match(short_form, long_tokens):
    return sh_match_tokens(short_form, list(long_tokens)) is not None


def link_pairs(tokens, labels, max_gap=4):
    """Link each abbreviation span to a long-form span in the same sentence.

    Prefer character matches, then shorter gaps. A long form without a
    character match must be within max_gap tokens. Each long form can link to
    one abbreviation. Evaluation uses this for predictions and for legacy
    gold records that have no explicit pairs.
    """
    abbreviations, long_forms = spans_from_labels(labels)
    used_long_forms = set()
    pairs = []
    for short_start, short_end in abbreviations:
        best_rank = None
        best_long_form = None
        for long_start, long_end in long_forms:
            long_form = (long_start, long_end)
            if long_form in used_long_forms:
                continue
            gap = max(short_start - long_end, long_start - short_end)
            if gap < 0:
                continue
            matches = chars_match(tokens[short_start], tokens[long_start:long_end])
            if not matches and gap > max_gap:
                continue
            rank = (not matches, gap)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_long_form = long_form
        if best_long_form is not None:
            used_long_forms.add(best_long_form)
            pairs.append(((short_start, short_end), best_long_form))
    return pairs


def pair_strings(tokens, labels):
    """(abbreviation, long form) strings, normalised for comparison."""
    out = set()
    for (a, lf) in link_pairs(tokens, labels):
        ac_text = tokens[a[0]]
        lf_text = " ".join(tokens[lf[0]:lf[1]]).lower()
        out.add((ac_text, lf_text))
    return out


_LEXICON_CACHE = {}


def load_lexicon(path=None):
    """Load and cache the optional English-word lexicon."""
    path = Path(path or LEXICON_PATH)
    if path not in _LEXICON_CACHE:
        if path.exists():
            with open(path, encoding="utf-8") as source:
                _LEXICON_CACHE[path] = {
                    word.strip().lower() for word in source if word.strip()
                }
        else:
            print(f"[warn] word list not found at {path}; dictionary features disabled")
            _LEXICON_CACHE[path] = set()
    return _LEXICON_CACHE[path]


_SUFFIXES = [("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
             ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("est", "")]


def in_lexicon(token, lexicon):
    """True if the token (or a simple inflection of it) is an English word.
    Hyphenated tokens count if every part is a word."""
    if not lexicon:
        return False
    w = token.lower()
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        return bool(parts) and all(in_lexicon(p, lexicon) for p in parts)
    if not w.isalpha():
        return False
    if w in lexicon:
        return True
    for suf, rep in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2 and (w[: -len(suf)] + rep) in lexicon:
            return True
    return False
