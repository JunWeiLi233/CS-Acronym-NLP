"""Shared helpers: loading data, train/dev split, BIO -> spans, the
Schwartz-Hearst character matcher, abbreviation <-> long-form linking,
and the English word list.

Every model in this project (baselines, rule-based, CRF) produces one
label per token from the scheme of Zilio et al. (2024):

    B-AC  abbreviation (always a single token)
    B-LF  first token of a long form
    I-LF  later token of a long form
    B-O   any other token
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

LABELS = ["B-AC", "B-LF", "I-LF", "B-O"]
OUTSIDE = "B-O"

_WORD_RE = re.compile(r"\w", re.UNICODE)


# --------------------------------------------------------------------------
# Data loading and splitting
# --------------------------------------------------------------------------

def default_data_path():
    """The clean dataset if prepare_data.py produced one, else the partial one."""
    clean = DATA_DIR / "SDU-train-data.jsonl"
    return clean if clean.exists() else DATA_DIR / "SDU-train-data.partial.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def doc_key(record):
    """Document a sentence belongs to.

    SDU has no paper ids, so each sentence is its own document. If a record
    carries a "doc_id" (e.g. our own annotated test papers), sentences from
    the same paper are grouped together.
    """
    return str(record.get("doc_id", record["id"]))


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


# --------------------------------------------------------------------------
# Token helpers
# --------------------------------------------------------------------------

def is_word(token):
    return bool(_WORD_RE.search(token))


def upper_count(token):
    return sum(c.isupper() for c in token)


# Roman numerals made only of I, V, X (section numbers, "Table IV", ...).
# We deliberately leave out C, D, L, M because CC, CD, DC, LC, MD are real
# abbreviations in CS text (e.g. the POS tag CC).
ROMAN_IVX = re.compile(r"^(?=[IVX])(X{0,3})(IX|IV|V?I{0,3})$")
ROMAN_FULL = re.compile(
    r"^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


# --------------------------------------------------------------------------
# BIO <-> spans
# --------------------------------------------------------------------------

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
    labels = [OUTSIDE] * n
    for s, e in lfs:
        labels[s] = "B-LF"
        for i in range(s + 1, e):
            labels[i] = "I-LF"
    for s, e in acs:
        labels[s] = "B-AC"
    return labels


# --------------------------------------------------------------------------
# Schwartz & Hearst (2003) character matching
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Linking abbreviations to long forms (used for pair evaluation)
# --------------------------------------------------------------------------

def link_pairs(tokens, labels, max_gap=4):
    """Link each abbreviation span to a long-form span in the same sentence.

    Preference order for each abbreviation: long forms whose characters
    match (S&H), then the nearest one. A non-matching long form is only
    accepted when it is within `max_gap` tokens (e.g. "LF ( AC )" has a gap
    of 1, "AC = LF" a gap of 1). Each long form is used at most once.
    Applied identically to gold and predicted labels so pairing scores are
    comparable across systems.
    """
    acs, lfs = spans_from_labels(labels)
    used, pairs = set(), []
    for a in acs:
        best = None
        for lf in lfs:
            if lf in used:
                continue
            gap = a[0] - lf[1] if lf[1] <= a[0] else lf[0] - a[1]
            if gap < 0:
                continue
            match = chars_match(tokens[a[0]], tokens[lf[0]:lf[1]])
            if not match and gap > max_gap:
                continue
            key = (not match, gap)
            if best is None or key < best[0]:
                best = (key, lf)
        if best is not None:
            used.add(best[1])
            pairs.append((a, best[1]))
    return pairs


def pair_strings(tokens, labels):
    """(abbreviation, long form) strings, normalised for comparison."""
    out = set()
    for (a, lf) in link_pairs(tokens, labels):
        ac_text = tokens[a[0]]
        lf_text = " ".join(tokens[lf[0]:lf[1]]).lower()
        out.add((ac_text, lf_text))
    return out


# --------------------------------------------------------------------------
# English dictionary (Kreuzthaler et al., 2016)
# --------------------------------------------------------------------------

_LEXICON_CACHE = {}


def load_lexicon(path=None):
    path = Path(path or LEXICON_PATH)
    if path not in _LEXICON_CACHE:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                _LEXICON_CACHE[path] = {w.strip().lower() for w in f if w.strip()}
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
