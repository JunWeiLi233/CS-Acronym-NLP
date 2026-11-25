"""Token features for the CRF, grouped for train_crf.py --ablation.

The feature design draws on Vanopstal et al. (2010).

  form      lowercased token, prefixes/suffixes up to 4 chars, word shape
            ("XXX", "Xxxxx", "x-XX") and a collapsed shape ("X", "Xx", "x-X")
  ortho     all caps, internal capitals, initial capital, digits, hyphen,
            vowel, Roman numeral, punctuation, length, plural abbreviation
  context   the form/ortho/dict/pos features of the two tokens on each side
  paren     inside parentheses, next to "(" / ")", distance to the next "("
  pos       spaCy part-of-speech tags (only if spaCy is installed)
  dict      token is in an English word list (Kreuzthaler et al., 2016)
  csit      token or its singular form is in the Kaggle CS & IT acronym lexicon
  defmatch  definition match: the token's first letter starts a word within
            the preceding 3 x N words (N = token length); for long-form
            tokens, whether a following parenthesised short form starts
            with / contains the token's first letter
  rules     short-form, long-form, and repetition cues from Schwartz–Hearst
            and glossary matches

Both CRF backends accept strings, booleans, and numbers as feature values.
"""

import json
import re

from rule_based import find_definitions, find_glossary_definitions
from utils import (
    ROMAN_FULL,
    ROOT,
    in_lexicon,
    is_word,
    load_lexicon,
    strip_plural,
    upper_count,
)

ALL_GROUPS = ("form", "ortho", "context", "paren", "pos", "dict", "defmatch", "rules", "csit")
PROPOSAL_GROUPS = ("form", "ortho", "context", "paren", "pos", "dict", "defmatch")
CSIT_LEXICON_PATH = ROOT / "data" / "lexicon" / "csit_acronyms.json"

VOWELS = set("aeiouAEIOU")
NEIGHBOR_FEATURES = (
    "lower", "sshape", "all_caps", "init_cap", "internal_cap", "has_digit",
    "punct", "in_dict", "oov", "pos", "rule_lf", "rule_sf",
)


def word_shape(token):
    shape = []
    for char in token:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        else:
            shape.append(char)
    return "".join(shape)


def short_shape(token):
    return re.sub(r"(.)\1+", r"\1", word_shape(token))


def _bucket(value, cap):
    return str(value) if value <= cap else f">{cap}"


class FeatureExtractor:
    def __init__(self, groups=PROPOSAL_GROUPS, lexicon=None):
        self.groups = set(groups)
        unknown = self.groups - set(ALL_GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups: {unknown}")
        self.lexicon = load_lexicon() if lexicon is None else lexicon
        if "csit" in self.groups:
            with CSIT_LEXICON_PATH.open(encoding="utf-8") as source:
                self.csit_acronyms = frozenset(json.load(source))

    def _token_features(self, token, pos_tag=None, tag=None):
        features = {}
        if "form" in self.groups:
            features["lower"] = token.lower()
            features["shape"] = word_shape(token)[:12]
            features["sshape"] = short_shape(token)
            for length in range(1, min(len(token), 4) + 1):
                features[f"pre{length}"] = token[:length]
                features[f"suf{length}"] = token[-length:]
        if "ortho" in self.groups:
            letters = [char for char in token if char.isalpha()]
            if letters and all(char.isupper() for char in letters):
                features["all_caps"] = True
            if token[:1].isupper():
                features["init_cap"] = True
            if any(char.isupper() for char in token[1:]):
                features["internal_cap"] = True
            if any(char.isdigit() for char in token):
                features["has_digit"] = True
            if "-" in token:
                features["has_hyphen"] = True
            if any(char in VOWELS for char in token):
                features["has_vowel"] = True
            if ROMAN_FULL.match(token):
                features["roman"] = True
            if not is_word(token):
                features["punct"] = True
            features["len"] = _bucket(len(token), 10)
            features["n_upper"] = _bucket(upper_count(token), 4)
            if len(token) > 2 and token.endswith("s") and upper_count(token[:-1]) >= 2:
                features["plural_abbr"] = True
        if "dict" in self.groups and is_word(token):
            if in_lexicon(token, self.lexicon):
                features["in_dict"] = True
            else:
                features["oov"] = True
        if "csit" in self.groups and (
            token in self.csit_acronyms or strip_plural(token) in self.csit_acronyms
        ):
            features["in_csit"] = True
        if "pos" in self.groups and pos_tag is not None:
            features["pos"] = pos_tag
            if tag is not None:
                features["tag"] = tag
        return features

    def _paren_features(self, tokens):
        n = len(tokens)
        inside = [False] * n
        depth = 0
        for i, t in enumerate(tokens):
            if t == ")":
                depth = max(0, depth - 1)
            if depth > 0:
                inside[i] = True
            if t == "(":
                depth += 1
        next_open = [None] * n
        nxt = None
        for i in range(n - 1, -1, -1):
            next_open[i] = None if nxt is None else nxt - i
            if tokens[i] == "(":
                nxt = i
        feats = []
        for i in range(n):
            f = {}
            if inside[i]:
                f["in_paren"] = True
            if i > 0 and tokens[i - 1] == "(":
                f["after_open"] = True
            if i + 1 < n and tokens[i + 1] == ")":
                f["before_close"] = True
            if i + 1 < n and tokens[i + 1] == "(":
                f["before_open"] = True
            if i > 0 and tokens[i - 1] in ("=", ":"):
                f["after_eq"] = True
            if i + 1 < n and tokens[i + 1] in ("=", ":"):
                f["before_eq"] = True
            f["next_open"] = "none" if next_open[i] is None else _bucket(next_open[i], 8)
            feats.append(f)
        return feats

    def _defmatch_features(self, tokens):
        n = len(tokens)
        words_before = []  # first letters of word tokens seen so far
        feats = [{} for _ in range(n)]
        for i, t in enumerate(tokens):
            if is_word(t) and t[0].isalpha() and 2 <= len(t) <= 10:
                window = words_before[-3 * len(t):]
                if t[0].lower() in window:
                    feats[i]["def_first_letter"] = True
            if is_word(t):
                words_before.append(t[0].lower())

        # long-form side: is there a short, parenthesised token ahead whose
        # first letter / letters relate to this token?
        short_ahead = []  # (position, token) for "( X )" candidates
        for i in range(1, n - 1):
            if tokens[i - 1] == "(" and tokens[i + 1] == ")" and 2 <= len(tokens[i]) <= 10:
                short_ahead.append((i, tokens[i]))
        for i, t in enumerate(tokens):
            if not (is_word(t) and t[0].isalpha()):
                continue
            c = t[0].lower()
            for pos, sf in short_ahead:
                if 0 < pos - i <= 2 * len(sf) + 2:
                    if sf[0].lower() == c:
                        feats[i]["sf_ahead_starts"] = True
                    if c in sf.lower():
                        feats[i]["sf_ahead_contains"] = True
                    break
        return feats

    def _rule_features(self, tokens):
        feats = [{} for _ in tokens]
        for sf_idx, (s, e) in find_definitions(tokens) + find_glossary_definitions(tokens):
            feats[sf_idx]["rule_sf"] = True
            feats[s]["rule_lf"] = "B"
            for k in range(s + 1, e):
                feats[k]["rule_lf"] = "I"
        defined = {tokens[i] for i, f in enumerate(feats) if f.get("rule_sf")}
        for i, t in enumerate(tokens):
            if t in defined and not feats[i].get("rule_sf"):
                feats[i]["rule_sf_repeat"] = True
        return feats

    def sent2features(self, tokens, pos=None, tags=None):
        """Build one feature dictionary per token, in sentence order."""
        use_pos = "pos" in self.groups and pos is not None
        token_features = [
            self._token_features(
                token, pos[i] if use_pos else None, tags[i] if use_pos and tags else None,
            )
            for i, token in enumerate(tokens)
        ]
        for group, extract in (
            ("paren", self._paren_features),
            ("defmatch", self._defmatch_features),
            ("rules", self._rule_features),
        ):
            if group in self.groups:
                for features, extra in zip(token_features, extract(tokens)):
                    features.update(extra)

        sentence_features = []
        for i, current in enumerate(token_features):
            features = {"bias": 1.0, **current}
            if i == 0:
                features["BOS"] = True
            if i == len(tokens) - 1:
                features["EOS"] = True
            if "context" in self.groups:
                for offset in (-2, -1, 1, 2):
                    neighbor_index = i + offset
                    if 0 <= neighbor_index < len(tokens):
                        neighbor = token_features[neighbor_index]
                        for key in NEIGHBOR_FEATURES:
                            if key in neighbor:
                                features[f"{offset:+d}:{key}"] = neighbor[key]
                    else:
                        features[f"{offset:+d}:pad"] = True
                if i + 1 < len(tokens):
                    features["lower|+1"] = f"{tokens[i].lower()}|{tokens[i + 1].lower()}"
            sentence_features.append(features)
        return sentence_features
