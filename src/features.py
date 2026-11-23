"""CRF features, adapted from Vanopstal et al. (2010) as listed in the
proposal. Features are organised in groups so that each group can be
removed for the ablation study (train_crf.py --ablation).

  form      lowercased token, prefixes/suffixes up to 4 chars, word shape
            ("XXX", "Xxxxx", "x-XX") and a collapsed shape ("X", "Xx", "x-X")
  ortho     all caps, internal capitals, initial capital, digits, hyphen,
            vowel, Roman numeral, punctuation, length, plural abbreviation
  context   the form/ortho/dict/pos features of the two tokens on each side
  paren     inside parentheses, next to "(" / ")", distance to the next "("
  pos       spaCy part-of-speech tags (only if spaCy is installed)
  dict      token is in an English word list (Kreuzthaler et al., 2016)
  defmatch  definition match: the token's first letter starts a word within
            the preceding 3 x N words (N = token length); for long-form
            tokens, whether a following parenthesised short form starts
            with / contains the token's first letter
  rules     (extension, not in the proposal) output of the Schwartz & Hearst
            rules as features: token is a short form / inside a long form
            found by the rule-based system

Feature values are str / bool / float so the same dicts work with
sklearn-crfsuite and with the fallback CRF in simple_crf.py.
"""

import re

from rule_based import find_definitions, find_glossary_definitions
from utils import ROMAN_FULL, in_lexicon, is_word, load_lexicon, upper_count

ALL_GROUPS = ("form", "ortho", "context", "paren", "pos", "dict", "defmatch", "rules")
PROPOSAL_GROUPS = ("form", "ortho", "context", "paren", "pos", "dict", "defmatch")

VOWELS = set("aeiouAEIOU")


def word_shape(token):
    out = []
    for c in token:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)


def short_shape(token):
    return re.sub(r"(.)\1+", r"\1", word_shape(token))


def _bucket(n, cap):
    return str(n) if n <= cap else f">{cap}"


class FeatureExtractor:
    def __init__(self, groups=PROPOSAL_GROUPS, lexicon=None):
        self.groups = set(groups)
        unknown = self.groups - set(ALL_GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups: {unknown}")
        self.lexicon = load_lexicon() if lexicon is None else lexicon

    # ---- single-token features (also used for neighbours) -----------------

    def _token_features(self, tok, pos_tag=None, tag=None):
        f = {}
        g = self.groups
        if "form" in g:
            low = tok.lower()
            f["lower"] = low
            f["shape"] = word_shape(tok)[:12]
            f["sshape"] = short_shape(tok)
            for k in range(1, 5):
                if len(tok) >= k:
                    f[f"pre{k}"] = tok[:k]
                    f[f"suf{k}"] = tok[-k:]
        if "ortho" in g:
            letters = [c for c in tok if c.isalpha()]
            if letters and all(c.isupper() for c in letters):
                f["all_caps"] = True
            if tok[:1].isupper():
                f["init_cap"] = True
            if any(c.isupper() for c in tok[1:]):
                f["internal_cap"] = True
            if any(c.isdigit() for c in tok):
                f["has_digit"] = True
            if "-" in tok:
                f["has_hyphen"] = True
            if any(c in VOWELS for c in tok):
                f["has_vowel"] = True
            if ROMAN_FULL.match(tok):
                f["roman"] = True
            if not is_word(tok):
                f["punct"] = True
            f["len"] = _bucket(len(tok), 10)
            f["n_upper"] = _bucket(upper_count(tok), 4)
            if len(tok) > 2 and tok.endswith("s") and upper_count(tok[:-1]) >= 2:
                f["plural_abbr"] = True
        if "dict" in g and is_word(tok):
            if in_lexicon(tok, self.lexicon):
                f["in_dict"] = True
            else:
                f["oov"] = True
        if "pos" in g and pos_tag is not None:
            f["pos"] = pos_tag
            if tag is not None:
                f["tag"] = tag
        return f

    # ---- sentence-level features --------------------------------------------

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
        feats = [dict() for _ in range(n)]
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
        feats = [dict() for _ in tokens]
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
        """List of feature dicts, one per token."""
        n = len(tokens)
        use_pos = "pos" in self.groups and pos is not None
        base = [self._token_features(t, pos[i] if use_pos else None,
                                     tags[i] if use_pos and tags else None)
                for i, t in enumerate(tokens)]
        extra = [dict() for _ in range(n)]
        if "paren" in self.groups:
            for e, f in zip(extra, self._paren_features(tokens)):
                e.update(f)
        if "defmatch" in self.groups:
            for e, f in zip(extra, self._defmatch_features(tokens)):
                e.update(f)
        if "rules" in self.groups:
            for e, f in zip(extra, self._rule_features(tokens)):
                e.update(f)

        neighbour_keys = ("lower", "sshape", "all_caps", "init_cap", "internal_cap",
                          "has_digit", "punct", "in_dict", "oov", "pos", "rule_lf", "rule_sf")
        out = []
        for i in range(n):
            f = {"bias": 1.0}
            f.update(base[i])
            f.update(extra[i])
            if i == 0:
                f["BOS"] = True
            if i == n - 1:
                f["EOS"] = True
            if "context" in self.groups:
                for off in (-2, -1, 1, 2):
                    j = i + off
                    if 0 <= j < n:
                        src = {**base[j], **extra[j]}
                        for k in neighbour_keys:
                            if k in src:
                                f[f"{off:+d}:{k}"] = src[k]
                    else:
                        f[f"{off:+d}:pad"] = True
                if i + 1 < n:  # bigram cues
                    f["lower|+1"] = f"{tokens[i].lower()}|{tokens[i + 1].lower()}"
            out.append(f)
        return out
