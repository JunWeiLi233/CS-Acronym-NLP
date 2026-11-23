"""Part-of-speech tags from spaCy for the CRF "pos" feature group.

spaCy is optional. If it (or the en_core_web_sm model) is not installed,
get_pos_tags returns None and the CRF trains without POS features.

    pip install spacy
    python -m spacy download en_core_web_sm

Tags are computed on our own tokens (so they line up with the labels) and
cached in data/processed/pos_cache.json, keyed by sentence text.
"""

import hashlib
import json

from utils import DATA_DIR

CACHE_PATH = DATA_DIR / "pos_cache.json"
_NLP = None


def _load_spacy(model="en_core_web_sm"):
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load(model, disable=["ner", "lemmatizer", "parser"])
        except (ImportError, OSError) as err:
            print(f"[info] spaCy POS tags unavailable ({err.__class__.__name__}); "
                  "training without the 'pos' feature group")
            _NLP = False
    return _NLP


def _key(tokens):
    return hashlib.md5("\u0001".join(tokens).encode("utf-8")).hexdigest()


def spacy_available():
    return bool(_load_spacy())


def get_pos_tags(token_lists, use_cache=True):
    """Return a list of (pos, tag) list pairs aligned with token_lists, or None."""
    nlp = _load_spacy()
    if not nlp:
        return None
    from spacy.tokens import Doc

    cache = {}
    if use_cache and CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [toks for toks in token_lists if _key(toks) not in cache]
    if todo:
        docs = (Doc(nlp.vocab, words=list(toks)) for toks in todo)
        for toks, doc in zip(todo, nlp.pipe(docs, batch_size=256)):
            cache[_key(toks)] = [[t.pos_ for t in doc], [t.tag_ for t in doc]]
        if use_cache:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)
    return [tuple(cache[_key(toks)]) for toks in token_lists]
