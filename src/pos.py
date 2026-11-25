"""Tag the project's tokens using spaCy so POS tags align with BIO labels.

The CRF can omit POS features when spaCy or en_core_web_sm is unavailable.
The cache in data/processed/pos_cache.json is keyed by token sequence.
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


def get_pos_tags(token_lists, use_cache=True):
    """Return coarse and fine POS tags for each sentence, or None without spaCy."""
    nlp = _load_spacy()
    if not nlp:
        return None
    from spacy.tokens import Doc

    cache = {}
    if use_cache and CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as cache_file:
            cache = json.load(cache_file)
    uncached = [tokens for tokens in token_lists if _key(tokens) not in cache]
    if uncached:
        docs = (Doc(nlp.vocab, words=list(tokens)) for tokens in uncached)
        for tokens, doc in zip(uncached, nlp.pipe(docs, batch_size=256)):
            cache[_key(tokens)] = [[token.pos_ for token in doc], [token.tag_ for token in doc]]
        if use_cache:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CACHE_PATH.open("w", encoding="utf-8") as cache_file:
                json.dump(cache, cache_file)
    return [tuple(cache[_key(tokens)]) for tokens in token_lists]
