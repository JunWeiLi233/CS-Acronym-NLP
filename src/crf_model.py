"""CRF model wrapper used by train_crf.py and predict.py.

Backends:
  crfsuite  python-crfsuite (installed with `pip install python-crfsuite`
            or as a dependency of sklearn-crfsuite). L-BFGS with L1 + L2.
  simple    the numpy/scipy CRF in simple_crf.py (L2 only), used when
            python-crfsuite is not installed.
"""

import os
import pickle
import tempfile

from features import PROPOSAL_GROUPS, FeatureExtractor
from rule_based import propagate
from simple_crf import LinearChainCRF


def crfsuite_available():
    try:
        import pycrfsuite  # noqa: F401
        return True
    except ImportError:
        return False


class CrfsuiteCRF:
    """Thin fit/predict wrapper around python-crfsuite that pickles cleanly."""

    def __init__(self, c1=0.1, c2=0.05, max_iterations=150, verbose=False):
        self.params = {"c1": c1, "c2": c2, "max_iterations": max_iterations,
                       "feature.possible_transitions": True}
        self.verbose = verbose
        self._model_bytes = None
        self._tagger = None

    def fit(self, X, y):
        import pycrfsuite
        trainer = pycrfsuite.Trainer(verbose=self.verbose)
        for xseq, yseq in zip(X, y):
            trainer.append(xseq, yseq)
        trainer.set_params(self.params)
        fd, path = tempfile.mkstemp(suffix=".crfsuite")
        os.close(fd)
        try:
            trainer.train(path)
            with open(path, "rb") as f:
                self._model_bytes = f.read()
        finally:
            os.remove(path)
        self._tagger = None
        return self

    def _get_tagger(self):
        if self._tagger is None:
            import pycrfsuite
            fd, path = tempfile.mkstemp(suffix=".crfsuite")
            with os.fdopen(fd, "wb") as f:
                f.write(self._model_bytes)
            self._tagger = pycrfsuite.Tagger()
            self._tagger.open(path)
        return self._tagger

    def predict(self, X):
        tagger = self._get_tagger()
        return [tagger.tag(xseq) if xseq else [] for xseq in X]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state


def make_crf(backend="auto", c1=0.1, c2=0.05, max_iterations=150, verbose=False):
    if backend == "auto":
        backend = "crfsuite" if crfsuite_available() else "simple"
    if backend == "crfsuite":
        return CrfsuiteCRF(c1=c1, c2=c2, max_iterations=max_iterations, verbose=verbose), backend
    if backend == "simple":
        return LinearChainCRF(c2=c2, max_iterations=max_iterations, verbose=True), backend
    raise ValueError(f"unknown backend {backend}")


class CRFModel:
    """Feature extraction + CRF + optional post-processing, saved as one pickle."""

    def __init__(self, groups=PROPOSAL_GROUPS, backend="auto", c1=0.1, c2=0.05,
                 max_iterations=150, propagation=None):
        self.groups = tuple(groups)
        self.extractor = FeatureExtractor(self.groups)
        self.crf, self.backend = make_crf(backend, c1, c2, max_iterations)
        self.hparams = {"c1": c1, "c2": c2, "max_iterations": max_iterations}
        self.propagation = propagation
        self.use_pos = False

    def featurize(self, records, fit=False):
        sentence_tags = None
        if "pos" in self.groups and (fit or self.use_pos):
            from pos import get_pos_tags

            sentence_tags = get_pos_tags([record["tokens"] for record in records])
            if fit:
                self.use_pos = sentence_tags is not None
            elif sentence_tags is None:
                print("[warn] model was trained with POS tags but spaCy is unavailable now")
        if sentence_tags is None:
            return [self.extractor.sent2features(record["tokens"]) for record in records]
        return [
            self.extractor.sent2features(record["tokens"], pos_tags, fine_tags)
            for record, (pos_tags, fine_tags) in zip(records, sentence_tags)
        ]

    def fit(self, records):
        features = self.featurize(records, fit=True)
        labels = [record["labels"] for record in records]
        self.crf.fit(features, labels)
        return self

    def predict(self, records, propagation="default"):
        """Predict BIO labels, using the model's propagation setting by default."""
        if propagation == "default":
            propagation = self.propagation
        predictions = [list(labels) for labels in self.crf.predict(self.featurize(records))]
        if propagation:
            predictions = propagate(records, predictions, scope=propagation)
        return predictions

    def save(self, path):
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as model_file:
            pickle.dump(self, model_file)

    @staticmethod
    def load(path):
        with open(path, "rb") as model_file:
            return pickle.load(model_file)
