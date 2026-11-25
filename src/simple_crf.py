"""Linear-chain CRF for environments without python-crfsuite.

L-BFGS minimizes the negative log-likelihood with an L2 penalty on the
feature weights. Viterbi finds the highest-scoring label sequence.
"""

import time
from collections import Counter

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp


class LinearChainCRF:
    """Train and decode a first-order CRF with sparse token features."""

    def __init__(self, c2=0.1, max_iterations=150, min_freq=1, verbose=True):
        self.c2 = c2
        self.max_iterations = max_iterations
        self.min_freq = min_freq
        self.verbose = verbose

    @staticmethod
    def _items(fdict):
        for name, value in fdict.items():
            if isinstance(value, bool):
                if value:
                    yield name, 1.0
            elif isinstance(value, (int, float)):
                if value:
                    yield name, float(value)
            else:
                yield f"{name}={value}", 1.0

    def _build_vocab(self, X):
        counts = Counter()
        for sequence in X:
            for features in sequence:
                for name, _ in self._items(features):
                    counts[name] += 1
        names = sorted(name for name, count in counts.items() if count >= self.min_freq)
        self.feature_index_ = {name: index for index, name in enumerate(names)}

    def _matrix(self, X):
        rows, columns, values, lengths = [], [], [], []
        token_index = 0
        for sequence in X:
            lengths.append(len(sequence))
            for features in sequence:
                for name, value in self._items(features):
                    column = self.feature_index_.get(name)
                    if column is not None:
                        rows.append(token_index)
                        columns.append(column)
                        values.append(value)
                token_index += 1
        matrix = sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(token_index, len(self.feature_index_)),
            dtype=np.float64,
        )
        return matrix, np.array(lengths)

    def _unpack(self, theta):
        F, L = self.n_features_, self.n_labels_
        W = theta[: F * L].reshape(F, L)
        rest = theta[F * L:]
        trans = rest[: L * L].reshape(L, L)
        start = rest[L * L: L * L + L]
        end = rest[L * L + L:]
        return W, trans, start, end

    @staticmethod
    def _pad(E, lengths):
        S, T, L = len(lengths), int(lengths.max()), E.shape[1]
        P = np.zeros((S, T, L))
        mask = np.arange(T)[None, :] < lengths[:, None]
        P[mask] = E
        return P, mask

    def _objective(self, theta):
        W, trans, start, end = self._unpack(theta)
        X, lengths, Y, mask = self._X, self._lengths, self._Y, self._mask
        E = self._pad(X @ W, lengths)[0]  # S x T x L
        S, T, L = E.shape
        last = lengths - 1
        rows = np.arange(S)

        # forward
        alpha = np.empty((S, T, L))
        alpha[:, 0] = start + E[:, 0]
        for t in range(1, T):
            a = logsumexp(alpha[:, t - 1, :, None] + trans[None], axis=1) + E[:, t]
            alpha[:, t] = np.where(mask[:, t, None], a, alpha[:, t - 1])
        logZ = logsumexp(alpha[rows, last] + end, axis=1)

        # backward
        beta = np.zeros((S, T, L))
        beta[rows, last] = end
        for t in range(T - 2, -1, -1):
            b = logsumexp(trans[None] + (E[:, t + 1] + beta[:, t + 1])[:, None, :], axis=2)
            active = (t < last)[:, None]
            beta[:, t] = np.where(active, b, beta[:, t])

        # unary marginals
        marg = np.exp(alpha + beta - logZ[:, None, None]) * mask[..., None]

        # pairwise marginals summed over positions
        pair = np.zeros((L, L))
        for t in range(1, T):
            m = mask[:, t]
            if not m.any():
                break
            lp = (alpha[m, t - 1, :, None] + trans[None] + E[m, t, None, :]
                  + beta[m, t, None, :] - logZ[m, None, None])
            pair += np.exp(lp).sum(axis=0)

        # gold path score
        Yp = Y  # S x T, padded with 0
        gold = (E[rows[:, None], np.arange(T)[None], Yp] * mask).sum()
        prev, cur = Yp[:, :-1], Yp[:, 1:]
        gold += (trans[prev, cur] * mask[:, 1:]).sum()
        gold += start[Yp[:, 0]].sum() + end[Yp[rows, last]].sum()

        nll = logZ.sum() - gold + self.c2 * (W ** 2).sum()

        # gradients
        flat_mask = mask.ravel()
        P = marg.reshape(S * T, L)[flat_mask]
        G = P - self._Yonehot
        dW = self._X.T @ G + 2 * self.c2 * W
        obs_trans = np.zeros((L, L))
        np.add.at(obs_trans, (prev[mask[:, 1:]], cur[mask[:, 1:]]), 1)
        dtrans = pair - obs_trans
        dstart = marg[:, 0].sum(axis=0) - np.bincount(Yp[:, 0], minlength=L)
        dend = marg[rows, last].sum(axis=0) - np.bincount(Yp[rows, last], minlength=L)
        grad = np.concatenate([np.asarray(dW).ravel(), dtrans.ravel(), dstart, dend])
        return nll, grad

    def fit(self, X, y):
        """Fit the feature and transition weights to labeled sequences."""
        started = time.time()
        self.classes_ = sorted({label for sequence in y for label in sequence})
        self.label_index_ = {label: index for index, label in enumerate(self.classes_)}
        self._build_vocab(X)
        self._X, self._lengths = self._matrix(X)
        self.n_features_, self.n_labels_ = self._X.shape[1], len(self.classes_)
        S, T = len(y), int(self._lengths.max())
        self._Y = np.zeros((S, T), dtype=int)
        for index, sequence in enumerate(y):
            self._Y[index, : len(sequence)] = [self.label_index_[label] for label in sequence]
        self._mask = np.arange(T)[None, :] < self._lengths[:, None]
        self._Yonehot = np.eye(self.n_labels_)[self._Y[self._mask]]

        L = self.n_labels_
        theta0 = np.zeros(self.n_features_ * L + L * L + 2 * L)
        if self.verbose:
            print(f"  fallback CRF: {self.n_features_} features, {S} sequences, {L} labels")
        result = minimize(
            self._objective, theta0, jac=True, method="L-BFGS-B",
            options={"maxiter": self.max_iterations},
        )
        self.W_, self.trans_, self.start_, self.end_ = self._unpack(result.x)
        if self.verbose:
            print(f"  done in {time.time() - started:.0f}s, {result.nit} iterations, loss {result.fun:.1f}")
        del self._X, self._Y, self._mask, self._Yonehot, self._lengths
        return self

    def predict(self, X):
        """Return the highest-scoring label sequence for each input."""
        feature_matrix, lengths = self._matrix(X)
        emissions = feature_matrix @ self.W_
        predictions = []
        start = 0
        for length in lengths:
            end = start + length
            path = self._viterbi(emissions[start:end]) if length else []
            predictions.append([self.classes_[label] for label in path])
            start = end
        return predictions

    def _viterbi(self, emissions):
        token_count, label_count = emissions.shape
        scores = self.start_ + emissions[0]
        previous_label = np.zeros((token_count, label_count), dtype=int)
        for index in range(1, token_count):
            transitions = scores[:, None] + self.trans_
            previous_label[index] = transitions.argmax(axis=0)
            scores = transitions.max(axis=0) + emissions[index]

        path = [int((scores + self.end_).argmax())]
        for index in range(token_count - 1, 0, -1):
            path.append(int(previous_label[index, path[-1]]))
        return path[::-1]
