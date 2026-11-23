"""A small linear-chain CRF in numpy/scipy.

This is a fallback for when sklearn-crfsuite cannot be installed. It has
the same fit / predict interface (lists of feature-dict sequences in, lists
of label sequences out), trains with L-BFGS on the L2-regularised negative
log-likelihood, and decodes with Viterbi. sklearn-crfsuite is faster and
supports L1 (c1), so train_crf.py prefers it when available.

Objective:  sum_i -log p(y_i | x_i)  +  c2 * ||w||^2
"""

import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp


class LinearChainCRF:
    def __init__(self, c2=0.1, max_iterations=150, min_freq=1, verbose=True):
        self.c2 = c2
        self.max_iterations = max_iterations
        self.min_freq = min_freq
        self.verbose = verbose

    # ---- featurisation -------------------------------------------------------

    @staticmethod
    def _items(fdict):
        for k, v in fdict.items():
            if isinstance(v, bool):
                if v:
                    yield k, 1.0
            elif isinstance(v, (int, float)):
                if v:
                    yield k, float(v)
            else:
                yield f"{k}={v}", 1.0

    def _build_vocab(self, X):
        counts = {}
        for seq in X:
            for fd in seq:
                for name, _ in self._items(fd):
                    counts[name] = counts.get(name, 0) + 1
        names = sorted(n for n, c in counts.items() if c >= self.min_freq)
        self.feature_index_ = {n: i for i, n in enumerate(names)}

    def _matrix(self, X):
        rows, cols, vals, lengths = [], [], [], []
        r = 0
        idx = self.feature_index_
        for seq in X:
            lengths.append(len(seq))
            for fd in seq:
                for name, v in self._items(fd):
                    j = idx.get(name)
                    if j is not None:
                        rows.append(r)
                        cols.append(j)
                        vals.append(v)
                r += 1
        M = sparse.csr_matrix((vals, (rows, cols)), shape=(r, len(idx)), dtype=np.float64)
        return M, np.array(lengths)

    # ---- parameters ----------------------------------------------------------

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

    # ---- objective -----------------------------------------------------------

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

    # ---- public API ----------------------------------------------------------

    def fit(self, X, y):
        t0 = time.time()
        self.classes_ = sorted({l for seq in y for l in seq})
        self.label_index_ = {l: i for i, l in enumerate(self.classes_)}
        self._build_vocab(X)
        self._X, self._lengths = self._matrix(X)
        self.n_features_, self.n_labels_ = self._X.shape[1], len(self.classes_)
        S, T = len(y), int(self._lengths.max())
        self._Y = np.zeros((S, T), dtype=int)
        for i, seq in enumerate(y):
            self._Y[i, : len(seq)] = [self.label_index_[l] for l in seq]
        self._mask = np.arange(T)[None, :] < self._lengths[:, None]
        self._Yonehot = np.eye(self.n_labels_)[self._Y[self._mask]]

        L = self.n_labels_
        theta0 = np.zeros(self.n_features_ * L + L * L + 2 * L)
        if self.verbose:
            print(f"  fallback CRF: {self.n_features_} features, {S} sequences, {L} labels")
        res = minimize(self._objective, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iterations})
        self.W_, self.trans_, self.start_, self.end_ = self._unpack(res.x)
        if self.verbose:
            print(f"  done in {time.time() - t0:.0f}s, {res.nit} iterations, loss {res.fun:.1f}")
        del self._X, self._Y, self._mask, self._Yonehot, self._lengths
        return self

    def predict(self, X):
        M, lengths = self._matrix(X)
        E_all = M @ self.W_
        out, pos = [], 0
        for n in lengths:
            E = E_all[pos: pos + n]
            pos += n
            out.append([self.classes_[k] for k in self._viterbi(E)] if n else [])
        return out

    def _viterbi(self, E):
        n, L = E.shape
        score = self.start_ + E[0]
        back = np.zeros((n, L), dtype=int)
        for t in range(1, n):
            cand = score[:, None] + self.trans_
            back[t] = cand.argmax(axis=0)
            score = cand.max(axis=0) + E[t]
        score = score + self.end_
        path = [int(score.argmax())]
        for t in range(n - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]
