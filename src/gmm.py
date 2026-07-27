"""Full-covariance Gaussian Mixture Model in pure MLX.

Drop-in subset of sklearn.mixture.GaussianMixture (covariance_type="full"):
fit / _estimate_log_prob / predict_proba and the fitted attributes
weights_ / means_ / covariances_. Inputs and outputs are mlx arrays.
"""

import math

import mlx.core as mx


class GaussianMixture:
    """EM for a full-covariance GMM with k-means++ init and restarts.

    Args:
        n_components: number of mixture components K.
        covariance_type: only "full" is supported.
        max_iter: max EM iterations per run.
        tol: convergence threshold on the change of the mean log-likelihood.
        n_init: number of restarts; the run with the best final log-likelihood
            is kept.
        random_state: int seed (mapped to mx.random.key) or None.
        reg_covar: diagonal regularisation added to each covariance.
    """

    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "full",
        max_iter: int = 100,
        tol: float = 1e-3,
        n_init: int = 1,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
    ):
        if covariance_type != "full":
            raise NotImplementedError(
                f"only covariance_type='full' is supported, got {covariance_type!r}"
            )
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.reg_covar = reg_covar

        self.weights_: mx.array | None = None
        self.means_: mx.array | None = None
        self.covariances_: mx.array | None = None
        self.lower_bound_: float | None = None

    # ── public API ──────────────────────────────────────────────────────

    def fit(self, x: mx.array) -> GaussianMixture:
        x = mx.array(x, dtype=mx.float32)
        key = mx.random.key(0 if self.random_state is None else self.random_state)
        best_ll = -math.inf
        for _i in range(self.n_init):
            key, sub = mx.random.split(key)
            w, mu, cov, ll = self._em_run(x, sub)
            if ll > best_ll:
                best_ll = ll
                self.weights_, self.means_, self.covariances_ = w, mu, cov
        self.lower_bound_ = best_ll
        mx.eval(self.weights_, self.means_, self.covariances_)
        return self

    def estimate_log_prob(self, x: mx.array) -> mx.array:
        """log N(x; mu_k, Sigma_k) per component → (N, K)."""
        assert self.means_ is not None and self.covariances_ is not None
        x = mx.array(x, dtype=mx.float32)
        return _log_gaussian_full(
            x, self.means_, self.covariances_, self.n_components, x.shape[1]
        )

    def predict_proba(self, X: mx.array) -> mx.array:
        """Posterior responsibilities → (N, K)."""
        assert self.weights_ is not None
        log_resp = self.estimate_log_prob(X) + mx.log(self.weights_ + 1e-12)
        return mx.softmax(log_resp, axis=1)

    # ── internals ───────────────────────────────────────────────────────

    def _em_run(
        self, X: mx.array, key: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, float]:
        _N, D = X.shape
        K = self.n_components
        resp = self._kmeanspp_resp(X, key, K)
        ll_prev = -math.inf
        for _iter in range(self.max_iter):
            w, mu, cov = self._m_step(X, resp)
            log_prob = _log_gaussian_full(X, mu, cov, K, D)
            log_resp = log_prob + mx.log(w + 1e-12)
            log_norm = mx.logsumexp(log_resp, axis=1)
            ll = float(log_norm.mean().item())
            resp = mx.softmax(log_resp, axis=1)
            if abs(ll - ll_prev) < self.tol:
                break
            ll_prev = ll
        w, mu, cov = self._m_step(X, resp)
        ll = float(
            mx.logsumexp(
                _log_gaussian_full(X, mu, cov, K, D) + mx.log(w + 1e-12), axis=1
            )
            .mean()
            .item()
        )
        return w, mu, cov, ll

    def _m_step(
        self, X: mx.array, resp: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        N, D = X.shape
        Nk = resp.sum(axis=0) + 1e-12  # (K,)
        w = Nk / N
        mu = (resp.T @ X) / Nk.reshape(-1, 1)  # (K, D)
        covs = []
        for k in range(self.n_components):
            diff = X - mu[k]  # (N, D)
            cw = diff * resp[:, k : k + 1]
            cov = (cw.T @ diff) / Nk[k]
            cov = cov + self.reg_covar * mx.eye(D, dtype=mx.float32)
            covs.append(cov)
        return w, mu, mx.stack(covs, axis=0)

    def _kmeanspp_resp(self, X: mx.array, key: mx.array, K: int) -> mx.array:
        """k-means++ seeding + a few Lloyd iterations → one-hot resp."""
        N = X.shape[0]
        key, sub = mx.random.split(key)
        idx = int(mx.random.randint(0, N, (1,), key=sub).item())
        centers: list[mx.array] = [X[idx]]
        for _j in range(K - 1):
            d2 = mx.min(mx.stack([((X - c) ** 2).sum(axis=1) for c in centers]), axis=0)
            key, sub = mx.random.split(key)
            probs = d2 / mx.maximum(d2.sum(), 1e-12)
            idx = int(
                mx.random.categorical(
                    mx.log(probs + 1e-12).reshape(1, -1), key=sub
                ).item()
            )
            centers.append(X[idx])
        cent = mx.stack(centers)
        for _iter in range(10):
            d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1)
            lab = mx.argmin(d, axis=1)
            new_cent = []
            for k in range(K):
                mask = mx.where(lab == k, 1.0, 0.0).reshape(-1, 1)
                if float(mask.sum().item()) == 0.0:
                    new_cent.append(cent[k])
                else:
                    new_cent.append((X * mask).sum(axis=0) / mask.sum())
            new = mx.stack(new_cent)
            if mx.allclose(new, cent, rtol=1e-4, atol=1e-6):
                break
            cent = new
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1)
        lab = mx.argmin(d, axis=1)
        return mx.stack([mx.where(lab == k, 1.0, 0.0) for k in range(K)], axis=1)


def _log_gaussian_full(
    X: mx.array, mu: mx.array, cov: mx.array, K: int, D: int
) -> mx.array:
    """log N(x; mu_k, Sigma_k) for all components → (N, K)."""
    cpu = mx.Device(mx.cpu)  # cholesky/solve are CPU-only in MLX
    cols = []
    for k in range(K):
        diff = (X - mu[k]).T
        L = mx.linalg.cholesky(cov[k], stream=cpu)
        y = mx.linalg.solve(L, diff, stream=cpu)
        maha = (y * y).sum(axis=0)
        log_det = 2.0 * mx.log(mx.abs(mx.diag(L))).sum()
        cols.append(-0.5 * (D * math.log(2.0 * math.pi) + maha + log_det))
    result = mx.stack(cols, axis=1)
    mx.eval(result)  # materialise to keep the lazy graph bounded across EM iterations
    return result


def _unary_from_gmm(gmm: GaussianMixture, feats: mx.array) -> tuple[mx.array, float]:
    """Unary potentials from a (pre-fit) GMM: -log p(x|k) - log pi_k,
    row-shifted for numerical safety."""
    assert gmm.weights_ is not None, "GMM must be fitted before calling _unary_from_gmm"
    log_prob = gmm.estimate_log_prob(feats)
    log_resp = log_prob + mx.log(gmm.weights_ + 1e-12)
    ll = float(mx.logsumexp(log_resp, axis=1).mean().item())
    return -(log_resp - mx.max(log_resp, axis=1, keepdims=True)), ll


def gmm_unary(
    feats: mx.array, K: int, max_samples: int = 16384
) -> tuple[GaussianMixture, mx.array, float]:
    """Fit a full-covariance GMM and return (gmm, unary potentials)."""
    N = feats.shape[0]
    if max_samples < N:
        idx = mx.argsort(mx.random.uniform(shape=(N,), key=mx.random.key(0)))[
            :max_samples
        ]
        fit_x = feats[idx]
        n_init = 3
    else:
        fit_x = feats
        n_init = 10
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        max_iter=200,
        n_init=n_init,
        random_state=42,
        reg_covar=1e-3,
    )
    gmm.fit(fit_x)
    assert gmm.weights_ is not None and gmm.means_ is not None
    return gmm, *_unary_from_gmm(gmm, feats)
