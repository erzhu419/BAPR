"""Joint regime belief tracker b(h, z) — BAPR Minimum redesign (GPT-5.5 v3).

Replaces the run-length-only BOCD with a joint belief over
  (h = run-length, z = discrete regime cluster).

Key differences from common/belief_tracker.py:
  - belief is shape (H, K), not (H,)
  - per-regime observation model p(y | z) ~ N(mu_z, diag(sigma_z^2))
    where y = (reward_residual, q_std_spike, reg_norm_div) ∈ R^3
    (the same 3 channels SurpriseComputer already extracts).
  - cold start: collect first `warmup_samples` y vectors into a buffer,
    then run k-means(K) once to seed (mu_z, sigma_z). Belief stays uniform
    during the cold-start period.
  - online: BOCD-style joint update of b(h,z); EMA refresh of (mu_z, sigma_z)
    weighted by current marginal mu(z).

Provides:
  - `belief` (H, K)
  - `rho` marginal over h, shape (H,) — drop-in replacement for old ρ(h)
  - `mu_z` marginal over z, shape (K,) — new "which regime am I in" signal
  - `effective_window`, `entropy` — same semantics as before
"""
from typing import Dict, Optional
import numpy as np


class RegimeBeliefTracker:
    """Joint b(h, z) belief — see module docstring."""

    def __init__(self,
                 max_run_length: int = 20,
                 num_regimes: int = 4,
                 hazard_rate: float = 0.0165,
                 warmup_samples: int = 10_000,
                 ewma_alpha: float = 0.05,
                 obs_dim: int = 3,
                 seed: int = 0):
        self.H = int(max_run_length)
        self.K = int(num_regimes)
        self.D = int(obs_dim)
        self.hazard = float(hazard_rate)
        self.warmup_samples_target = int(warmup_samples)
        self.ewma_alpha = float(ewma_alpha)
        self.rng = np.random.default_rng(seed)

        # Joint belief, uniform init
        self.belief = np.ones((self.H, self.K), dtype=np.float32) / (self.H * self.K)

        # Per-regime observation model (filled by k-means after warmup)
        self.mu: Optional[np.ndarray] = None         # (K, D)
        self.var: Optional[np.ndarray] = None        # (K, D)
        self.is_seeded: bool = False

        # Warmup buffer
        self._warmup_buf: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset belief to uniform; preserves learned (mu_z, sigma_z)."""
        self.belief = np.ones((self.H, self.K), dtype=np.float32) / (self.H * self.K)

    def observe(self, y: np.ndarray) -> bool:
        """Push observation y (shape (D,)) into the tracker.

        During warmup: appends to buffer; once buffer reaches target size,
        triggers k-means seeding. Returns True when seeding just happened.

        After warmup: BOCD-style joint update of b(h,z).
        """
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        assert y.shape == (self.D,), f"expected y of shape ({self.D},), got {y.shape}"

        if not self.is_seeded:
            self._warmup_buf.append(y)
            if len(self._warmup_buf) >= self.warmup_samples_target:
                self._seed_kmeans()
                return True
            return False

        self._update(y)
        return False

    @property
    def rho(self) -> np.ndarray:
        """Marginal over run-length: shape (H,). Drop-in for old BOCD ρ(h)."""
        return self.belief.sum(axis=1)

    @property
    def mu_marginal(self) -> np.ndarray:
        """Marginal over regime z: shape (K,)."""
        return self.belief.sum(axis=0)

    @property
    def entropy(self) -> float:
        b = self.belief.flatten()
        b = b[b > 1e-10]
        return float(-np.sum(b * np.log(b)))

    @property
    def effective_window(self) -> float:
        return float(np.sum(self.rho * np.arange(self.H)))

    @property
    def num_warmup_collected(self) -> int:
        return len(self._warmup_buf)

    def serialize(self) -> Dict:
        return {
            'belief': self.belief.copy(),
            'mu': None if self.mu is None else self.mu.copy(),
            'var': None if self.var is None else self.var.copy(),
            'is_seeded': self.is_seeded,
            'warmup_buf': list(self._warmup_buf),
        }

    def load(self, state: Dict):
        self.belief = state['belief'].astype(np.float32)
        self.mu = None if state.get('mu') is None else state['mu'].astype(np.float32)
        self.var = None if state.get('var') is None else state['var'].astype(np.float32)
        self.is_seeded = bool(state['is_seeded'])
        self._warmup_buf = list(state.get('warmup_buf', []))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_kmeans(self):
        """Run mini k-means(K) on warmup buffer to seed (mu, var)."""
        Y = np.stack(self._warmup_buf).astype(np.float32)  # (N, D)
        N = Y.shape[0]

        # k-means++ init
        idx = self.rng.integers(0, N)
        centers = [Y[idx]]
        for _ in range(self.K - 1):
            d2 = np.min(
                np.stack([np.sum((Y - c) ** 2, axis=1) for c in centers], axis=0),
                axis=0)
            probs = d2 / (d2.sum() + 1e-12)
            idx = self.rng.choice(N, p=probs)
            centers.append(Y[idx])
        centers = np.stack(centers)  # (K, D)

        # Lloyd iterations
        for _ in range(20):
            d2 = np.sum((Y[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, K)
            labels = np.argmin(d2, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(self.K):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centers[k] = Y[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]  # keep stale center
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < 1e-4:
                break

        # Compute per-cluster variance
        self.mu = centers.astype(np.float32)
        var = np.zeros((self.K, self.D), dtype=np.float32)
        for k in range(self.K):
            mask = (labels == k)
            if mask.sum() > 1:
                var[k] = Y[mask].var(axis=0) + 1e-3
            else:
                var[k] = np.ones(self.D, dtype=np.float32)
        self.var = var

        # Degeneracy check: warn if clusters too close relative to spread
        # (symptom: warmup data was all one regime → spurious sub-clusters).
        # Use min pairwise center distance vs max within-cluster std.
        pair_d = np.linalg.norm(self.mu[:, None, :] - self.mu[None, :, :], axis=-1)
        np.fill_diagonal(pair_d, np.inf)
        min_inter = float(pair_d.min())
        max_intra_std = float(np.sqrt(self.var.max()))
        self.kmeans_quality = min_inter / max(max_intra_std, 1e-6)
        if self.kmeans_quality < 1.5:
            print(f"[RegimeBeliefTracker] WARNING: k-means looks degenerate "
                  f"(min_inter_dist={min_inter:.3f}, max_intra_std={max_intra_std:.3f}, "
                  f"quality_ratio={self.kmeans_quality:.2f}). "
                  f"Likely cause: warmup data lacked regime diversity. "
                  f"Consider longer warmup or env with more frequent switches.")

        self.is_seeded = True
        self._warmup_buf = []  # release memory

    def _likelihood(self, y: np.ndarray) -> np.ndarray:
        """Per-regime Gaussian likelihood p(y|z), shape (K,).

        Renormalized to sum to 1 (so likelihood acts as a soft assignment
        rather than absolute density). This avoids the case where y is far
        from all clusters → all log_p very negative → after exp() all near
        zero → joint update normalization explodes (1e-10 underflow) and
        belief gets reset to uniform (memory loss).
        """
        diff = y[None, :] - self.mu                           # (K, D)
        log_p = -0.5 * np.sum(diff * diff / self.var, axis=1) \
                - 0.5 * np.sum(np.log(2 * np.pi * self.var), axis=1)
        log_p -= log_p.max()                                  # numerical stab
        ell = np.exp(log_p).astype(np.float32)
        # Renormalize so even if y is an outlier, ell is a valid soft
        # assignment over K. Keeps belief update well-conditioned.
        ell /= (ell.sum() + 1e-12)
        return ell                                            # (K,)

    def _update(self, y: np.ndarray):
        """One BOCD-style joint update of b(h, z)."""
        ell = self._likelihood(y)                             # (K,)
        scaled = self.belief * ell[None, :]                   # (H, K)

        # Growth: shift h -> h+1, weighted by (1-hazard)
        new_belief = np.zeros_like(self.belief)
        new_belief[1:, :] = (1.0 - self.hazard) * scaled[:-1, :]

        # Changepoint: aggregate evidence at h=0, distribute by prior pi0(z)
        # Prior on next regime: uniform over K (ignorant)
        cp_evidence = self.hazard * float(scaled.sum())
        pi0 = np.ones(self.K, dtype=np.float32) / self.K
        new_belief[0, :] = cp_evidence * pi0

        # Normalize
        total = float(new_belief.sum())
        if total > 1e-10:
            new_belief /= total
        else:
            new_belief = np.ones_like(self.belief) / (self.H * self.K)
        self.belief = new_belief

        # EWMA refresh of (mu_z, var_z), weighted by marginal mu(z).
        # Welford-style: variance MUST use OLD mean, not the just-updated one
        # (same bug SurpriseComputer fixed in common/belief_tracker.py:87).
        mu_marg = self.belief.sum(axis=0)                     # (K,)
        a = self.ewma_alpha
        for k in range(self.K):
            w = float(mu_marg[k])
            if w > 1e-3:
                rate = a * w
                old_mu = self.mu[k].copy()
                self.mu[k] = (1.0 - rate) * old_mu + rate * y
                # Variance update uses (y - old_mu) * (y - new_mu) — the
                # symmetric Welford formulation; positive-definite as long as
                # rate ∈ (0, 1).
                self.var[k] = (1.0 - rate) * self.var[k] \
                              + rate * (y - old_mu) * (y - self.mu[k]) \
                              + 1e-4
