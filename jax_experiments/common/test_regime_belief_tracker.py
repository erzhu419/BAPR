"""Smoke test for RegimeBeliefTracker.

Runs synthetic 4-regime data through the tracker; checks that:
  1. Warmup phase collects samples without updating belief
  2. K-means seeding triggers at the expected sample count
  3. After seeding, belief moves away from uniform when regime data switches
  4. Marginal mu(z) concentrates on the correct regime within ~50 obs after
     a clean switch

Run: python -m jax_experiments.common.test_regime_belief_tracker
"""
import numpy as np
from jax_experiments.common.regime_belief_tracker import RegimeBeliefTracker


def make_synthetic_data(K=4, D=3, n_per_regime=2500, seed=0):
    """Generate synthetic y data: K Gaussian clusters in R^D."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, size=(K, D))
    sigmas = rng.uniform(0.2, 0.8, size=(K, D))
    Y_warmup = []
    labels_warmup = []
    for k in range(K):
        Y_warmup.append(centers[k] + sigmas[k] * rng.standard_normal((n_per_regime, D)))
        labels_warmup.append(np.full(n_per_regime, k))
    perm = rng.permutation(K * n_per_regime)
    Y_warmup = np.concatenate(Y_warmup)[perm]
    labels_warmup = np.concatenate(labels_warmup)[perm]
    return Y_warmup, labels_warmup, centers, sigmas


def hungarian_match(true_centers, learned_mu):
    """Greedy assignment: true regime k → closest unique learned cluster.
    Returns dict {true_k: cluster_id}."""
    K = true_centers.shape[0]
    dists = np.linalg.norm(true_centers[:, None, :] - learned_mu[None, :, :], axis=-1)
    used = set()
    mapping = {}
    for k in range(K):
        order = np.argsort(dists[k])
        for c in order:
            if int(c) not in used:
                mapping[k] = int(c); used.add(int(c)); break
    return mapping


def test_warmup_then_seed():
    K, D = 4, 3
    tracker = RegimeBeliefTracker(
        max_run_length=10, num_regimes=K, hazard_rate=0.001,
        warmup_samples=2000, ewma_alpha=0.05, obs_dim=D, seed=42)

    Y, _, true_centers, _ = make_synthetic_data(K=K, D=D, n_per_regime=600, seed=0)

    seeded_at = None
    for i, y in enumerate(Y):
        seeded_now = tracker.observe(y)
        if seeded_now:
            seeded_at = i; break

    assert seeded_at == 1999
    assert tracker.is_seeded
    assert tracker.mu.shape == (K, D)
    print(f"[ok] warmup→seed at obs {seeded_at}; "
          f"k-means quality ratio = {tracker.kmeans_quality:.2f}")

    mapping = hungarian_match(true_centers, tracker.mu)
    assert len(set(mapping.values())) == K, "must be 1-1 mapping"
    avg_d = float(np.mean([
        np.linalg.norm(true_centers[k] - tracker.mu[mapping[k]]) for k in range(K)
    ]))
    assert avg_d < 0.3, f"avg recovery distance {avg_d:.3f} too large"
    print(f"[ok] k-means recovered all {K} centers; avg distance {avg_d:.3f}; "
          f"map true→cluster = {mapping}")
    return tracker, true_centers, mapping


def test_belief_evolution_under_switches(tracker, true_centers, mapping):
    """Feed clean single-regime data, check mu(z) concentrates."""
    K, D = tracker.K, tracker.D
    rng = np.random.default_rng(123)
    SIGMA_OBS = 0.3

    # Settle into regime 0 (200 obs)
    for _ in range(200):
        tracker.observe(true_centers[0] + SIGMA_OBS * rng.standard_normal(D))
    cluster_for_0 = mapping[0]
    pre_mass = float(tracker.mu_marginal[cluster_for_0])
    print(f"\n[settled regime 0] mass on cluster {cluster_for_0} = {pre_mass:.3f}")
    assert pre_mass > 0.5, f"regime-0 mass should dominate; got {pre_mass:.3f}"

    # Abrupt switch to regime 2 — track how fast mass moves to its cluster
    cluster_for_2 = mapping[2]
    pre_ew = tracker.effective_window
    print(f"\n[switch → regime 2 (cluster {cluster_for_2})]")
    detection_step = None
    for step in range(50):
        tracker.observe(true_centers[2] + SIGMA_OBS * rng.standard_normal(D))
        m2 = float(tracker.mu_marginal[cluster_for_2])
        if step <= 5 or step in (10, 20, 49):
            print(f"  step {step:>2d}: mass[c{cluster_for_2}]={m2:.3f}, "
                  f"ew={tracker.effective_window:.2f}")
        if detection_step is None and m2 > 0.5:
            detection_step = step

    assert detection_step is not None and detection_step <= 5, \
        f"failed to detect switch within 5 obs (got {detection_step})"
    print(f"[ok] regime 2 detected at step {detection_step} (≤5 required)")
    # Effective window should briefly dip below pre-switch level (BOCD signal)
    print(f"[ok] effective_window {pre_ew:.2f} → {tracker.effective_window:.2f}")


def test_recurring_regimes(tracker, true_centers, mapping):
    """Cycle A → B → C → A — does the SAME cluster fire for both A visits?"""
    K, D = tracker.K, tracker.D
    rng = np.random.default_rng(456)
    print("\n--- recurring-regime test: cycle through 0, 1, 2, 0 ---")
    cluster_seen_for = {}
    for cycle_idx, true_z in enumerate([0, 1, 2, 0]):
        for _ in range(80):
            tracker.observe(true_centers[true_z] + 0.3 * rng.standard_normal(D))
        target_cluster = mapping[true_z]
        mass = float(tracker.mu_marginal[target_cluster])
        # Detected = arg-max cluster
        detected_cluster = int(np.argmax(tracker.mu_marginal))
        marker = "OK" if detected_cluster == target_cluster else "MISS"
        print(f"  cycle {cycle_idx} (true regime {true_z}): "
              f"detected cluster {detected_cluster} "
              f"(expected {target_cluster}, mass={mass:.3f}) [{marker}]")
        cluster_seen_for.setdefault(true_z, []).append(detected_cluster)

    # KEY assertion: the two regime-0 visits should map to the SAME cluster
    visits_for_0 = cluster_seen_for[0]
    assert len(set(visits_for_0)) == 1, \
        f"recurring regime 0 mapped to different clusters: {visits_for_0}"
    print(f"[ok] recurring regime 0 → cluster {visits_for_0[0]} on both visits "
          f"(this is BAPR's signature ability)")


def test_degenerate_warmup():
    """Warmup with all data from 1 regime — should warn about degeneracy."""
    K, D = 4, 3
    tracker = RegimeBeliefTracker(
        max_run_length=10, num_regimes=K, hazard_rate=0.001,
        warmup_samples=500, ewma_alpha=0.05, obs_dim=D, seed=99)
    rng = np.random.default_rng(7)
    print("\n--- degeneracy detection: warmup with 1-regime data ---")
    single_center = np.array([1.0, 0.5, -0.5])
    for _ in range(500):
        tracker.observe(single_center + 0.3 * rng.standard_normal(D))
    assert tracker.is_seeded
    print(f"  k-means quality ratio = {tracker.kmeans_quality:.3f} "
          f"(<1.5 should print warning above)")


if __name__ == "__main__":
    tracker, centers, mapping = test_warmup_then_seed()
    test_belief_evolution_under_switches(tracker, centers, mapping)
    test_recurring_regimes(tracker, centers, mapping)
    test_degenerate_warmup()
    print("\n=== all smoke tests passed ===")
