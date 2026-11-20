# JMLR Upgrade Protocol: Brax Mechanism Extension (v50) and Standard MuJoCo Multi-Morphology Zero-Extra-Budget Suite (v51-samestep)

Frozen before full-batch execution. Existing confirmatory cohorts (`reproducibility/v32`, `reproducibility/reacher_confirmation`, `reproducibility/halfcheetah_runtime_comparison`, `reproducibility/submission_extensions_v47`) remain strictly unmodified and unpooled.

## 1. Brax Frozen-Policy Deep Mechanism & Generalization Audit (`jmlr-brax-mechanism-v50`)
- **Runtime pinned**: JAX `0.4.38`, Brax `0.10.5`, Flax `0.10.4`, MuJoCo `3.1.6`.
- **Policy cohort**: All 10 prospective HalfCheetah canonical policies (`88003..88179`) and frozen five-head inverse ensemble (`model_seed = 20260732`), zero retraining.
- **Endpoints**:
  1. Unified reporting of single-step first-correct delay, sustained 8-step switch delay, and 32-step post-switch regret on the primary 4-mode benchmark.
  2. Complete filter & ensemble ablation matrix: `δ ∈ {0.90, 0.95, 0.98, 1.00}` with `h = 0.002`, pure untempered Bayesian lock-in (`h = 0, δ = 1.00`) vs. pure tempered forgetting (`h = 0, δ = 0.98`), memoryless `instantaneous`, single-head `K = 1`, and `O(d)` channel-factored filter (reporting both positive gains and null/negative contrasts transparently).
  3. High-dimensional `M = 2^6 = 64` polarity hypercube (15 four-block episodes sampling 43 distinct modes while scoring all 64 candidates online at every step, including single-joint Hamming-1 flips): `O(d)` channel-factored binary belief filter vs. `O(2^d)` joint 64-mode filter, oracle, and uncompensated baselines.
  4. Non-bijective lossy actuator attenuation (`0.5x` torque loss + polarity inversions): box-projected compensation validating the two-term **Infeasibility Clipping Gap + Causal Inference Loss** decomposition.

## 2. Standard Gymnasium MuJoCo v5 Multi-Morphology & Zero-Extra-Budget Suite (`jmlr-mujoco-v5-multimorphology-v51-samestep`)
- **Runtime pinned**: PyTorch `2.5.1+cu121`, Gymnasium `1.2.0`, MuJoCo `3.2.7`.
- **Vectorized reset handling**: Explicit `gym.vector.AutoresetMode.SAME_STEP` across 8 vectorized environments so every step executes 8 real physics transitions (`62,500 * 8 = 500,000` exact real physics transitions per arm) and terminal states are extracted strictly from `infos["final_obs"]`, with full replay-buffer + RNG state checkpointing.
- **Morphologies**: `HalfCheetah-v5` (`d=6`), `Ant-v5` (`d=8`, 27D proprioceptive state), `Walker2d-v5` (`d=6`), `Hopper-v5` (`d=3`).
- **Policy seeds**: `{51003, 51021, 51039, 51057, 51079}` (`n = 5` independent seeds per morphology).
- **Equal online interaction budget**: `500,000` real physics transitions per controller (`0` external provider interactions):
  1. `bapr` (**Zero-Extra-Budget Self-Calibrated BAPR**): trains strictly in fixed reference mode 0 with 4-step uniform exploratory bursts at trigger probability `0.0125` (`~5%` steps), then fits both its 5-head inverse ensemble (`4,500` AdamW updates) and a structure-matched 5-head shared-physics forward ensemble (`bapr_forward_shared`, `4,500` AdamW updates) on the exact same 90%/10% split of its 500k mode-0 replay buffer (`v_t = G_0 a_t`, zero multimode labels).
  2. `bapr_noburst` (**Burst-Free Mode-0 Control**): identical to `bapr` with `0%` uniform bursts, isolating the effect of burst exploration on canonical balance headroom and self-calibrated identification.
  3. `sac`: equal-capacity `[256, 256]` actor / `[256, 256, 256]` twin-critic memoryless SAC trained across switching modes.
  4. `escp256`: capacity-matched `[256, 256]` / `[256, 256, 256]` recurrent ESCP with 16-step `FC(256)->GRU(128)->8D tanh` context encoder trained with Luo et al. (2022)'s authentic RMDM objective (EMA prototypes `τ = 0.995`, within-regime consistency `50.0`, cross-regime RBF Cholesky `logdet` diversity `0.025` at `r = 80.0`).
  5. `rma`: equal-budget Rapid Motor Adaptation (`400,000` privileged extrinsics base steps + `100,000` online history-adapter steps = `500,000` total steps).
  6. `belief_sac`: same-information `[256, 256]` / `[256, 256, 256]` SAC receiving `[s_t, b_t]` from the exact same frozen self-calibrated estimator.
- **Additional generalization & coverage accounting**: `O(d)` channel-factored hypercube switching on `HalfCheetah-v5` (`M=64`) and `Ant-v5` (`M=256`), `0.5x` lossy gain projection across all four morphologies, full-horizon zero-padded 32-step switch regret alongside survival-conditioned regret, and explicit reporting of observed switch coverage (`/225`) and seeds reaching switches (`/5`).

## Post-run implementation clarification (2026-11-20)

The 120 final manifests report fresh starts. The historical split checkpoint
contained replay and NumPy generator state but omitted the live MuJoCo and
Torch RNG state. The released runner writes all recovery state in one atomic
file for future retries. It cannot reconstruct state missing from the old
format. No published result or registered decision was recalculated.

The wider ESCP and two-phase RMA arms are task adaptations. Belief-SAC's 500k
direct multimode steps are separate from the paired BAPR replay used to fit
their shared estimator. The publication reports these costs separately.
