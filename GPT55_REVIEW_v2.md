================================================================================
GPT-5.5 ROUND 2 RESPONSE
================================================================================
Below I’m treating line numbers as approximate offsets in the supplied dump, since the dump has no explicit numbering. I’ll reference `file.py:~line` plus the exact function/block so you can jump quickly.

---

# Executive Round-2 verdict

After seeing the full repo, the biggest missed issues are **not** the BOCD mapping alone. There are several code-level problems that can completely change the interpretation of the results:

1. **Evaluation is not measuring non-stationary generalization.**  
   `test_tasks` are identical to `train_tasks`, and `evaluate()` evaluates only `tasks[0]`, not a task distribution or switching sequence.  
   Pointers: `train.py:~199-205`, `train.py:~72-118`.

2. **Task labels stored in replay are wrong at switch boundaries.**  
   `collect_samples()` records `prev_task_id`, rolls out under that task, then pushes all transitions with `env.current_task_id` **after** the rollout, which may already have switched.  
   Pointer: `train.py:~149-160`.

3. **BOCD is partially oracle-informed and partially neutered.**  
   The tracker is reset when the hidden environment task ID changes. This uses privileged switch information and also makes the first post-switch surprise return zero because `SurpriseComputer` resets.  
   Pointer: `train.py:~162-165`, `bapr.py:~191-194`, `belief_tracker.py:~64-80`.

4. **The ablations are not controlled.**  
   `BAPRNoBocd`, `BAPRNoAdaptBeta`, `BAPRFixedDecay`, and `BAPRNoRmdm` do not simply remove one BAPR component; most use ESCP-style independent critic targets while full BAPR uses a shared min target.  
   Pointers: `bapr.py:~103-111` vs `bapr_ablations.py:~67-78`.

5. **The paper claims RE-SAC-style IPM / OOD critic regularization, but the code does not implement it.**  
   `weight_reg` in `RESAC` is added to the **policy loss**, while being independent of policy parameters, so it has zero gradient effect. BAPR/ESCP do not include the claimed critic regularizers either.  
   Pointers: `resac.py:~63-72`, `bapr.py:~103-113`, `escp.py:~91-101`.

6. **Context warmup is only used during updates, not during data collection.**  
   During the first `context_warmup_iters`, training zeroes context embeddings, but rollout still conditions the policy on the random context network. This breaks the claimed “pure RESAC warmup”.  
   Pointers: `bapr.py:~94-99`, `escp.py:~106-111`, `brax_env.py:~287-300`.

7. **HalfCheetah evaluation reward is likely scaled incorrectly.**  
   HalfCheetah never terminates in this wrapper, so `evaluate()` runs `n_episodes * max_episode_steps` as one long trajectory and returns one accumulated reward, not average per episode.  
   Pointers: `brax_env.py:~59-64`, `train.py:~94-114`.

These are accept-blocking if not fixed or honestly disclosed.

---

# Task A — Code-level audit

## A1. Implementation bugs / logical inconsistencies

### 1. Train/test task leakage

**File:** `jax_experiments/train.py:~199-205`  
**Code:**
```python
train_tasks = env.sample_tasks(config.task_num)
test_tasks = env.sample_tasks(config.test_task_num)
```

**File:** `jax_experiments/envs/brax_env.py:~190-205`  
**Code:**
```python
def sample_tasks(self, n_tasks: int) -> List[Dict]:
    tasks = []
    rng = np.random.RandomState(self.seed + 42)
    for _ in range(n_tasks):
        ...
```

Every call to `sample_tasks()` uses the same `RandomState(self.seed + 42)`. Therefore, if `task_num == test_task_num`, `test_tasks == train_tasks`.

Then evaluation only uses the first task:

**File:** `train.py:~90-94`
```python
if tasks is not None:
    env.set_task(tasks[0])
```

So the reported evaluation is not OOD, not held-out, and not non-stationary. It is deterministic eval on the first training task.

**Actionable fix:**
- Use separate RNG streams or pass an offset.
- Evaluate across all held-out tasks or a non-stationary held-out schedule.

---

### 2. Replay task IDs are wrong around switches

**File:** `train.py:~149-160`, `collect_samples()`  
**Current code:**
```python
prev_task_id = env.current_task_id  # record before rollout
(obs, act, rew, nobs, done), ep_rewards = env.rollout(
    policy_params, n_steps, rng_key, context_params=context_params)

# Zero-copy push: JAX arrays go directly to GPU-native replay buffer
task_ids = jnp.full(n_steps, env.current_task_id, dtype=jnp.int32)
replay_buffer.push_batch_jax(obs, act, rew.reshape(-1, 1),
                             nobs, done.reshape(-1, 1), task_ids)
```

But `env.rollout()` uses the current system for the whole rollout and only switches after:

**File:** `brax_env.py:~405-411`
```python
self._step_counter += n_steps
self._check_switch()   # switches _current_sys so next rollout uses new task
self._state = final_state
```

Thus, the rollout transitions were generated under `prev_task_id`, but may be stored using the **next** `env.current_task_id`.

Given server config:

**File:** `run_server_BAPR.sh:~147-151`
```bash
CHANGING_PERIOD=40000
...
samples_per_iter=4000
```

One 4000-step rollout per 40000-step task gets mislabeled, i.e. around 10% label contamination. With default config `changing_period=20000`, it is around 20%.

This directly harms `compute_rmdm_loss()`, since the context network is supervised by `task_id`.

**Actionable fix:**
```python
rollout_task_id = env.current_task_id
...
task_ids = jnp.full(n_steps, rollout_task_id, dtype=jnp.int32)
```

If you later implement intra-rollout switching, return a vector of task IDs from `env.rollout()`.

---

### 3. BOCD uses privileged switch information and is reset exactly at hidden task changes

**File:** `train.py:~162-165`
```python
if hasattr(agent, 'reset_episode') and env.current_task_id != prev_task_id:
    agent.reset_episode()
```

**File:** `bapr.py:~191-194`
```python
def reset_episode(self):
    self.belief_tracker.reset()
    self.surprise_computer.reset()
```

This is inconsistent with the paper’s “switches are not signaled” claim. Worse, it reduces the chance that BOCD detects the actual switch: after reset, the next call to `SurpriseComputer.compute()` returns zero:

**File:** `belief_tracker.py:~73-80`
```python
if self.ema_reward is None:
    self.ema_reward = r_mean
    self.ema_reward_var = 1.0
    self.ema_q_std = q_std_mean
    return 0.0
```

So the first post-switch batch initializes the EMA instead of producing a high surprise.

**Consequence:** BOCD is neither fully unsupervised nor reliably detecting changes. It is being reset by an oracle switch signal and then prevented from firing on that switch.

**Actionable fix:**
- Do not call `agent.reset_episode()` on task switch during training.
- If you want episode resets, reset only on actual environment `done`, not hidden regime switch.
- Log hidden switch times for analysis only.

---

### 4. Context warmup is not applied during rollout

**Files:**  
- `bapr.py:~94-99`, `escp.py:~106-111`: context zeroed during update.  
- `brax_env.py:~287-300`: context always used during rollout.

Update-time warmup:

```python
ep = ctx_m(obs)
ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
```

But data collection always does:

```python
ctx_net = nnx.merge(context_graphdef, context_params)
ep = ctx_net(pre_obs[None])
action, _ = policy.sample(pre_obs[None], key1, ep)
```

So during early training:
- Critic/policy are trained with `ep = 0`.
- Policy collects data with random nonzero `ep`.

This is an off-policy distribution mismatch and invalidates the “pure RESAC warmup” claim from `GOLDEN_BASELINE.md`.

**Actionable fix:** pass a `use_context` boolean into `env.rollout()` and zero `ep` there during warmup.

---

### 5. Full BAPR and ESCP use different critic targets

**Full BAPR critic target:** `bapr.py:~103-111`
```python
tq = tm(next_aug, na).min(axis=0) - alpha * nlp
tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq
...
return jnp.mean((pq - tv[None]) ** 2), pq
```

**ESCP critic target:** `escp.py:~91-101`
```python
tq_all = tm(next_aug, na) - alpha * nlp
tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_all
...
return jnp.mean((pq - tv_all) ** 2), pq
```

This is a major algorithmic difference:
- BAPR uses a shared clipped/min target.
- ESCP uses independent per-head targets.

This may explain BAPR underperformance: full BAPR is more pessimistic before the adaptive beta even acts.

However, `GOLDEN_BASELINE.md` says independent targets were toxic for BAPR:

```md
❌ Independent target Q ... DO NOT DO THIS
```

That means the comparison is not apples-to-apples:
- ESCP has a more optimistic target that may improve reward.
- BAPR has a safer target that may stabilize Q-std and BOCD.

**Actionable fix:** make the critic target choice explicit in config and run controlled comparisons:

```python
config.critic_target_mode in {"min", "independent"}
```

Then report BAPR/ESCP under both modes.

---

### 6. BAPR ablations are not controlled ablations

**File:** `bapr_ablations.py:~40-145`, helper `_build_escp_scan_with_beta()` uses independent targets:
```python
tq_all = tm(next_aug, na) - alpha * nlp
tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_all
```

But full BAPR uses min target. Therefore:

- `BAPRNoBocd` is not “BAPR minus BOCD”; it is closer to ESCP.
- `BAPRNoAdaptBeta` is not “BAPR minus adaptive beta”; it also changes critic target.
- `BAPRFixedDecay` also changes critic target.
- `BAPRNoRmdm` uses independent target too.

This invalidates component attribution.

**Actionable fix:** factor full BAPR’s scan body into a shared builder with flags:
```python
use_context: bool
use_bocd_beta: bool
use_fixed_decay: bool
critic_target_mode: Literal["min", "independent"]
```

Then each ablation changes exactly one flag.

---

### 7. RE-SAC weight regularization is a no-op

**File:** `resac.py:~63-72`
```python
def policy_loss_fn(pp):
    pm = nnx.merge(gd_policy, pp)
    cm = nnx.merge(gd_critic, new_c_p)
    ...
    reg = cm.compute_reg_norm()
    return (jnp.exp(la) * lp - lcb).mean() + weight_reg * reg.mean(), lp
```

The gradient is taken with respect to `pp`:

```python
(p_loss, lp), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(p_p)
```

But `reg = cm.compute_reg_norm()` depends on critic parameters `new_c_p`, not policy parameters `pp`. Therefore `weight_reg * reg.mean()` contributes **zero gradient**.

The code comment says:

**File:** `resac.py:~1`
```python
"""RE-SAC: SAC + ensemble + OOD reg + LCB policy — scan-fused."""
```

But the critic is not regularized.

**Actionable fix:** put the regularizer in `critic_loss_fn(cp)`:

```python
reg = nnx.merge(gd_critic, cp).compute_reg_norm().mean()
return td_loss + weight_reg * reg, pq
```

Also decide whether BAPR/ESCP should inherit this. Right now the paper claims BAPR uses IPM/aleatoric risk regularization, but BAPR code does not.

---

### 8. Paper critic loss diverges from code

Paper Eq. `bapr_critic_loss` claims:

```tex
L_Q = TD error + λ_ale ∑ ||W||_1 + β_ood σ_ens(s,a)
```

But full BAPR critic loss is only TD MSE:

**File:** `bapr.py:~103-113`
```python
pq = cm(obs_aug, act)
return jnp.mean((pq - tv[None]) ** 2), pq
```

No:
- `weight_reg`
- `beta_ood`
- behavior cloning
- OOD regularization
- critic ensemble variance penalty

This is a direct paper-code mismatch.

---

### 9. Surprise signal mostly does not use “recent rollout” data

**File:** `bapr.py:~211-227`
```python
last_obs = jnp.array(stacked_batch["obs"][-1])
last_act = jnp.array(stacked_batch["act"][-1])
q_std = self._compute_q_stats(..., last_obs, last_act, warmup)
...
if recent_rewards is not None and len(recent_rewards) > 0:
    reward_signal = np.array(recent_rewards, dtype=np.float32)
else:
    reward_signal = stacked_batch["rew"][-1].flatten()
```

Problems:

1. `stacked_batch["obs"][-1]` is a random replay minibatch, not recent transitions.
2. For HalfCheetah, `done = 0` always in this wrapper, so `ep_rewards` is often empty and BAPR falls back to random replay rewards.
3. The Q-std surprise comes from random replay, which may contain stale tasks.

**Consequence:** BOCD is not detecting the latest regime change; it is detecting noise in random replay samples.

**Actionable fix:** return recent rollout `(obs, act, rew)` from `collect_samples()` and compute surprise from that.

---

### 10. HalfCheetah evaluation is not episode-normalized

**File:** `brax_env.py:~59-64`
```python
if brax_name == 'halfcheetah':
    reward = ...
    done = jnp.float32(0.0)
```

**File:** `train.py:~84-114`, `evaluate()`
```python
n_steps = n_episodes * config.max_episode_steps
...
for i in range(n_steps):
    ep_r += rew_np[i]
    if done_np[i] > 0.5:
        ep_rewards.append(ep_r)
...
if not ep_rewards:
    ep_rewards = [ep_r]
return float(np.mean(ep_rewards)), ...
```

For HalfCheetah with `eval_episodes=5`, this returns a 5000-step return as one episode. Standard MuJoCo reporting is usually 1000-step return. So HalfCheetah numbers may be inflated by roughly `eval_episodes`.

This affects BAPR vs ESCP equally if both evaluated the same way, but it invalidates comparison to papers and table claims.

**Actionable fix:** segment by `max_episode_steps` even if `done` is false.

---

### 11. `compute_rmdm_loss()` silently caps task diversity at 20 tasks while config uses 40

**File:** `context_net.py:~43-50`
```python
max_tasks = 20
unique_tasks = jnp.unique(task_ids, size=max_tasks, fill_value=-1)
```

**File:** `default.py:~11-15`
```python
task_num: int = 40
test_task_num: int = 40
```

With replay batches sampled across 40 tasks, only 20 task IDs can participate in the RMDM loss. This is silent truncation.

**Actionable fix:** pass `max_tasks=config.task_num` into `compute_rmdm_loss()` as a static config value, or reduce `task_num`.

---

### 12. Context architecture is not actually history-based ESCP

Config includes:

**File:** `default.py:~62`
```python
rnn_fix_length: int = 16  # history window for context (FC mode = no RNN)
```

But `ContextNetwork` maps only current observation:

**File:** `context_net.py:~12-32`
```python
class ContextNetwork(nnx.Module):
    ...
    def __call__(self, obs):
        ...
```

For gravity/damping/mass identification, a single state often does not identify the dynamics; transitions `(s,a,r,s')` or a short history are much more informative.

The current context encoder may learn state-distribution clusters, not true dynamics modes.

**Actionable fix:** implement a transition/history encoder:
```python
context_input = concat(s_t, a_t, r_t, s_{t+1} - s_t)
```
or an RNN/TCN over the last `rnn_fix_length` transitions.

---

### 13. Paper says randomized or Poisson switching; code uses deterministic boundary switching

Paper:

```tex
regime switches occur at randomized intervals (drawn from a Poisson distribution...)
```

Code:

**File:** `brax_env.py:~231-236`
```python
if self._step_counter % self._changing_interval == 0:
    idx = int(self._step_counter / self._changing_period) % len(self._tasks)
```

Switches are deterministic and aligned to rollout boundaries because `rollout()` switches only after the scan.

**Actionable fix:** either:
- update the paper to deterministic periodic switching, or
- implement per-step switching inside the rollout scan.

---

### 14. `BadBAPR` does not validate the theorem as written

Paper says Bad-BAPR validates Q-dependent belief breaking contraction. But code does:

**File:** `bad_bapr.py:~140-149`
```python
q_current = cm_current(obs_aug, act)
q_std_current = q_current.std(axis=0).mean()
q_dependent_lambda = jnp.clip(q_std_current / 10.0, 0.0, 1.0)
effective_beta = beta_base - q_dependent_lambda * penalty_scale
```

This is not a BOCD belief depending on Q, nor a Q-dependent Bellman operator. It is a policy-loss coefficient depending on current Q-ensemble std.

This may still be a useful bad variant, but it is not the theorem’s counterexample.

**Actionable fix:** rename it to `QStdAdaptiveBetaBAPR` or implement the actual theorem-matching Q-dependent mixture.

---

## A2. Code-level differences between BAPR and ESCP/RESAC that may explain BAPR underperformance

### BAPR vs ESCP

| Component | ESCP | BAPR | Likely effect |
|---|---:|---:|---|
| Critic target | Independent per-head target | Shared min target | BAPR more pessimistic; ESCP may learn higher-return policy |
| Policy beta | Static `beta=-2` | `beta_eff <= -2` | BAPR can only become more conservative |
| BOCD overhead | None | Surprise + belief + adaptive beta | Adds noise source |
| Surprise source | N/A | Random replay / episodic rewards | Can trigger false conservatism |
| Context | Same architecture | Same architecture | No representational advantage for BAPR |
| Replay | Uniform | Uniform | BAPR does not actually forget stale data |
| Oracle task reset | No | Yes, via `reset_episode()` | Invalidates BOCD story and can suppress detection |

Main point: in this codebase, BAPR is essentially **ESCP plus a noisy conservatism scheduler, but with a more conservative critic target**. That is a structural disadvantage if the environment is smooth continuous gravity variation.

### BAPR vs RESAC

| Component | RESAC | BAPR |
|---|---:|---:|
| Context network | No | Yes |
| Policy/critic input | Raw obs | Obs + EP |
| Critic target | Independent target | Shared min target |
| LCB | Static | Adaptive, always more conservative |
| Weight regularization | Intended but no-op | None |
| Initialization | No context net before policy | Context net consumes RNG before policy |

One subtle fairness issue: RESAC and BAPR do not have matched initialization because BAPR constructs `ContextNetwork` before policy/critic using the same `nnx.Rngs(seed)`. RESAC does not. BAPR vs ESCP is matched; BAPR/ESCP vs RESAC/SAC are not.

---

## A3. Paper-code divergences

Major divergences:

1. **BOCD not purely unsupervised**  
   Paper: switches unobserved.  
   Code: hidden task switch triggers tracker reset.  
   Pointer: `train.py:~162-165`.

2. **Evaluation not held-out and not non-stationary**  
   Paper: held-out non-stationary tasks.  
   Code: same tasks; only `tasks[0]`.  
   Pointers: `train.py:~90-94`, `envs/brax_env.py:~190-205`.

3. **Critic regularization absent**  
   Paper Eq. `bapr_critic_loss`: critic IPM/OOD penalties.  
   Code: TD MSE only.  
   Pointer: `bapr.py:~103-113`.

4. **RMDM labels are explicit task IDs**  
   Paper says mode labels only in sim; okay, but current eval uses same sampled task IDs and does not test deployment without labels.

5. **Randomized switching not implemented**  
   Paper: Poisson/randomized intervals.  
   Code: deterministic periodic switching.  
   Pointer: `brax_env.py:~231-236`.

6. **Mode mixture theorem not implemented**  
   Paper proves a belief-weighted mixture over mode operators.  
   Code uses a single shared critic and scalar `beta_eff`. The paper has a “tractable approximation” remark, but the empirical claims should not sound like direct validation of the theorem.

7. **Adaptive lambda equation mismatch**  
   Paper currently says baseline-subtracted lambda:
   ```tex
   λ_w = max(0, hbar/(H-1) - EMA)
   ```
   Full BAPR code uses:
   ```python
   return clip(ew / max_h, 0, 1)
   ```
   Pointer: `bapr.py:~195-205`.  
   `GOLDEN_BASELINE.md` explicitly says EMA subtraction is poison, so the paper must be changed.

8. **Detection delay theorem unsupported by implemented likelihood**  
   The likelihood is a fixed Gaussian variance schedule over run lengths:
   ```python
   variances = base_var + arange(H) * var_growth
   ```
   It does not infer a mode, only reweights run-length bins based on surprise magnitude. Claims about convergence to the “correct mode” are not supported by this implementation.

---

# Task B — New rescue strategies after seeing full repo

These are new relative to the Round-1 scalar-lambda discussion.

## 1. Make BAPR actually “amnesic” through replay, not just beta

The paper defines replay staleness, but code samples uniformly forever:

**File:** `replay_buffer.py:~104-129`
```python
idx = jax.random.randint(rng_key, (n_batches, batch_size), 0, self.size)
```

BOCD should change the replay distribution after a detected switch.

Practical scheme:
- Maintain previous `lambda_w`.
- If `lambda_w > 0.3`, sample 50–80% from the last `W=50k` transitions.
- Otherwise sample uniformly.
- Optionally decay recent fraction as lambda decays.

This directly attacks the “regime staleness” problem. Current BAPR does not.

Expected impact: high. This is more likely to beat ESCP than only changing beta.

---

## 2. Use recent rollout rewards and transition residuals for surprise

Current surprise uses episode returns if any, else random replay rewards. For HalfCheetah, episode returns are usually absent.

Better:
- Always compute `r_mean`, `r_std`, reward drop from the latest rollout’s step rewards.
- Compute Q-std on latest rollout `(obs, act)`.
- Add TD residual surprise:
  \[
  |\hat Q(s,a) - (r + \gamma V(s'))|
  \]
  averaged over recent transitions.

TD residual is more aligned with “model/value misalignment after switch” than raw reward.

---

## 3. Replace current context net with a transition/history encoder

The existing context network is `obs -> ep`. For dynamics identification, use:

```python
x_t = concat(s_t, a_t, r_t, s_{t+1} - s_t)
```

Then feed a short sequence to:
- GRU,
- temporal convolution,
- or mean-pooled MLP over last `rnn_fix_length`.

The config already has `rnn_fix_length`; use it.

This is the cleanest “borrow from ESCP original” move.

---

## 4. BOCD-generated pseudo-labels for unsupervised RMDM

You already have `bapr_unsupervised` in `train.py` choices, but no dump for it. The natural version:

- Use BOCD changepoints to segment replay.
- Assign segment IDs as pseudo-task IDs.
- Train RMDM on segment IDs instead of simulator `task_id`.

This removes the sim-label crutch and better matches deployment.

---

## 5. Per-mode / per-context beta instead of scalar global beta

Use context embedding to predict beta:

```python
beta_eff(s) = beta_base - penalty_scale * lambda_w * g(e_s)
```

where `g(e)` is a small bounded MLP. This lets BAPR be conservative only in ambiguous regions/modes, not globally.

A safe version:
```python
g(e) = sigmoid(MLP(stop_gradient(e)))
```

Keep `lambda_w` frozen per update to preserve the theorem’s spirit.

---

## 6. CP-triggered short-horizon finetuning

When BOCD fires:
- run extra critic updates on recent transitions,
- temporarily increase target update `tau`,
- optionally freeze actor for N updates to avoid exploiting stale Q.

This is cheap and directly targets post-switch recovery.

---

## 7. Context-balanced minibatches

RMDM needs multiple task IDs per batch. Uniform replay can be dominated by recent/current tasks or noisy old tasks. Add a sampler that ensures:
- at least `m` task IDs per batch,
- at least `k` samples per task ID.

This improves context geometry and makes the RMDM loss meaningful.

---

## 8. Switch from run-length posterior mean to CP hazard statistic

Even if you keep the current likelihood, log and use:

```python
cp_mass = belief[0]
recent_mass = belief[:h_recent].sum()
entropy = normalized_entropy(belief)
```

Then beta can depend on a small state machine:
- high surprise or high entropy → enter conservative mode,
- sustained low surprise → exit.

This is more robust than raw expected run length.

---

# Task C — BAPR vs ESCP head-to-head architectural comparison

## 1. Critic

**ESCP:** independent target per ensemble head.  
Pointer: `escp.py:~91-101`.

**BAPR:** shared min target.  
Pointer: `bapr.py:~103-111`.

**ESCP advantage:** less pessimistic critic, potentially higher reward in smooth continuous tasks.

**Borrow move:** expose target mode as config and run:
```bash
--critic_target_mode min
--critic_target_mode independent
```
for both ESCP and BAPR.

Do not claim BAPR improvement unless target mode is controlled.

---

## 2. Policy objective

**ESCP:**
```python
lcb = qv.mean(axis=0) + beta * qv.std(axis=0)
```

**BAPR:**
```python
effective_beta = beta_base - weighted_lambda * penalty_scale
lcb = qv.mean(axis=0) + effective_beta * qv.std(axis=0)
```

Since `beta_base = -2`, BAPR is always at least as conservative as ESCP. If lambda is noisy, BAPR loses.

**ESCP advantage:** no false-positive conservatism.

**Borrow move:** make BAPR adaptive beta two-sided or gated:
```python
if detector_confident:
    beta_eff = beta_base - penalty
else:
    beta_eff = beta_base
```
Do not continuously penalize from raw `ew/H`.

---

## 3. Target update

Same soft update:

```python
new_t_p = tau * critic + (1 - tau) * target
```

No structural ESCP advantage here.

Potential BAPR-specific improvement:
- temporarily increase `tau` after detected change to adapt target faster.

---

## 4. Ensemble

Same vectorized `EnsembleCritic`.

But because BAPR uses min target, its ensemble members are pulled toward a common conservative target, which may reduce useful diversity. ESCP’s independent targets preserve head-specific variation.

**Borrow move:** if independent target is unstable for BOCD, use hybrid:
```python
target = mix * independent + (1 - mix) * min_target
```
with `mix=0` during warmup/BOCD high surprise and `mix=1` during stable periods.

---

## 5. Context

Same context network and RMDM.

But neither is true ESCP original if original used history. The repo’s ESCP advantage over BAPR is not context; it is mostly simpler training and less conservatism.

**Borrow move for BAPR:** implement history-based EP.

---

## 6. Replay

Same uniform replay. This is a missed opportunity for BAPR.

ESCP implicitly tolerates stale replay because context labels/task IDs allow the critic to condition on task. BAPR claims “amnesia” but does not filter replay.

**Borrow move:** from ESCP’s context idea, sample task-balanced batches and condition critic strongly; from BOCD, sample recent-heavy after switch.

---

## 7. Env interaction

Same rollout path, but BAPR additionally resets detector on hidden task switch.

**ESCP advantage:** no fragile detector state and no oracle reset mismatch.

**Borrow move:** remove oracle reset, make BAPR detector consume the same information ESCP consumes.

---

# Task D — Top-3 concrete diffs

## Change 1 — Fix rollout task IDs, remove oracle BOCD reset, return recent rollout stats

### Before

**File:** `jax_experiments/train.py:~149-165`
```python
prev_task_id = env.current_task_id  # record before rollout
(obs, act, rew, nobs, done), ep_rewards = env.rollout(
    policy_params, n_steps, rng_key, context_params=context_params)

# Zero-copy push: JAX arrays go directly to GPU-native replay buffer
task_ids = jnp.full(n_steps, env.current_task_id, dtype=jnp.int32)
replay_buffer.push_batch_jax(obs, act, rew.reshape(-1, 1),
                             nobs, done.reshape(-1, 1), task_ids)

# Reset belief only when TASK switches (not on every episode end)
if hasattr(agent, 'reset_episode') and env.current_task_id != prev_task_id:
    agent.reset_episode()

return ep_rewards
```

### After

```python
# Task/system used for this whole scan rollout.
# env.rollout() switches only after the scan, so labels must use this ID.
rollout_task_id = env.current_task_id

(obs, act, rew, nobs, done), ep_rewards = env.rollout(
    policy_params, n_steps, rng_key, context_params=context_params)

task_ids = jnp.full(n_steps, rollout_task_id, dtype=jnp.int32)
replay_buffer.push_batch_jax(
    obs, act, rew.reshape(-1, 1), nobs, done.reshape(-1, 1), task_ids
)

# Do NOT reset BOCD from hidden task switches. That is privileged information
# and suppresses first post-switch surprise. Hidden switches may be logged only.
recent_rollout = {
    "obs": obs,
    "act": act,
    "rew": rew.reshape(-1, 1),
    "next_obs": nobs,
    "done": done.reshape(-1, 1),
    "task_id": task_ids,
    "rollout_task_id": int(rollout_task_id),
}

return ep_rewards, recent_rollout
```

Then in `train.py:~244`:

### Before
```python
ep_rewards = collect_samples(agent, env, replay_buffer, config, config.samples_per_iter)
```

### After
```python
ep_rewards, recent_rollout = collect_samples(
    agent, env, replay_buffer, config, config.samples_per_iter
)
```

And in BAPR dispatch:

### Before
```python
metrics = agent.multi_update(
    stacked, current_iter=iteration,
    recent_rewards=ep_rewards if ep_rewards else None)
```

### After
```python
metrics = agent.multi_update(
    stacked,
    current_iter=iteration,
    recent_rewards=ep_rewards if ep_rewards else None,
    recent_rollout=recent_rollout,
)
```

**Expected HC effect:**  
- Direct training lift from correct RMDM labels: `+200` to `+800` average HC reward.  
- More important: removes an accept-blocking methodological flaw.

---

## Change 2 — Compute BAPR surprise from latest rollout, not random replay

### Before

**File:** `jax_experiments/algos/bapr.py:~211-227`
```python
# Q-std from critic (always from replay batch — evaluates uncertainty)
last_obs = jnp.array(stacked_batch["obs"][-1])
last_act = jnp.array(stacked_batch["act"][-1])
q_std = self._compute_q_stats(
    nnx.state(self.critic, nnx.Param),
    nnx.state(self.context_net, nnx.Param),
    last_obs, last_act, warmup)

# Reward signal: prefer actual rollout episode rewards (direct task signal)
# over random replay batch rewards (may be from old tasks)
if recent_rewards is not None and len(recent_rewards) > 0:
    reward_signal = np.array(recent_rewards, dtype=np.float32)
else:
    reward_signal = stacked_batch["rew"][-1].flatten()

surprise = self.surprise_computer.compute(reward_signal, np.array([float(q_std)]))
self.belief_tracker.update(surprise)
```

### After

```python
# Prefer most recent rollout transitions for BOCD. Replay minibatches are
# deliberately mixed across old regimes and are a bad changepoint signal.
if recent_rollout is not None:
    # Use a bounded suffix to avoid giant host transfers.
    w = getattr(self.config, "surprise_window", 1024)
    robs = jnp.asarray(recent_rollout["obs"][-w:])
    ract = jnp.asarray(recent_rollout["act"][-w:])
    rrew = np.asarray(recent_rollout["rew"][-w:]).reshape(-1)

    q_std = self._compute_q_stats(
        nnx.state(self.critic, nnx.Param),
        nnx.state(self.context_net, nnx.Param),
        robs, ract, warmup,
    )
    reward_signal = rrew
else:
    # Fallback for compatibility.
    last_obs = jnp.asarray(stacked_batch["obs"][-1])
    last_act = jnp.asarray(stacked_batch["act"][-1])
    q_std = self._compute_q_stats(
        nnx.state(self.critic, nnx.Param),
        nnx.state(self.context_net, nnx.Param),
        last_obs, last_act, warmup,
    )
    reward_signal = np.asarray(stacked_batch["rew"][-1]).reshape(-1)

surprise = self.surprise_computer.compute(
    reward_signal,
    np.asarray([float(q_std)], dtype=np.float32),
)
self.belief_tracker.update(surprise)
```

Add config:

**File:** `configs/default.py`
```python
surprise_window: int = 1024
```

Also fix `SurpriseComputer` so stable Q-ratio contributes zero-ish surprise.

### Before

**File:** `belief_tracker.py:~91-98`
```python
self.ema_q_std = self.ema_alpha * q_std_mean + (1 - self.ema_alpha) * self.ema_q_std
q_std_ratio = q_std_mean / max(self.ema_q_std, 1e-6)
...
surprise = 0.5 * reward_z + 0.3 * q_std_ratio + 0.2 * reg_div
```

### After

```python
old_q_std = self.ema_q_std
self.ema_q_std = (
    self.ema_alpha * q_std_mean
    + (1 - self.ema_alpha) * self.ema_q_std
)

# Only positive spikes should increase surprise.
q_std_spike = max(q_std_mean / max(old_q_std, 1e-6) - 1.0, 0.0)

self.last_surprise_r = float(reward_z)
self.last_surprise_q = float(q_std_spike)
self.last_surprise_kappa = float(reg_div)

surprise = 0.5 * reward_z + 0.3 * q_std_spike + 0.2 * reg_div
return float(np.clip(surprise, 0.0, 10.0))
```

**Expected HC effect:**  
`+500` to `+1500`, mainly by reducing false lambda spikes and rescuing bad seeds.

---

## Change 3 — Add belief-aware recent replay sampling after detected switches

### Before

**File:** `replay_buffer.py:~104-129`
```python
idx = jax.random.randint(
    rng_key, (n_batches, batch_size), 0, self.size)
```

Uniform replay ignores the staleness problem BAPR claims to solve.

### After

Add method:

**File:** `replay_buffer.py`
```python
def sample_stacked_mixed(self, n_batches: int, batch_size: int,
                         rng_key=None, recent_frac: float = 0.0,
                         recent_window: int = 50_000):
    """Uniform + recent replay mixture for post-changepoint adaptation."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    recent_frac = float(np.clip(recent_frac, 0.0, 0.9))
    n_recent = int(batch_size * recent_frac)
    n_uniform = batch_size - n_recent

    k1, k2 = jax.random.split(rng_key)

    idx_uniform = jax.random.randint(
        k1, (n_batches, n_uniform), 0, self.size
    )

    if n_recent > 0:
        lo = max(0, self.size - recent_window)
        idx_recent = jax.random.randint(
            k2, (n_batches, n_recent), lo, self.size
        )
        idx = jnp.concatenate([idx_uniform, idx_recent], axis=1)
    else:
        idx = idx_uniform

    return {
        "obs": self.obs[idx],
        "act": self.act[idx],
        "rew": self.rew[idx],
        "next_obs": self.next_obs[idx],
        "done": self.done[idx],
        "task_id": self.task_id[idx],
    }
```

Use it in `train.py:~265-274`:

### Before
```python
stacked = replay_buffer.sample_stacked(
    config.updates_per_iter, config.batch_size, rng_key=sample_key)
```

### After
```python
if config.algo == "bapr":
    prev_lam = float(getattr(agent, "_current_weighted_lambda", 0.0))
    recent_frac = min(0.8, max(0.0, prev_lam))
    stacked = replay_buffer.sample_stacked_mixed(
        config.updates_per_iter,
        config.batch_size,
        rng_key=sample_key,
        recent_frac=recent_frac,
        recent_window=getattr(config, "recent_replay_window", 50_000),
    )
else:
    stacked = replay_buffer.sample_stacked(
        config.updates_per_iter, config.batch_size, rng_key=sample_key)
```

Add config:

```python
recent_replay_window: int = 50_000
```

**Expected HC effect:**  
`+700` to `+2000` in true abrupt-switch settings. On smooth continuous gravity with weak switches, expected smaller: `+200` to `+800`. Main benefit is post-switch recovery and reducing stale-buffer poisoning.

---

# Additional must-fix diffs not in top 3

## Fix evaluation

**File:** `train.py:evaluate()`

Minimum fix:
```python
def evaluate(agent, env, config, tasks=None, n_episodes=10):
    ...
    task_returns = []

    eval_tasks = tasks if tasks is not None else [None]
    for task in eval_tasks:
        if task is not None:
            env.set_task(task)

        rew_np, done_np = env.eval_rollout(
            policy_params,
            n_episodes * config.max_episode_steps,
            rng_key,
            context_params=context_params,
        )

        ep_rewards = []
        ep_r = 0.0
        steps_in_ep = 0
        for r, d in zip(rew_np, done_np):
            ep_r += float(r)
            steps_in_ep += 1
            if d > 0.5 or steps_in_ep >= config.max_episode_steps:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                steps_in_ep = 0
                if len(ep_rewards) >= n_episodes:
                    break

        task_returns.append(np.mean(ep_rewards))

    return float(np.mean(task_returns)), float(np.std(task_returns))
```

Also sample test tasks with different seed.

---

## Fix RE-SAC regularizer placement

**File:** `resac.py:critic_loss_fn`

```python
td = jnp.mean((pq - tv_all) ** 2)
reg = cm.compute_reg_norm().mean()
return td + weight_reg * reg, pq
```

Then decide whether BAPR should include the same regularizer.

---

## Fix RMDM `max_tasks`

**File:** `context_net.py`
```python
def compute_rmdm_loss(..., max_tasks: int = 64):
    unique_tasks = jnp.unique(task_ids, size=max_tasks, fill_value=-1)
```

Pass from config:
```python
compute_rmdm_loss(ep, tids, rbf_r, cons_w, div_w, max_tasks=config.task_num)
```

Because JIT needs static shape, capture `max_tasks` in `_build_scan_fn()`.

---

# Task E — Updated mock NeurIPS/ICML review

## Summary

The paper proposes BAPR, a combination of robust ensemble SAC, ESCP-style context conditioning, and Bayesian online change detection to adapt conservatism in piecewise-stationary continuous control. The theoretical component proves that a frozen belief-weighted mixture of Bellman operators is a contraction and provides a counterexample for Q-dependent beliefs. The empirical codebase implements a JAX/Flax NNX agent with scan-fused updates and Brax non-stationary environments.

The idea of connecting change-point detection to adaptive robustness is interesting. However, after inspecting the implementation, I find substantial mismatches between the paper’s claims and the actual experimental code. Most importantly, evaluation does not appear to measure held-out non-stationary generalization; train and test tasks are sampled identically and evaluation uses only the first task. BOCD is reset using hidden task switch information, while the actual surprise signal often comes from random replay rather than recent transitions. Several ablations are uncontrolled because they change the critic target in addition to the named component. The code also omits critic regularizers claimed in the paper.

## Strengths

1. **Interesting high-level direction.**  
   Adaptive robustness under piecewise stationarity is important, and BOCD is a plausible mechanism.

2. **Efficient implementation.**  
   The scan-fused JAX implementation is thoughtful and likely fast.

3. **Clear frozen-belief design principle.**  
   The theoretical insistence that belief weights be frozen during Bellman backup is sensible and useful as an implementation audit rule.

4. **Good engineering around vectorized ensembles.**  
   `EnsembleCritic` is clean and avoids Python-loop overhead.

5. **Ablation intent is good.**  
   The repo attempts to isolate BOCD, RMDM, adaptive beta, and heuristic decay, though current ablations are not controlled.

## Weaknesses

1. **Evaluation protocol is flawed.**  
   `test_tasks` are identical to `train_tasks`, and `evaluate()` uses only `tasks[0]`. This does not support claims about non-stationary generalization or OOD robustness.

2. **BOCD is not truly online unsupervised change detection.**  
   The training loop resets the BOCD tracker when the hidden task ID changes. This uses privileged information and suppresses first post-switch surprise.

3. **Ablations are invalid as component attribution.**  
   Full BAPR uses a min target, while most ablations use independent per-head targets. This confounds “remove BOCD” with “change critic target”.

4. **Paper-code mismatch in the loss.**  
   The paper claims IPM/aleatoric/OOD critic penalties, but BAPR critic loss is plain TD MSE. RESAC’s regularizer is currently placed in the policy loss and has zero gradient effect.

5. **BAPR does not address replay staleness despite motivating it.**  
   Replay remains uniform over all history. Adaptive beta alone does not forget stale regimes.

6. **Context warmup mismatch.**  
   Updates zero context during warmup, but rollouts still use random context embeddings.

7. **The theorem is only loosely connected to the implementation.**  
   The proved object is a frozen belief-weighted mixture over mode-conditional Bellman operators. The implemented object is a single shared critic with scalar adaptive beta.

8. **HalfCheetah return calculation likely not episode-normalized.**  
   Because HalfCheetah never terminates, evaluation returns a multi-episode-length trajectory as one return.

## Questions for the authors

1. Why does evaluation use only `tasks[0]`? Do results change when averaging over all held-out tasks or held-out switching schedules?

2. Why is BOCD reset on hidden task switches? Is this information available to the agent?

3. Are reported ablations rerun with the same critic target as full BAPR?

4. Where in the released code are the paper’s critic regularizers implemented?

5. Does BAPR still outperform ESCP if both use the same critic target mode and the same evaluation protocol?

6. What is the performance under true intra-rollout random switch times?

7. For HalfCheetah, are returns normalized to 1000-step episodes?

## Score

Current submission: **Weak Reject / 4**

The core idea has promise, but the experimental evidence is not yet reliable because of train/test leakage, oracle reset, evaluation issues, and uncontrolled ablations.

## Confidence

**High on implementation concerns, medium on final empirical outcome.**

The code issues are direct and concrete. I am less certain whether fixing them will improve or reduce BAPR’s numbers; my expectation is that fair evaluation will reduce some reported gains but that replay amnesia + recent surprise could make BAPR genuinely stronger on abrupt-switch tasks.

## What would tip this to accept?

I would reconsider positively if the authors provide:

1. **Correct evaluation:**
   - distinct held-out task RNG,
   - evaluation over all held-out tasks or held-out switch sequences,
   - proper episode normalization.

2. **No oracle BOCD reset:**
   - hidden task ID only used for logging / optional supervised RMDM,
   - BOCD updated from recent rollout signals.

3. **Controlled ablations:**
   - same critic target mode,
   - same replay,
   - same context warmup,
   - only one component removed at a time.

4. **Replay amnesia experiment:**
   Show that BOCD improves post-switch recovery by changing replay sampling or update weighting, not only beta.

5. **Post-switch metrics:**
   Report:
   - detection delay,
   - reward drop area-under-curve after switch,
   - recovery time,
   - false-positive rate,
   not just final average reward.

6. **Paper-code alignment:**
   Either implement the claimed regularizers and randomized switching or remove those claims.

If those are fixed and BAPR beats ESCP on post-switch regret or recovery time across 5–10 seeds, the paper becomes much more compelling even if final average return is similar.

================================================================================
Tokens: prompt=58485, completion=17936
