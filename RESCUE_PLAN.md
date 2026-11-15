# BAPR Rescue Plan — GPT-5.5 Top-3 Diffs

**Source**: GPT-5.5 round-2 review of full repo (`GPT55_REVIEW_v2.md`).
**Date**: 2026-04-26

## Status overview

| # | 改动 | 文件 | 预期 HC 提升 | 状态 |
|---|------|------|------------|------|
| 1 | 移除 oracle BOCD reset，传递 recent rollout 给 BAPR | `train.py` | **+200~800** | 🟡 部分完成（仅 reset 已移除）|
| 2 | BAPR surprise 用 recent_rollout 而非 replay；q_std 改单向 spike | `bapr.py` + `belief_tracker.py` | **+500~1500** | ⏳ 待实施 |
| 3 | 加 `sample_stacked_mixed`：BOCD λ_w 高时偏向 recent replay | `replay_buffer.py` + `train.py` | **+700~2000** | ⏳ 待实施 |

**总预期 HC 提升：+1400~4300**（BAPR 14327 → 15700~18600，可超过 ESCP 15958）

---

## Change 1 — Remove oracle BOCD reset + propagate recent rollout

### 致命性
**Reviewer-fatal bug**：训练循环用 `env.current_task_id`（隐藏信息）触发 `agent.reset_episode()` →
- 抹掉 paper 主张要解决的 changepoint detection 问题
- 第一次切换后的最强 surprise 信号被强制清零
- 任何审稿人看一眼代码必 reject

### 状态
- [x] **已移除 oracle reset**（commit `c65e224`，2026-04-26）
- [ ] **未实施**：让 `collect_samples` 返回 `recent_rollout` dict 并传给 `agent.multi_update`

### 完整 diff（剩余部分）

#### `jax_experiments/train.py` — `collect_samples` 返回值

**Before:**
```python
return ep_rewards
```

**After:**
```python
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

#### `jax_experiments/train.py` — caller

**Before:**
```python
ep_rewards = collect_samples(agent, env, replay_buffer, config, config.samples_per_iter)
```

**After:**
```python
ep_rewards, recent_rollout = collect_samples(
    agent, env, replay_buffer, config, config.samples_per_iter)
```

#### `jax_experiments/train.py` — BAPR dispatch

**Before:**
```python
metrics = agent.multi_update(
    stacked, current_iter=iteration,
    recent_rewards=ep_rewards if ep_rewards else None)
```

**After:**
```python
metrics = agent.multi_update(
    stacked,
    current_iter=iteration,
    recent_rewards=ep_rewards if ep_rewards else None,
    recent_rollout=recent_rollout,
)
```

### 预期 HC 影响
- **Direct training lift** from correct RMDM labels: **+200 ~ +800**
- **更重要**：移除一个 accept-blocking methodological flaw

---

## Change 2 — Compute BAPR surprise from latest rollout, not random replay

### 问题
当前 `bapr.py:multi_update` 从 replay buffer **随机采样的旧 batch** 计算 surprise，可能来自几百 iter 前的旧 regime。这跟 BAPR 的 changepoint detection 完全相悖。

### 完整 diff

#### `jax_experiments/algos/bapr.py` — surprise 计算

**Before:**
```python
# Q-std from critic (always from replay batch — evaluates uncertainty)
last_obs = jnp.array(stacked_batch["obs"][-1])
last_act = jnp.array(stacked_batch["act"][-1])
q_std = self._compute_q_stats(
    nnx.state(self.critic, nnx.Param),
    nnx.state(self.context_net, nnx.Param),
    last_obs, last_act, warmup)

# Reward signal: prefer actual rollout episode rewards
if recent_rewards is not None and len(recent_rewards) > 0:
    reward_signal = np.array(recent_rewards, dtype=np.float32)
else:
    reward_signal = stacked_batch["rew"][-1].flatten()

surprise = self.surprise_computer.compute(
    reward_signal, np.array([float(q_std)]))
self.belief_tracker.update(surprise)
```

**After:**
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

#### `jax_experiments/configs/default.py` — 加配置

```python
surprise_window: int = 1024
```

#### `jax_experiments/common/belief_tracker.py::SurpriseComputer.compute` — q_std 改单向 spike

当前 `q_std_ratio` 同时包含正向 spike 和负向（Q 收敛也算 surprise，错的）。改成单向。

**Before:**
```python
self.ema_q_std = self.ema_alpha * q_std_mean + (1 - self.ema_alpha) * self.ema_q_std
q_std_ratio = q_std_mean / max(self.ema_q_std, 1e-6)
...
surprise = 0.5 * reward_z + 0.3 * q_std_ratio + 0.2 * reg_div
```

**After:**
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

### 预期 HC 影响
**+500 ~ +1500**，主要通过减少 false λ spike 救起卡死 seed。

---

## Change 3 — Belief-aware recent replay sampling

### 动机
Paper 第 3.x 节明确声称 BAPR 解决 "regime staleness"，但代码里 replay buffer 是**纯均匀采样**——完全没有按 BOCD 信号偏向最近经验。这是 paper-code mismatch。

### 完整 diff

#### `jax_experiments/common/replay_buffer.py` — 新增 method

```python
def sample_stacked_mixed(self, n_batches: int, batch_size: int,
                         rng_key=None, recent_frac: float = 0.0,
                         recent_window: int = 50_000):
    """Uniform + recent replay mixture for post-changepoint adaptation.

    When BOCD detects a regime change (high λ_w), bias replay toward
    recent transitions to accelerate adaptation to the new regime.

    Args:
        recent_frac: fraction of batch from recent_window ([0, 0.9])
        recent_window: how many past samples count as 'recent'
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    recent_frac = float(np.clip(recent_frac, 0.0, 0.9))
    n_recent = int(batch_size * recent_frac)
    n_uniform = batch_size - n_recent

    k1, k2 = jax.random.split(rng_key)

    idx_uniform = jax.random.randint(
        k1, (n_batches, n_uniform), 0, self.size)

    if n_recent > 0:
        lo = max(0, self.size - recent_window)
        idx_recent = jax.random.randint(
            k2, (n_batches, n_recent), lo, self.size)
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

#### `jax_experiments/train.py` — 训练循环

**Before:**
```python
stacked = replay_buffer.sample_stacked(
    config.updates_per_iter, config.batch_size, rng_key=sample_key)
```

**After:**
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

#### `jax_experiments/configs/default.py`

```python
recent_replay_window: int = 50_000
```

### 预期 HC 影响
- **Abrupt-switch settings**: +700 ~ +2000
- **Smooth continuous gravity（当前 v10）**: +200 ~ +800
- 主要好处：post-switch recovery 加速 + 减少 stale buffer 污染

---

## 部署顺序建议

### Phase A（已部分完成）
- [x] 移除 oracle reset（commit `c65e224`）

### Phase B（v11/v12/v13 ablation 跑完后）
看完 P2/P3/P4 哪个对 variance 帮助最大，把 winner 与下面合并：
- [ ] Change 1 剩余部分：`recent_rollout` 传递
- [ ] Change 2：surprise from rollout + 单向 q_std spike
- [ ] Change 3：belief-aware replay sampling

### Phase C（论文实验重跑）
- 4 envs × 5 seeds × 4 algos = 80 runs，with all 3 changes
- 加 post-switch metrics（detection delay, recovery time, AUC）
- 加 CUSUM baseline 对照

---

## 风险评估

| 改动 | 实施风险 | 失败时回滚成本 |
|------|---------|---------------|
| Change 1 (rest) | 低 | 低（撤回 train.py）|
| Change 2 | 中 | 中（surprise 行为整体改变）|
| Change 3 | 中 | 低（保持 sample_stacked 老 path 兼容）|

---

## 关联资料

- [GOLDEN_BASELINE.md](GOLDEN_BASELINE.md) — v10 baseline 状态 + 4 个 "毒药" 改动
- [GPT55_REVIEW_v1.md](GPT55_REVIEW_v1.md) — Round 1 review（algorithmic ideas）
- [GPT55_REVIEW_v2.md](GPT55_REVIEW_v2.md) — Round 2 review（line-level bugs）
- 第三方对比基准（v10 当前结果）：HC iter ~67%
  - ESCP 15958 (avg 4 seeds)
  - RESAC 15390
  - BAPR 14327 ← target after rescue: ≥ ESCP
