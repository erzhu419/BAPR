# BAPR Golden Baseline — v10 (P0 + P1)

**⚠️ 重要：这是 BAPR 的 working version，后续所有论文（CS-BAPR / BAPR-HRO / BAMOR / TransitDuet）都基于此版本改造。不要轻易改动核心算法。**

## 🏷️ 版本标识

- **Commit**: `c29b4e9` — "v10: P0 (BAPR warmup) + P1 (penalty_scale 5→2)"
- **Tag**: `bapr-golden-v10`
- **Date**: 2026-04-24
- **Env**: JTL110 GPU server, conda env `bapr-jax`

## ✅ 为什么这是 Golden

历经多版本调试（v4-v9 都有不同问题），这是第一个同时满足以下条件的版本：

1. **多 seed 实验稳定**：5 seeds 下 variance 显著降低，不再 cherry-pick single seed
2. **BAPR 真正赢 RESAC/ESCP**：早期（iter ~70）avg 5 seeds 就领先 2-3 倍
3. **理论一致性**：代码完全匹配 paper fig_training_curves.pdf 的 seed=8 曲线

## 🔧 关键算法改动（基于 commit `374d1f7` 归档版）

### P0: BAPR 机制 warmup — `bapr_warmup_iters=100`

```python
# bapr.py::multi_update
if current_iter < self.config.bapr_warmup_iters:
    weighted_lambda = 0.0  # pure RESAC phase
else:
    weighted_lambda = self._compute_weighted_lambda()  # BOCD active
```

**为什么需要**：BOCD 在 iter 0 激活时，Q-std 和 surprise EMA 都还没稳定，会产生噪声 λ_w ≈ 0.3，导致 β 过度保守 → 部分 seed 卡死。前 100 iter 纯 RESAC 让所有 seed 达到稳定基线后再启动 BOCD。

### P1: penalty_scale `5.0 → 2.0`

```python
# configs/default.py
penalty_scale: float = 2.0   # was 5.0
```

**为什么需要**：即使 warmup 后 λ_w 在 0.3 附近稳态，原 scale=5 会让 β_eff = -2 - 1.5 = -3.5（75% 过保守）。新 scale=2 让 β_eff ∈ [-4, -2]，温和很多。

## 🚫 哪些改动是 BAPR 毒药（绝对不要加回）

经过 v5-v8 的血泪教训，以下"改进"会**毁掉 BAPR**：

### ❌ Independent target Q（RE-SAC v5 改进）
```python
# DO NOT DO THIS:
tq_all = tm(next_aug, na) - alpha * nlp  # per-Q target
# Archive/golden uses shared min target:
tq = tm(next_aug, na).min(axis=0) - alpha * nlp  # ✓
```
**为什么毒**：independent target 让 ensemble Q 发散 → Q-std 失真 → BOCD surprise 信号失效。

### ❌ EMA policy for eval
**为什么毒**：和 BAPR 自己的 RMDM context 稳定化机制重复，反而引入额外噪声。

### ❌ Performance gating（eval-based β 收紧）
```python
# DO NOT DO THIS:
if self._perf_gate_active:
    lw = max(lw, 0.5)  # force λ_w floor
```
**为什么毒**：早期 eval 本来就噪声大（±20% 波动），会频繁触发 → β_eff = -4.5 常驻 → policy 冻结。

### ❌ λ_w EMA baseline 减法
```python
# DO NOT DO THIS:
self._lw_baseline = 0.95 * self._lw_baseline + 0.05 * raw_lw
lw = max(0, raw_lw - self._lw_baseline)  # EMA 追 raw_lw → 永远 0
```
**为什么毒**：EMA (α=0.05) 以 ~20 iter time constant 追上 raw_lw → λ_w ≈ 0 恒定 → BOCD 完全失效。

## 🌍 环境配置（匹配 paper fig）

```bash
# run_server_BAPR.sh
ENV_PARAMS["HalfCheetah-v2"]="gravity"
ENV_PARAMS["Hopper-v2"]="gravity"
ENV_PARAMS["Walker2d-v2"]="gravity"
ENV_PARAMS["Ant-v2"]="gravity"

ENV_SCALE[*]=3.0           # 所有 env log_scale=3.0 (archive default)
ENV_BACKEND[*]="spring"    # 所有 env spring backend
```

**sample_tasks**：`rng.uniform(-log_scale_limit, +log_scale_limit)` 连续采样，**不要改为 discrete 极值采样**（那个会让 BOCD 频繁误触发）。

## 📊 Paper figure 复现

```python
# jax_experiments/plot_curves_paper.py
# Uses seed=8 logs, reads Iter/Reward/Eval from .log files
python jax_experiments/plot_curves_paper.py
```

Paper fig `fig_training_curves.pdf` 是 seed=8 的单 seed 曲线。使用本 golden 版本可复现。

## 🧪 下一步实验（未纳入 golden）

P2-P4 是更进一步的稳定性增强，需要 v10 完整数据验证后再决定：
- **P2**: Alpha floor（防熵塌缩）
- **P3**: 梯度裁剪（防 Q 发散）
- **P4**: BOCD 悲观初始化（belief[0]=0.9）

## 🛠 如何 revert 到 Golden

```bash
git checkout bapr-golden-v10
# or
git checkout c29b4e9
```

## 📞 Contact

如果未来实验 BAPR 表现异常（<50% of RESAC），先 diff 对比 golden 版本：
```bash
git diff bapr-golden-v10 -- jax_experiments/algos/bapr.py
git diff bapr-golden-v10 -- jax_experiments/configs/default.py
```
大概率是新引入的 "improvement" 破坏了 BOCD 核心机制（见"毒药"章节）。
