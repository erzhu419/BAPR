下面是基于当前版本 `paper.tex` 的继续审稿意见。

与上一份 `GPT_review.md` 相比，这一版论文已经加入了实验、joint belief、per-transition belief ablation、Bad-BAPR 反例实验、超参敏感性和更完整的 limitation。因此这份 review 不再重复“没有实验”版本的通用意见，而是按当前稿件的真实状态，模拟 NeurIPS / ICLR / ICML reviewer 会如何继续卡点。

---

# Paper Summary (审稿人总结)

本文提出 BAPR，用于 piecewise-stationary continuous control。核心结构包括：

1. 使用 BOCD 维护 run-length belief，用于检测 regime switch；
2. 使用 RE-SAC 的 robust ensemble 框架作为稳定学习 backbone；
3. 使用 RMDM/context embedding 做 mode identification；
4. 使用 belief-derived scalar penalty 调节 LCB 系数 `beta_eff`；
5. 给出 frozen belief Bellman operator 的 contraction proof；
6. 给出 Q-dependent belief 破坏 contraction 的 Lean 4 机器验证反例；
7. 新增 MuJoCo / Brax style 非平稳实验，包括 HalfCheetah、Ant、Walker2d、Hopper。

当前版本比之前明显更完整，尤其是 Ant discrete-regime 上 BAPR 对 ESCP 的分离结果很强。但是论文现在暴露出新的核心问题：

> 实验结果显示真正起主要作用的是 RMDM/context conditioning，而不是 BOCD。

这会动摇论文标题和主贡献叙事，因为论文把 BOCD / Bayesian Amnesia 放在中心，但 ablation 里 `w/o BOCD` 只掉 1%，而 `w/o RMDM` 掉 55%。

---

# Strengths (优点)

## 1. 论文已经从纯理论稿变成完整系统稿

相比上一版，现在有：

* 主实验；
* discrete-regime benchmark；
* perturbation sweep；
* component ablation；
* penalty-scale sensitivity；
* Bad-BAPR 实证探针；
* negative result 讨论；
* limitation appendix。

这对 RL 顶会很重要。

当前版本已经不再是“没有实验必拒”的状态。

---

## 2. Ant discrete-regime 结果很有说服力

Ant 上：

* ESCP: `4,600 +- 4,406`
* BAPR no per-trans: `20,679 +- 3,402`
* 提升约 `+349%`
* BAPR 与 ESCP distribution zero-overlap

这是当前实验部分最强的结果。

如果这个结果能扩展到更多 seeds，并补足公平 baseline，这可以成为论文的 empirical anchor。

---

## 3. 诚实报告 negative results 是加分项

论文没有隐藏：

* Walker2d / Hopper 失败；
* per-transition belief storage 反而伤害性能；
* Bad-BAPR 没有发散，甚至在 HalfCheetah 上更好；
* BOCD 在 HalfCheetah 上不是主要贡献；
* sensitivity 还不完整。

这种诚实是优点。

但是注意：

> 诚实报告 negative results 加分，但不能让 negative results 反过来拆掉主 claim。

当前版本有这个风险。

---

## 4. Lean 4 formal verification 仍然是强 differentiator

Lean 证明包括：

* frozen belief contraction；
* post-update contraction；
* Q-dependent belief counterexample；
* sharp threshold；
* approximate contraction components；
* beta monotonicity；
* context embedding equivalence。

在 RL 论文中，这仍然很少见。

如果表述足够克制，它是强加分项。

---

## 5. Theory-practice gap 已经开始正面处理

当前版本加入了：

* mode vs run-length 的区分；
* tractable approximation remark；
* shared critic approximation；
* scope and limitations；
* function approximation error budget。

这比上一版更成熟。

---

# Major Weaknesses (主要问题)

下面是当前版本最容易被 reviewer 攻击的地方。

---

# 1. 最大问题：BOCD 叙事被自己的 ablation 削弱

论文标题和主线是：

> Bayesian Amnesic Piecewise-Robust RL

但是 Table ablation 给出的信息是：

* `BAPR w/o BOCD`: `-1%`
* `BAPR-FixedDecay`: `-14%`
* `BAPR w/o adaptive beta`: `-28%`
* `BAPR w/o RMDM`: `-55%`

这会让 reviewer 得出一个很直接的结论：

> The main performance gain appears to come from supervised/contextual mode representation rather than Bayesian online change detection.

这非常危险。

因为你的理论重点是 BOCD belief 和 frozen belief contraction，但实验告诉 reviewer：

> BOCD is almost decorative on the main ablation benchmark.

论文甚至用了 “decorative” 这个词，这在投稿版本里风险很高。

建议：

1. 不要在正文中用 “decorative” 这种自毁式措辞；
2. 增加一个 BOCD 真正强于 FixedDecay / no BOCD 的 benchmark；
3. 设计 recurring / irregular / unseen switch schedule，使 fixed decay 和 static beta 明显失败；
4. 如果做不到，就重构论文叙事：把贡献从“BOCD 是核心性能来源”改成“BOCD 是形式化安全调节层，RMDM 是主要性能来源”。

---

# 2. BOCD run-length 机制存在概念冲突

正文 Introduction 说：

> Short run-lengths indicate a recent regime change, triggering heightened caution; long run-lengths indicate stability.

这符合标准 BOCD 直觉。

但是 Method 中的 likelihood 是：

[
p(\xi | h) = N(0, sigma_0^2 + sigma_g h)
]

也就是：

* h 越大，variance 越大；
* high surprise 更容易被 large h 解释；
* high surprise 会把 posterior 推向 long run-length，而不是 short run-length。

正文还明确写：

> high surprise causes bar h to spike above baseline

于是当前机制变成：

> 不是 “change 后 run-length reset 到 0”，而是 “surprise 后 expected run-length 变大”。

这和 BOCD 的语义冲突。

更严重的是 Appendix 里又写：

> BOCD posterior shifts to short run-lengths within detection delay.

这与 Method 中 `bar h spike` 的描述矛盾。

Reviewer 可能会写：

> The BOCD posterior is used inconsistently: the paper alternates between interpreting surprise as evidence for short run-lengths and as evidence for long run-lengths.

这是当前稿件的 critical bug。

建议二选一：

## 方案 A：回到标准 BOCD

让 regime change 后 posterior mass 真的转移到 small run-length。

此时 conservatism 应该来自：

* `p(h = 0)`；
* `p(h <= h0)`；
* posterior entropy；
* expected inverse run-length；
* `1 - E[h]/H`。

而不是 `E[h]/H - EMA`。

## 方案 B：承认这不是标准 run-length BOCD

如果你想保留 “large h variance absorbs surprise, bar h spikes” 的机制，那就不要把它称为标准 change-point posterior。

可以改成：

> BOCD-inspired surprise memory / effective uncertainty window

然后弱化 detection-delay claim。

---

# 3. “without explicit labels” 仍然站不住

Abstract 说：

> RMDM enables mode identification without explicit labels.

但是 Method 写：

> During training in simulation, the PS-MDP structure provides mode identity `tau_id`; these labels are used for RMDM auxiliary loss.

Algorithm 也把 `tau_id` 存进 replay buffer。

这就是 explicit mode labels in training。

Reviewer 会认为：

> The claim of label-free mode identification is misleading.

建议改成：

> RMDM uses simulator-provided mode labels during training and requires no mode labels at deployment.

或者补一个真正 fully unsupervised variant：

* BOCD segmentation pseudo-label；
* k-means pseudo mode；
* no ground-truth `tau_id`；
* 和 supervised RMDM 对比。

否则 “without explicit labels” 必须删掉。

---

# 4. Formal convergence claim 有 overclaim 风险

Abstract 和 Introduction 说：

> formal piecewise convergence rate theorem, with every component machine-verified

Appendix 里确实有很多 Lean component theorem。

但是当前 piecewise convergence theorem 仍依赖：

* mode separability；
* metastable period；
* Lipschitz surprise；
* projected contraction；
* non-expansive projection；
* bounded stochastic noise；
* correct belief tracking；
* function approximation capacity；
* optimization actually tracking projected Bellman updates。

这些条件对 deep RL 来说很强。

而且 Appendix scope 自己也承认：

> it does not guarantee end-to-end convergence of the full training loop.

所以 reviewer 可能会认为主文 overclaims。

建议把表述改成：

> component-wise formal error budget

而不是：

> formal piecewise convergence rate of BAPR training

尤其不要写：

> no informal steps

因为实际组合从 tabular operator 到 implemented deep RL still contains informal modeling assumptions。

---

# 5. Theory-implementation gap 仍然没有完全闭合

理论对象是：

[
T_BAPR = sum_m rho(m) T_m
]

实现对象是：

* shared critic `Q(s, e, rho, mu, a)`；
* scalar `beta_eff`；
* no independent per-mode operators；
* no exact mixture backup。

论文用 context embedding equivalence 来补 gap：

> 如果 context embedding injective，则 shared critic 可表示 per-mode Q。

但这还不够。

原因：

1. `phi: S -> R^de` 未必能从 state 单独识别 mode；
2. 同一个 state 可能在不同 dynamics 下出现，mode 信息不一定在 instantaneous state 中；
3. RMDM 学到 separable embedding 不等于 Bellman operator 等价；
4. scalar `beta_eff` 不等于 mode-weighted epistemic penalty；
5. neural network 表达能力不等于训练会找到那个解；
6. contraction proof 不覆盖 actor-critic joint optimization。

建议：

* 主文说清楚 guarantee applies to the abstract operator；
* implementation inherits only the frozen-parameter stability intuition；
* 不要写 “applies exactly to implemented algorithm”，除非真的实现 per-mode operators 或 oracle context。

---

# 6. 实验统计强度不足

当前实验已经有了，但统计上还不够顶会。

具体问题：

## Main table

Experiment setup 说：

> results are averaged over 10 random seeds

但 Table main results 只有单个数字，没有 `mean +- std` 或 CI。

而且 baselines list 包含 SAC，但 main table 只有 RE-SAC / ESCP / BAPR，没有 SAC。

## Discrete benchmark

* HalfCheetah: N=5，可以接受；
* Ant: N=3，偏少但有希望；
* Hopper: N=1；
* Walker2d: N=1。

N=1 不能支撑 method comparison。

## Ablation

Q2 ablation 是 single seed snapshot，并且有的 run 到 1499，有的 run 到 1000。

这会被 reviewer 认为不可靠。

## Sensitivity

Penalty scale sweep：

* 只有 seed 0；
* 只看 iter 300；
* 只 sweep `c_penalty`；
* 其他关键超参全部 deferred。

这不够支撑 Q5。

建议：

1. main result 所有表格统一 `mean +- std`；
2. ablation 至少 HalfCheetah + Ant，各 3 seeds；
3. sensitivity 至少 `c_penalty` 和 hazard rate 各 3 seeds；
4. Bad-BAPR 至少构造一个超过 threshold 的环境；
5. 删掉或降级 N=1 的结论。

---

# 7. Q3 / Q4 目前没有真正回答

Experiments 开头列出：

* Q3: How quickly does BAPR detect and adapt?
* Q4: Does adaptive conservatism behave as theoretically predicted?

但是 Q3/Q4 section 最后说：

> rendered visualization is deferred to the companion technical report.

这在投稿论文里很危险。

如果 Q3/Q4 是论文核心问题，就不能 deferred。

必须放进主文或 appendix：

* true mode timeline；
* BOCD posterior heatmap；
* `lambda_w` / `beta_eff` curve；
* switch-aligned average detection delay；
* false-positive rate；
* recovery time after switch。

这张图是 BAPR 的 money plot。

没有它，Bayesian Amnesia 的主张不够可见。

---

# 8. Bad-BAPR 实验现在反而削弱理论故事

论文证明：

> Q-dependent belief can break contraction.

但是实验写：

> Bad-BAPR does not diverge and exceeds BAPR by +18%.

这可以诚实报告，但当前写法会让 reviewer 产生疑问：

> If the bad version performs better on the tested benchmark, why should we use the frozen version?

你解释说 HalfCheetah below threshold，这在理论上合理。

但如果这样，必须补一个 above-threshold empirical counterexample。

建议构造：

* catastrophic wrong-regime penalty；
* reward gap deliberately large；
* high gamma；
* Q-dependent belief sensitivity controlled；
* plot Q-loss / Q-norm / return collapse。

否则 Bad-BAPR section 应该降级到 appendix，并把它表述为：

> The counterexample is a worst-case design boundary, not an empirical claim for all benchmarks.

---

# 9. Baseline fairness 还不够

当前 baselines：

* SAC；
* RE-SAC；
* ESCP；
* BAPR。

但是实际主表没有 SAC。

另外，piecewise-stationary RL reviewer 可能会要求：

* sliding-window SAC；
* reset-on-change SAC；
* replay buffer flush baseline；
* recurrent SAC / history-conditioned SAC；
* PEARL / VariBAD style latent context；
* oracle mode-conditioned SAC；
* FixedDecay；
* supervised ESCP with same mode labels。

尤其是因为 BAPR 的 RMDM 使用 simulator mode labels，必须有公平对照：

> ESCP or SAC + same supervised RMDM labels

否则 reviewer 会说 BAPR 的优势可能来自 label supervision，而不是 BOCD。

---

# 10. Hyperparameter inconsistency

当前稿件中 `c_penalty` 有明显不一致：

* discrete benchmark 使用 `c_penalty = 0.5`；
* sensitivity sweep 说 `0.5` 是 sweet spot；
* appendix hyperparameter table 默认值写 `c_penalty = 5.0`；
* action-ranking appendix 又用 typical `c_penalty = 5`；
* Q3/Q4 讨论里又出现 `c_penalty = 2` 时 `beta_eff` as low as `-3`。

这会让 reviewer 怀疑实验设置和理论讨论没有统一。

建议：

* 如果主实验用 0.5，appendix default 也改成 0.5；
* 旧的 `5.0` 全部改为 historical value 或删掉；
* 每张表 caption 写清楚 beta_base、c_penalty、hazard、Hmax。

---

# Minor Weaknesses (次要问题)

## 1. 语言有些过于自我削弱

建议删除或改写：

* “decorative”
* “Contrary to the textbook reading”
* “deferred to companion technical report”
* “we expect it to survive seed averaging”

这些措辞在 review 中会被抓住。

更稳妥的写法：

> Single-seed ablations suggest RMDM is the dominant contributor on HalfCheetah; multi-seed confirmation is ongoing.

---

## 2. Walker2d / Hopper failure 讨论太长

负结果可以保留，但当前篇幅过大，会稀释主线。

建议：

* 主文只保留一段；
* sweep table 放 appendix；
* 不要把失败环境解释成过多 narrative；
* 强调 “we report but do not use these as evidence for superiority”。

---

## 3. `<5% overhead` 需要证据

当前说 BAPR 比 RE-SAC 慢 `<5%`。

这需要：

* wall-clock table；
* GPU type；
* batch size；
* env steps/sec；
* update steps/sec。

否则删掉或改成 qualitative statement。

---

## 4. PDF 还有排版 warning

当前 `paper.log` 显示多个 Overfull hbox，最大约 `87.7pt`。

这不是学术大问题，但投稿前需要处理，尤其是宽表格和 Lean theorem table。

---

# Questions for Authors (审稿人问题)

1. Why does removing BOCD only reduce performance by 1% on HalfCheetah if BOCD is central to BAPR?
2. Does the current BOCD posterior actually shift to short run-lengths after a change, or does the expected run-length spike upward?
3. Is RMDM trained with ground-truth simulator mode labels? If yes, why does the abstract claim mode identification without explicit labels?
4. What is the correct default `c_penalty`: 0.5 or 5.0?
5. Why is SAC absent from the main results table despite being listed as a baseline?
6. Are all baselines given the same mode labels or context supervision as BAPR?
7. Can BAPR outperform FixedDecay on a benchmark where switches recur unpredictably?
8. Can the Bad-BAPR counterexample be reproduced empirically in a deliberately above-threshold environment?
9. What are the detection delay, false-positive rate, and recovery time measured from actual runs?
10. Does the contraction theorem apply to the implemented shared critic, or only to the abstract mode-mixture operator?

---

# Required Revisions (必须修改)

如果目标是 NeurIPS / ICLR / ICML，建议按优先级处理。

## P0: 必须修

### 1. 修复 BOCD 语义冲突

统一以下三件事：

* run-length posterior 的数学定义；
* high surprise 后 posterior 如何变化；
* `lambda_w` 为什么表示 recent change。

当前 “short run-length means change” 和 “bar h spike means change” 不能同时成立。

---

### 2. 删除或改写 “without explicit labels”

改为：

> RMDM uses simulator-provided mode labels during training but requires no mode labels at deployment.

如果想保留无标签 claim，必须补 unsupervised pseudo-label 实验。

---

### 3. 重新定位主贡献

当前证据更支持：

> BAPR = RMDM-driven mode-conditioned robust RL + optional BOCD safety modulation

而不是：

> BOCD is the main performance driver.

除非补一个 BOCD 很强的实验。

---

### 4. 补 Q3/Q4 money plot

至少一张图：

* true mode；
* BOCD posterior heatmap；
* `beta_eff`；
* return/recovery curve。

最好再给 detection delay table。

---

### 5. 统一实验统计

至少：

* main table 全部 mean +- std；
* ablation 多 seed；
* sensitivity 多 seed；
* 删除或降级 N=1 结论。

---

## P1: 强烈建议

### 6. 增加公平 baseline

建议加入：

* FixedDecay multi-seed；
* replay-buffer reset / flush；
* sliding-window SAC；
* recurrent SAC；
* oracle mode-conditioned SAC；
* ESCP + same supervised RMDM labels。

---

### 7. 补 above-threshold Bad-BAPR

把 Lean counterexample 和实证真正连接起来。

否则当前 Bad-BAPR 结果会被认为是：

> The supposedly bad design works better empirically.

---

### 8. 降低 formal claim 强度

把：

> formal piecewise convergence rate theorem

改成：

> component-wise formal error budget for the frozen abstract operator

这样更安全，也更符合 appendix scope。

---

## P2: 投稿前打磨

### 9. 清理措辞

把过于口语或自我削弱的句子压缩。

### 10. 修 PDF overfull

特别是长表格、Lean theorem catalog、实验表格。

---

# Final Score (模拟顶会评分)

按当前版本，如果我是 NeurIPS / ICLR reviewer：

## Novelty

7 / 10

BOCD + robust ensemble + formal verification 的组合仍然有新意。

## Technical Quality

6.5 / 10

Lean proof 强，但 BOCD 语义、formal overclaim、theory-implementation gap 仍然明显。

## Empirical Quality

5.5 / 10

Ant 结果强，但 ablation 单 seed、Q3/Q4 缺图、BOCD 贡献不清。

## Clarity

6 / 10

结构完整，但当前叙事有冲突：BOCD 是中心，实验却说 RMDM 是中心。

## Significance

7 / 10

问题重要，系统有潜力。

---

# Overall Recommendation

当前我会给：

**6 / 10: Weak Reject / Borderline Reject**

主要原因不是“工作不够好”，而是：

1. BOCD 机制解释有内在冲突；
2. 实验没有证明 BOCD 是主贡献；
3. label-free claim 不准确；
4. ablation 和 sensitivity 统计强度不足；
5. formal claim 仍有 overclaim 风险。

如果修掉 P0，并补出一个 BOCD 明显优于 FixedDecay / no BOCD 的 benchmark，我会倾向：

**7 / 10: Weak Accept**

如果再补多 seed Ant / HalfCheetah、真实 BOCD heatmap、above-threshold Bad-BAPR 实证反例，并把 theoretical claim 表述得更克制，这篇可以冲：

**8 / 10: Accept**

---

# 最重要的修改建议一句话

当前版本最应该做的不是继续加理论，而是重新对齐三件事：

> 论文声称 BOCD 是 Bayesian Amnesia 的核心；数学上 BOCD posterior 的语义必须自洽；实验上 BOCD 必须在至少一个关键场景中不可替代。

只要这三件事对齐，BAPR 的论文形态会稳很多。
