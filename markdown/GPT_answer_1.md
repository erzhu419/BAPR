# 我：
最新push了一下，但是有几个问题：
我们复现/扩展 RE-SAC/ESCP/BAPR 时，结果出现反直觉现象：论文里 RE-SAC 和 ESCP 都应当比 SAC 强，但我们当前 MuJoCo nonstationary benchmark 里 SAC 反而明显更强，Ant 上 SAC 约 8917±4688，而 ESCP/RE-SAC 只有约 2771/2818，BAPR v78/v79/v80 也只有约 1.7k-2.2k。这不符合 RE-SAC/ESCP 原论文趋势。

当前 BAPR 训练配置大致是：
backend=spring, env_type=discrete_mode, mean_dwell_iters=60, max_iters=800,
weight_reg=0.003, beta_ood=0.003, critic_target_mode=independent。
最近 v80 residual gate 也失败了，gate 基本是常数，不是真自适应。

请重点帮我排查：这是不是环境/评估协议没对齐？
需要比对 RE-SAC/ESCP 原论文的：
1. nonstationarity 定义、mode 切换、gravity/friction/身体参数扰动范围；
2. SAC/ESCP/RE-SAC 是否都在同一 nonstationary env 下训练和评估；
3. eval 是 deterministic 还是 stochastic，ID/OOD 怎么定义；
4. episode horizon、reward scale、termination、action repeat、observation/action normalization；
5. 训练步数和 update/rollout 比例；
6. 当前 SAC 是否其实跑在更容易的 station

# GPT：
结论先说：**是的，首要嫌疑不是 v80 gate，而是环境/评估协议和 baseline 定义没有对齐。** 现在这个结果不能直接解释成“BAPR/RE-SAC/ESCP 不如 SAC”，更像是：**你现在测到的是一个 Brax-spring、K=4 discrete-mode、stationary-per-mode deterministic eval 下的 SAC-ensemble baseline 很强；它和 ESCP/RE-SAC 原论文要比较的协议不是同一个问题。**

而且我看到几个 P0 级问题，任何一个都足以让 SAC 看起来反直觉地强。

## 1. 当前 eval 不是 nonstationary online adaptation eval

这是最关键的问题。

`evaluate()` 在 discrete-mode 下的逻辑是：如果 `tasks <= 10`，就逐个 task/mode 调 `env.set_task(task)`，每个 mode 跑 deterministic rollout，然后把所有 mode 的回报平均。
而 `eval_rollout()` 明确写着：**does NOT update step counter or switch task**。也就是说 eval 期间不会发生 regime switch。

所以你现在的主 eval 其实是：

> 训练在 piecewise nonstationary 环境里，评估在每个固定 mode 上做 stationary deterministic rollout，然后平均。

这不等价于 ESCP/BAPR 想证明的“突然变化后在线识别/恢复”。在这个协议下，SAC 只要学到一个跨 mode 还不错的平均策略，就会显得很强；ESCP/BAPR 的检测、recent replay、residual gate 反而不一定有用。

**这也解释了 v80 residual gate 为什么像常数：当前 eval 不考 switch recovery，gate 没有真正被“切换后的恢复收益”监督。**

## 2. 你现在的 discrete-mode 环境不是 ESCP 原论文协议

当前 `discrete_mode_env.py` 是你们自己为 BAPR “sweet spot” 设计的 K=4 semantic modes：例如 Ant 是 `normal`、`heavy_low_g`、`light_high_g`、`stiff_joints`，对应 body mass、gravity、dof damping 的乘法扰动。
这个 env 还使用 exponential dwell，默认 mean dwell 是 60 iters，每 iter 4000 steps，也就是平均 240k env steps 才切一次 mode。

而 `Config` 里也明确区分了两套环境：`continuous` 被注释为 “original ESCP-style continuous task family”，`discrete_mode` 被注释为 “K=4 semantic modes, exp dwell — BAPR sweet spot”。
`run_seed.sh` 现在无条件给所有算法传 `--env_type discrete_mode --mean_dwell_iters "$DWELL"`。

所以如果你现在拿这个结果去对照“ESCP 原论文趋势”，会有严重协议错位：**现在不是原 ESCP continuous sudden-change protocol，而是你们自定义的 BAPR discrete-mode protocol。**

## 3. 当前 “SAC” 也不是标准 vanilla SAC

这点很容易被忽略。`SACBase` 的 docstring 写的是 “Vanilla SAC with ensemble critic”，但实际它用了 `EnsembleCritic(... ensemble_size=config.ensemble_size)`，默认 `ensemble_size=10`。
它的 critic target 用的是 10 个 Q 的 `min(axis=0)`，policy loss 用的是 ensemble mean。

这不是通常论文里的 SAC baseline，通常至少应明确叫 **SAC-ensemble / SAC-10Q**。在你的当前环境里，它可能比 RE-SAC/ESCP 更稳，因为：

SAC-10Q target 用 10Q min，足够保守；policy 又用 mean，不像 RE-SAC/ESCP 那样用 `beta=-2` 的 LCB actor。RE-SAC/ESCP 的 actor 更悲观，在 Ant 这种环境下很可能 under-train。

所以现在看到 “SAC 8917，RE-SAC/ESCP 2771/2818”，第一反应不应该是“论文趋势错了”，而是：

> baseline name 不等价。你现在的 SAC 是 10-critic ensemble SAC，不是 vanilla SAC；而 RE-SAC/ESCP 加了更强 pessimism，可能在当前 stationary-per-mode eval 中被惩罚。

## 4. RE-SAC 原论文优势本来不是 MuJoCo sudden-change 结论

你引用的 RE-SAC 原论文是 bus fleet control。arXiv 摘要说的是在 realistic bidirectional bus corridor simulation 里，RE-SAC 比 vanilla SAC 有更高 cumulative reward，并降低 rare OOD states 的 oracle Q-value estimation error。([arXiv][1])
这不是一个“RE-SAC 在 MuJoCo nonstationary gravity/friction benchmark 必然强于 SAC”的原始结论。

所以 RE-SAC 这部分需要拆开讲：

**RE-SAC 在 bus stochastic/rare-state setting 下强于 SAC，不自动推出它在 Brax Ant K=4 mode switching 下强于 SAC-10Q。**
MuJoCo/Brax 迁移需要重新定义协议，不能直接借原论文趋势做 sanity check。

## 5. ESCP 实现可能也不是原 ESCP 的完整环境 probe 协议

你当前 `ContextNetwork` 是一个 MLP，输入是当前 observation，输出 L2-normalized context embedding。
`ESCP` 里也是 `ep = context_net(obs)`，RMDM loss 用 `task_ids` 做 within-task consistency / cross-task diversity。

这在工程上可以跑，但它和“通过 recent transitions / environment probe 识别 dynamics”的 ESCP 原意可能不完全等价。BAPR 论文里对 ESCP 的描述是：ESCP 训练 environment probe 产生 context vector，使 policy 区分不同 dynamics；而 BAPR 扩展这个 context-conditioning。([arXiv][2])
如果原 ESCP 的 probe 依赖一段交互轨迹或 transition history，而你这里只用当前 obs，那么在 dynamics 参数不直接由单帧状态可观测时，context 可能学不到真正 mode。

这会让 ESCP 在你的 benchmark 上变成：**一个带 task-ID supervised representation loss 的 observation encoder**，而不是完整的 online dynamics inference method。

## 6. 当前 eval horizon/episode 处理有一个非常危险的偏差

`max_episode_steps=1000` 在 config 里定义了。
但是 `eval_rollout()` 本身只根据 env 的 `done` auto-reset，不按 1000 step 强制截断。`evaluate()` 设 `n_steps = n_episodes * max_episode_steps`，然后只有在 `done` 时才记录一个 episode；如果一直不 done，就把整个 5000 steps 累加成一个 episode return。

这会造成严重偏差：

如果 SAC 在 Ant 上 5000 steps 都不死，它的 eval return 是一个 5000-step trajectory 的总和；
如果 ESCP/RE-SAC 中间摔倒并触发 done，它就变成若干短 episode 的平均。
这不是标准 “5 episodes × 1000 horizon” evaluation。

这点非常可能直接放大 SAC 的优势。你必须先把 eval 改成严格 horizon-based：

```python
for task in eval_tasks:
    for ep in range(n_episodes):
        reset env
        ep_r = 0
        for t in range(max_episode_steps):
            action = deterministic_policy(obs)
            obs, reward, done, info = env.step(action)
            ep_r += reward
            if done:
                break
        all_rewards.append(ep_r)
```

先用这个 sequential eval 做 audit，不要一开始就追求 scan-fused。等指标对了再优化。

## 7. SAC 是否其实跑在更容易的 stationary env？

如果所有结果都是通过当前 `run_seed.sh` 跑的，那训练环境本身不是 stationary：脚本对 `sac|escp|resac|bapr` 都传了 `--env_type discrete_mode`。
但有三个 caveat：

第一，**eval 是 stationary per mode**，所以 SAC 的高分很可能来自更容易的 stationary eval，而不是在线切换恢复。

第二，训练时 scan rollout 也写明了 “sys is fixed for the entire 4000-step scan”，task switching happens at iter boundary。
再叠加 mean dwell 60 iters，SAC 实际看到的是很长的 stationary blocks，而不是频繁 sudden changes。

第三，`run_seed.sh` 总是加 `--resume`，而 run name 只有 `TAG_ALGO_ENV_dwDWELL_sSEED`，不包含 backend、env_type、mode_variant、weight_reg、beta_ood、actor objective 等关键信息。
`train.py` 只要 `resume=True` 且 checkpoint 存在就会直接 load。

所以必须排查：**SAC 是否 resume 了旧 checkpoint，尤其是旧的 stationary/easier/milder config checkpoint。**
这是我建议你马上查的第一件事。

## 8. 还有一个配置陷阱：RE-SAC/ESCP 可能没用你以为的 `0.003`

`run_seed.sh` 只在 BAPR-like 算法分支里把 `BAPR_WEIGHT_REG`、`BAPR_BETA_OOD` 映射成 CLI flags。对 `escp|resac|sac` 分支，`EXTRA=""`。

这意味着：如果你是通过环境变量 `BAPR_WEIGHT_REG=0.003 BAPR_BETA_OOD=0.003` 跑 `resac`，它**不会生效**，除非你把 `--weight_reg 0.003 --beta_ood 0.003` 放在额外 CLI args 里。默认 config 是 `weight_reg=0.01, beta_ood=0.01`。

而且当前 `ESCP` 代码里 critic loss 没有 `weight_reg` 和 `beta_ood`，它更像 “context + ensemble LCB”，不是 “RE-SAC + context”。
所以你现在表格里的 “RE-SAC/ESCP/BAPR 都是 weight_reg=0.003, beta_ood=0.003” 很可能不成立。需要从每个 run 的实际 command/log/config dump 验证。

## 我建议的排查顺序

**先不要再改 v81/v82。先做 protocol audit。**

### A. 立刻加一个 `protocol_signature.json`

每个 run 开始时保存：

```json
{
  "algo": "sac",
  "env_name": "Ant-v2",
  "backend": "spring",
  "env_type": "discrete_mode",
  "mode_variant": "orig",
  "mode_definitions": "...",
  "mean_dwell_iters": 60,
  "dwell_distribution": "exponential",
  "max_iters": 800,
  "samples_per_iter": 4000,
  "updates_per_iter": 250,
  "ensemble_size": 10,
  "beta": -2.0,
  "weight_reg": 0.003,
  "beta_ood": 0.003,
  "critic_target_mode": "independent",
  "resume": false,
  "checkpoint_loaded": false,
  "eval_protocol": "stationary_per_mode",
  "eval_deterministic": true,
  "eval_horizon_enforced": true
}
```

尤其要记录 `checkpoint_loaded`。目前 `--resume` 太危险。

### B. 先把 eval 拆成三个指标

不要只报一个 `eval_reward`。

| 指标                      | 目的                                    |
| ----------------------- | ------------------------------------- |
| `eval_stationary_id`    | 每个训练 mode 固定评估，检查单 mode 能力            |
| `eval_stationary_ood`   | held-out 参数或 held-out mode，检查泛化       |
| `eval_switching_online` | eval 时真的切 mode，检查 adaptation/recovery |

现在你的 `eval_reward` 只接近第一类，不是第三类。

### C. 修复 horizon

所有算法都用严格 `n_episodes × max_episode_steps`，不允许“没 done 就累计 5000 steps 当一个 episode”。修完后重新看 Ant SAC 是否还在 8917。

### D. 禁用 resume，换唯一 run_name

至少做一轮：

```bash
rm -rf jax_experiments/results_paper/<TAG>_sac_Ant_dw60_s*
```

或者临时从 `run_seed.sh` 去掉 `--resume`。同时 run_name 加上：

```text
envtype_discrete_backend_spring_variant_orig_iters800_wr0003_bood0003_noresume
```

否则你无法排除旧 checkpoint 污染。

### E. 做 2×2 sanity matrix

只跑 Ant seeds 0–2 就够：

| Train                  | Eval                 | 预期解释                          |
| ---------------------- | -------------------- | ----------------------------- |
| stationary single mode | same stationary mode | SAC 应该强，这是正常                  |
| discrete switching     | stationary per mode  | 可能 SAC 强，说明平均策略有效             |
| discrete switching     | switching online     | ESCP/BAPR 才应该有机会体现 adaptation |
| continuous ESCP-style  | original-style eval  | 才能对 ESCP 原论文趋势做 sanity check  |

如果 SAC 只在 stationary-per-mode eval 强，而在 switching-online eval 掉得厉害，那问题就是 eval 协议。
如果 SAC 在 switching-online 也强，那说明 discrete-mode Ant 本身可以被单一 robust policy 解决，BAPR 机制没有被需要。

## 对你列的 6 个问题逐项回答

**1. nonstationarity 定义/扰动范围：没对齐。**
当前是 BAPR 自定义 K=4 semantic modes，gravity/body_mass/dof_damping 乘法扰动，exponential dwell。不是原 ESCP continuous sudden-change protocol。

**2. SAC/ESCP/RE-SAC 是否同一 env：训练大概率同一 env，但 eval 不是 nonstationary。**
`run_seed.sh` 会给所有算法传 discrete_mode；但 eval 是 fixed-mode deterministic rollout，不切换。

**3. eval deterministic / ID-OOD：deterministic；discrete-mode 下没有真正 OOD。**
`evaluate()` 使用 deterministic policy；discrete-mode 的 `sample_tasks()` 忽略 n，只返回同一组 K modes，所以 train/test tasks 本质上不是独立 OOD split。

**4. horizon/reward/termination/action repeat/normalization：当前不标准。**
Brax-spring 不是 Gym MuJoCo；reward/termination 是你自己在 `_build_core_fns` 里重构的；eval 没强制 1000-step horizon；没看到 observation/action normalization。

**5. 训练步数/update 比例：当前是 800×4000=3.2M env steps，250 updates/iter=200k updates。**
update/env-step ratio 约 0.0625，不是常见的 1 update per env step。这个对所有算法都一样，但对需要 context/RMDM 的方法可能更不利。配置默认写明 `samples_per_iter=4000`、`updates_per_iter=250`。

**6. SAC 是否跑在更容易的 stationary env：训练未必，但 eval 基本是 stationary；还必须排除 resume 污染。**
从脚本看 SAC 训练也会用 discrete_mode；但 eval 固定 mode、不切换，而且 `--resume` 可能加载旧 checkpoint。

## 最短可执行结论

现在不要继续调 v80 gate。下一步应该是：

1. 去掉 `--resume`，清空旧 SAC/ESCP/RE-SAC/BAPR checkpoints。
2. 把 `evaluate()` 改成严格 5 episodes × 1000 steps。
3. 新增 `eval_switching_online`，eval 时真的按 dwell/switch schedule 切换。
4. 把当前 “SAC” 改名成 `sac_ensemble10`，另跑一个真正 vanilla SAC：`ensemble_size=2` 或按原 SAC implementation。
5. 对 RE-SAC/ESCP 打印完整 config，确认 `weight_reg/beta_ood` 是否真的生效。
6. 复现 ESCP 原协议时用 `env_type=continuous`，不要用当前 `discrete_mode`。

只有这六步做完，才能判断“BAPR 机制是否失败”。现在这个结果更像是**benchmark/eval/baseline definition 不一致**，不是算法创新已经被否。

[1]: https://arxiv.org/abs/2603.18396 "[2603.18396] RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach"
[2]: https://arxiv.org/abs/2605.16170 "BAPR: Bayesian amnesic piecewise-robust reinforcement learning for non-stationary continuous control"
