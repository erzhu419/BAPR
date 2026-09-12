# BAPR-v3 Empirical Variance and Frozen Teacher Result

Date: 2026-07-15

## Execution audit

Scheduler tasks `t33157-t33168` all completed normally. Every run contains:

- iterations `0-1399` and `5.6M` environment steps;
- 70 finite evaluations for each robust/oracle/learned static and switching
  ladder entry;
- the requested fixed 500-step stochastic-mode dwell;
- `instant_classifier_weight=0` and
  `freeze_teacher_after_teacher=true`;
- no resumed legacy checkpoint.

Only `t33159`, which ran on local, left a checkpoint locally. No remote
checkpoint PKL was downloaded for this analysis.

## Final paired results

Values are the mean of the last five evaluations. Each cell is
`static / switching`. Comparisons between robust, oracle, and learned within a
row share one controller checkpoint; comparisons across rows do not.

| Variant | Family | Env | Robust | Oracle | Learned | Posterior accuracy | Mean gate |
|---|---|---|---:|---:|---:|---:|---:|
| empirical | deterministic mean | Ant | `1053 / 1306` | `1583 / 2427` | `1992 / 2629` | `0.572` | `0.639` |
| empirical | deterministic mean | HalfCheetah | `2505 / 2603` | `3734 / 3950` | `3175 / 2931` | `0.480` | `0.542` |
| empirical | mean plus variance | Ant | `1602 / 2381` | `1495 / 2729` | `1757 / 2947` | `0.599` | `0.658` |
| empirical | mean plus variance | HalfCheetah | `1637 / 1690` | `1788 / 1834` | `1774 / 1596` | `0.451` | `0.493` |
| empirical | variance only | Ant | `2182 / 2801` | `947 / 1929` | `1960 / 2865` | `0.305` | `0.399` |
| empirical | variance only | HalfCheetah | `2559 / 2246` | `2549 / 2194` | `2558 / 2542` | `0.254` | `0.290` |
| bounded NLL | deterministic mean | Ant | `2385 / 2482` | `-216 / -324` | `2422 / 2603` | `0.631` | `0.174` |
| bounded NLL | deterministic mean | HalfCheetah | `1823 / 2017` | `2537 / 2496` | `1782 / 1930` | `0.411` | `0.009` |
| bounded NLL | mean plus variance | Ant | `2092 / 2878` | `2237 / 2704` | `2142 / 3009` | `0.634` | `0.078` |
| bounded NLL | mean plus variance | HalfCheetah | `2287 / 2210` | `2900 / 2708` | `2399 / 2345` | `0.424` | `0.007` |
| bounded NLL | variance only | Ant | `1497 / 2473` | `1254 / 2782` | `1625 / 2564` | `0.335` | `0.046` |
| bounded NLL | variance only | HalfCheetah | `3029 / 2932` | `3391 / 3283` | `2985 / 2918` | `0.265` | `0.004` |

## Teacher preservation

The forgetting fix works. In every run:

- teacher-stage update flags are `base=0, residual=1, gate=0, critic=1`;
- all 198 logged deployment updates have
  `base=residual=gate=critic=0`;
- static oracle return has exactly zero range from iteration 1000 through
  iteration 1380.

Switching oracle return still varies because stochastic switching evaluation
uses transition disturbances; this is evaluation variance, not parameter
drift. The old final-oracle collapse is removed.

## Variance and inference diagnosis

The empirical EMA also works numerically, but it calibrates the wrong
statistic. Its learned variance closely matches its residual-second-moment
target, with mean log calibration error `0.064-0.083`. However, that target is
dominated by mean-model approximation error:

| Family | Env | Empirical mode variances | Expected variance order | Observed order | Accuracy |
|---|---|---|---|---|---:|
| deterministic mean | Ant | `.212,.207,.219,.201` | none | `3,1,0,2` | `.572` |
| deterministic mean | HalfCheetah | `.273,.276,.276,.262` | none | `3,0,2,1` | `.480` |
| mean plus variance | Ant | `.217,.206,.229,.210` | `0,1,2,3` approximately | `1,3,0,2` | `.599` |
| mean plus variance | HalfCheetah | `.290,.283,.281,.281` | `0,1,2,3` approximately | `3,2,1,0` | `.451` |
| variance only | Ant | `.215,.215,.215,.219` | `0,1,2,3` | `1,0,2,3` | `.305` |
| variance only | HalfCheetah | `.279,.280,.279,.280` | `0,1,2,3` | `0,2,3,1` | `.254` |

Variance-only posterior accuracy is therefore at four-class chance. The
fixed-state environment audit had already shown large real target-variance
separation, so the failure is not missing rollout noise. A single predictor's
total squared residual combines aleatoric noise, state-dependent model bias,
and finite-capacity error; its absolute second moment is not an aleatoric
variance estimator.

The bounded-NLL control has the opposite failure. Its variance stays at only
`.025-.029` while empirical residual targets are about `.19-.28`, producing
log calibration error `1.65-1.93`. The resulting surprise closes adaptation
almost completely on HalfCheetah (`gate=.004-.009`).

Mean-changing families are partially identifiable because posterior evidence
can use conditional mean differences. Their posterior accuracy reaches
`.41-.63`. The apparent learned gains in those rows therefore do not validate
variance inference.

## Comparison with fixed SAC and ESCP controls

| Family | Env | SAC | ESCP | NLL frozen | Empirical frozen |
|---|---|---:|---:|---:|---:|
| deterministic mean | Ant | `3505 / 3905` | `2511 / 3157` | `2422 / 2603` | `1992 / 2629` |
| deterministic mean | HalfCheetah | `2516 / 2593` | `2978 / 3079` | `1782 / 1930` | `3175 / 2931` |
| variance only | Ant | `3543 / 3394` | `1850 / 2806` | `1625 / 2564` | `1960 / 2865` |
| variance only | HalfCheetah | `2801 / 2752` | `3146 / 3103` | `2985 / 2918` | `2558 / 2542` |
| mean plus variance | Ant | `2333 / 2607` | `1497 / 2724` | `2142 / 3009` | `1757 / 2947` |
| mean plus variance | HalfCheetah | `2235 / 2140` | `2529 / 2570` | `2399 / 2345` | `1774 / 1596` |

No BAPR row exceeds both SAC and ESCP in both protocols. Empirical
deterministic-mean HalfCheetah is closest: it beats ESCP statically but is
`4.8%` lower in switching. NLL mean-plus-variance Ant has the strongest
switching score but remains below SAC statically.

These external comparisons are seed0 and cross-run, so they are secondary to
the within-checkpoint ladder. They are sufficient to reject promotion, not to
claim a statistically significant ranking.

## Scientific decision

Do not pull remote PKLs, expand seeds, or tune another variance/gate scalar.
This round establishes three separate facts:

1. frozen-teacher staging is now correct;
2. empirical residual EMA solves optimizer calibration but not aleatoric
   identification;
3. zero-mean actuator noise alone provides weak and run-dependent oracle
   control headroom, unlike the bus domain's state-dependent demand, traffic,
   and queue-cost randomness.

The next benchmark step must be oracle-first. Define a persistent stochastic
regime whose per-step exogenous distribution changes the optimal controller,
such as actuator packet loss or burst external load, while keeping robot
morphology fixed. Run only robust versus privileged-oracle headroom first. A
family proceeds to learned inference only if oracle exceeds robust by at least
10% in both static and switching on Ant and HalfCheetah.

If that gate passes, replace one-step residual variance with a causal sequence
estimator that explicitly consumes innovation moments. Its first acceptance
test is variance-mode classification and switch delay, before policy return.
For existing mean-changing positive controls, a separate controller experiment
may distill the frozen oracle action into the learned-context policy, but it
must branch from one shared teacher checkpoint to avoid the large SAC training
trajectory variation seen across these rows.
