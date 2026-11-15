# Expected-action system-ID ablation

## Purpose

Executed-action inverse system ID passes all five-seed frozen gates and
recovers 89.3% of true-oracle switching headroom. Its remaining weakness is
that the exploratory simulator exposed the realized noisy
`executed_action` as an inverse-model target.

This v4 development ablation tests whether that privileged realized signal is
necessary. It preserves the v3 network, data splits, policy checkpoints,
candidate transforms, sticky-filter search, online inputs, event streams, and
all gates. Only the training target changes.

## Supervision boundary

Training rollout returns the standard five transition fields only. For a
transition with commanded action `a` and known exploratory training mode `m`,
the target is

`clip(gain(m) * a, -1, 1)`.

The realized Gaussian actuator noise is never requested, stored, or used.
Training mode remains privileged; it is already required by the true-context
conditioned-policy curriculum and is a standard asymmetric-training signal.

Online inference remains limited to consecutive observations and commanded
action. It receives no mode ID, selected gain vector, realized executed
action, reward label, future transition, or switch clock.

## Interpretation and gates

Seeds `8,16` fit the inverse model, seed `24` selects filter parameters, and
controller seeds `101,211,307,419,523` rerun the closed-loop development
audit. These five seeds are no longer untouched because v3 has already
reported them; v4 is an ablation, not an independent confirmation.

The frozen thresholds remain:

- mode accuracy at least `0.85`;
- median and P90 switch delay at most `25` and `50` actions;
- Brier score at most `0.25`;
- at least 50% oracle-headroom recovery;
- at least 4/5 policy-seed wins over robust;
- termination gap no more than five percentage points.

A pass establishes that realized simulator noise supervision is unnecessary.
The final method must then be frozen and evaluated on newly trained policy
seeds. A failure means v3 must be presented explicitly as a privileged
simulator positive control rather than the deployable BAPR estimator.

## Scheduler graph

The scheduler-only development graph is:

- `t58608`: expected-action inverse-model training on an eligible GPU;
- `t58609-t58613`: five file-gated CPU closed-loop audits;
- `t58614`: aggregate, gated on all five audit manifests.

GPU placement is unpinned within `local`, `jtl110gpu`, and `node007`, with a
2.5 GB claim inherited from the measured v3 run. `jtl110gpu2` and
`jtl311linux` are excluded. The graph uses neither Slurm nor auto-adopt.

## Final result

All model and audit artifacts pass schema and hash validation. As in v3, the
remote aggregate was marked done without synchronizing its output; the same
pure-JSON aggregation was rerun locally after validating all five audit
manifests.

Removing realized `executed_action` supervision has negligible effect. The
seed-24 validation split passes with `0.9969` switching accuracy and selects
the same filter as v3: `hazard=0.002`, `evidence_scale=1.0`, and
`posterior_decay=0.98`.

| Arm | Switching return | Delta vs robust | Seed wins | Oracle headroom recovery |
|---|---:|---:|---:|---:|
| robust | 1023.4 | - | - | - |
| true oracle | 2327.7 | +1304.3 | 5/5 | 100% |
| expected-action soft | 2171.3 | +1147.9 | 5/5 | 86.2% |
| expected-action MAP | 2112.1 | +1088.7 | 5/5 | 82.7% |

The soft paired improvement is `+1147.9`, 95% CI
`[+604.7,+1691.1]`. All five policy seeds improve, and every seed recovers at
least `69.2%` of its oracle headroom. Inference accuracy is `0.9975`, Brier
score `0.0040`, median delay `5`, and P90 delay `12`. Termination remains
zero. Stationary mean (`2312.1`) and worst-mode return (`1980.8`) remain
close to oracle (`2320.3`, `2003.0`).

Relative to realized-action v3, soft switching changes only from `2190.5` to
`2171.3`, while inference accuracy changes from `0.9992` to `0.9975`.
Therefore realized simulator noise labels are not required. The mechanism is
now frozen as: expected transformed commanded-action supervision, independent
inverse ensemble, empirical residual variance, candidate-transform mixture
likelihood, and sticky causal posterior with the selected fixed filter.

The repeated policy seeds make this a successful development ablation, not
the final statistical confirmation. The final protocol must train robust and
true-context controllers from scratch on new policy seeds, use new event
streams, and evaluate the frozen v4 estimator without any further model,
filter, gate, or environment changes.
