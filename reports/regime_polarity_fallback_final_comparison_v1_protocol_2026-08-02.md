# Causal-fallback BAPR final comparison protocol

## Frozen method

The final BAPR method is the `mode_heads` policy, frozen expected-action
inverse-system-ID estimator, robust `final_seed_719`, and the causal
`evidence_1p0_k1` fallback as one stack. The fallback starts active, enters on
contradictory one-step evidence, posterior MAP change, or confidence below
`0.60`, and leaves after one supported transition with posterior confidence at
least `0.90`. It receives no physical mode, realized action, actuator gain,
switch clock, or future transition.

All source bundles, model files, runtime code, fallback configuration, and
completed selection/causal-ceiling analyses are recorded in one immutable
registration. Final tasks validate every recorded SHA-256 before training or
evaluation.

## Independent seeds

Five previously unused seed labels are fixed as
`2009,2113,2213,2311,2417`. For each seed:

- one new `mode_heads` student is trained with the already frozen combined-ten
  teacher, architecture, DAgger data, development events, checkpoint-selection
  rule, estimator, and fallback threshold;
- SAC, ESCP, and RE-SAC are trained from scratch in the identical
  `HalfCheetah-v2` `actuator_polarity` environment;
- no model is selected or removed using final-event performance.

The five untouched final event seeds are
`106301,106331,106367,106399,106451`. Every method sees the same stationary
tasks, switching mode streams, 250-action dwell, five 1000-action strict
episodes, rewards, termination semantics, and deterministic evaluation.

## Baseline budget

Each SAC/ESCP/RE-SAC controller receives `1400 x 4000 = 5.6M` environment
transitions and `350000` optimizer updates. SAC and ESCP use learning rate
`3e-4`. RE-SAC preserves the proven positive regularization sign with
`weight_reg=beta_ood=0.01`, `beta=-2`, and the original implementation-scale
learning rate `1e-5`. Any RE-SAC collapse is reported as a reproduction warning
and cannot be used to support BAPR; BAPR must pass against the strongest of all
three baselines for each registered seed slot.

This is a deployment-performance comparison, not a sample-efficiency claim.
BAPR's frozen teacher was constructed from ten robust and ten oracle source
controllers, whereas each reported baseline row is one independently trained
controller. Both the single strongest robust controller and the robust
population remain relevant additional controls in the historical analyses.

## Frozen decision

For every method, the five model-seed summaries first average the same five
event clusters. The primary comparison is BAPR versus the strongest
SAC/ESCP/RE-SAC return within each registered seed slot. The two five-seed
vectors use a conservative two-sample 95% interval with critical value
`2.7764`; event episodes are not treated as independent samples.

The final primary gate passes only when all conditions hold:

- mean switching advantage over the strongest baseline is positive;
- the conservative two-sample interval is strictly positive;
- BAPR wins at least `4/5` registered seed slots;
- mean stationary return retains at least `95%` of the strongest baseline;
- switching termination rate is no worse than the strongest baseline.

Failure is final for this benchmark. Seeds cannot be extended, reweighted, or
used to select another student, baseline hyperparameter, fallback threshold,
or environment setting.

## Scheduler graph

The graph contains 20 GPU training tasks: five BAPR students plus five each of
SAC, ESCP, and RE-SAC. It then contains 20 file-gated CPU audits, one per
method/model seed, and one immutable JSON aggregate. All task specifications
are submitted in one scheduler JSONL transaction.

GPU tasks allow only `jtl311linux` and request measured cold-start VRAM of
`2300 MB` for baselines or `2400 MB` for student distillation. CPU audits allow
only `node001-node006`, request `vram=0`, and cannot launch before their exact
model/bundle files exist. No task uses Slurm, auto-adopt, or another GPU node;
checkpoint resume is managed by the controller/student command.

## Submitted graph

The executable graph is `t65066-t65106`: GPU training is `t65066-t65085`,
file-gated CPU audit is `t65086-t65105`, and the aggregate is `t65106`. The
first launch wave placed exactly three measured tasks on each `jtl311linux`
GPU. The per-task one-third packing override is enabled only for these frozen
2300/2400 MB signatures; the scheduler-wide packing policy is unchanged.

Two earlier submission records are excluded from the experiment. Tasks
`t64981-t65021` exposed a fail-closed launch-input staging omission and never
entered valid training. Tasks `t65022-t65062` were cancelled while queued after
the scheduler's directory-only staging contract was identified. The final
graph stages every registered artifact through 65 existing directories, and
the first six tasks passed registration validation before training began.
The scheduler `submit-jsonl` path was also fixed to preserve
`resume_managed_by_cmd`; all 20 training records now carry that flag, an exact
checkpoint directory, and an explicit `--resume` command.

## Final result (2026-08-03)

All five students, 15 baseline bundles, 20 strict audit manifests, and the
aggregate `t65106` completed. A local immutable-analyzer rerun revalidated the
registration, manifests, and shared mode traces. Switching results are:

| method | mean +/- seed std | stationary | termination |
|---|---:|---:|---:|
| BAPR | 2015.3 +/- 32.1 | 2118.7 +/- 14.5 | 0.000 |
| SAC | 1222.5 +/- 772.4 | 1257.6 +/- 766.4 | 0.000 |
| ESCP | 1858.5 +/- 710.6 | 1912.5 +/- 723.1 | 0.000 |
| RE-SAC | -375.9 +/- 69.2 | -364.2 +/- 54.8 | 0.000 |

The registered seed-slot comparison against the stronger baseline is:

| seed | BAPR | strongest baseline | delta |
|---:|---:|---|---:|
| 2009 | 1968.1 | SAC 2370.3 | -402.2 |
| 2113 | 2034.0 | SAC 1499.4 | +534.6 |
| 2213 | 2053.6 | ESCP 2715.1 | -661.4 |
| 2311 | 2006.9 | ESCP 1569.5 | +437.4 |
| 2417 | 2014.1 | ESCP 2516.9 | -502.8 |

The primary result is therefore a failure: mean delta `-118.9`, conservative
95% interval `[-817.3,+579.5]`, and only `2/5` wins. Stationary retention
passes at `95.5%`, termination is tied at zero, and fallback is active for only
`1.136%` of switching actions. Pairwise exploratory comparisons are BAPR minus
SAC `+792.8` with `4/5` wins and BAPR minus ESCP `+156.9` with `3/5` wins, but
both intervals cross zero. These cannot replace the frozen primary test.

BAPR's narrow spread shows deployment consistency conditional on the shared
frozen teacher/data; it is not an apples-to-apples claim of lower RL training
variance. The remaining limitation is controller/teacher headroom near a 2k
return plateau, not termination or a large switching-inference loss.

RE-SAC is retained only as a reproduction warning. The completed tasks match
their registered configuration and exact `5.6M/350k` budget, but that
configuration mixes the bus-era `1e-5` learning rate with a low `0.0625`
update/data ratio. The local MuJoCo reference config at
`RE-SAC/experiment/configs/resac/halfcheetah.yaml` instead uses `3e-4`, roughly
one update per transition, `beta_bc=0.001`, and critic/actor ratio 2. A faithful
MuJoCo RE-SAC reproduction must be a separate protocol and cannot retroactively
change this final comparison.
