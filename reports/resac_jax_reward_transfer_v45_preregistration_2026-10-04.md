# JAX Reward Transfer Pilot (Frozen 2026-10-04)

V44 restored native stationary positive-shift performance by transferring only
the original bus minibatch reward transform. V45 tests that same intervention
in the unchanged V42 high-density HalfCheetah actuator-polarity world. V32 stays
frozen; no historical cohort is pooled or replaced.

Three new development seeds: 115003, 115023, 115043. Cross raw versus
bus-standardized training rewards with positive weight_reg=0.01 versus zero;
12 independent GPU tasks. The no-shift arm retains ensemble/LCB/OOD/BC.
For each minibatch, transform 10*(r-mean(r))/sample_std(r), ddof=1, substituting
std=1 when std<1e-6. Normalize within each scan step, not across stacked updates.
Replay stores raw rewards and all evaluations report raw returns.

Each run: 10k random warmup + 250*4000 interactions = 1.01M steps;
250*4000 = 1M critic updates and 500k actor updates. Preserve the V42 FP32
Torch-equation JAX networks, alpha loss, initialization, clipping, actor period,
positive norm sign and all physics/noise/switch schedules. Sampling and updating
remain blocked, not the native Torch interleaved loop. This is a bridge pilot,
not the full 8.4M-step BAPR comparison or original-paper reproduction.

Final stationary and switching audits reuse the frozen strict 1000-step V32
event windows, deterministic final actors and event pairing. Policy seed is the
unit of uncertainty. Report all four arm means/SDs, paired descriptive 95% t
intervals for both norm gaps, both absolute reward gains and their interaction.
Recovery requires improving the positive arm itself; a smaller gap caused by
damaging the no-shift arm is not recovery. Positive-shift superiority is a
separate question. Q scale alone is not a return-failure criterion.

Gate training on a four-arm scheduler GPU smoke: finite actual update logs,
restored budget/actor phase, positive versus zero bonus, per-batch reward moments
and exact raw controls. Focused tests cover source-formula parity, scan axes,
checkpoint continuation and real CLI controls. Gate each CPU audit on its four
compact actors; aggregate only after three audits. Sync compact actors/JSON only;
full checkpoints and replay remain remote. No local training, Slurm or adoption.
