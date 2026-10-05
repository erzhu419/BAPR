# JAX Positive-Sign Regularization Scale Pilot (Frozen 2026-10-04)

V45 showed that the bus minibatch reward transform rescues the positive-sign
regularized arm, but its strict switching return remains below the no-shift arm.
V46 measures whether the registered positive coefficient is too large in the
JAX actuator-polarity protocol. It changes only weight_reg from V45's 0.01 to
0.001 or 0.003; the sign remains positive and reward processing stays
10*(r-mean)/sample_std with ddof=1 and the same small-std fallback.

The same V45 seeds 115003/115023/115043 and the same 1.01M interaction, 1M
critic-update and 500k actor-update budget are used. The existing V45 compact
bus-standardized full and no-shift actors are the fixed paired baselines; only
six new GPU trainers are needed. Three CPU audits evaluate both new scales in
the identical stationary and switching windows, followed by one compact
aggregate. No historical checkpoint is resumed and no full checkpoint is
pulled.

The gate is a two-arm GPU smoke with finite logs, positive bonus, exact reward
moments and checkpoint continuation. Report each scale against V45 positive
0.01 and V45 no-shift by seed, with stationary and switching returns. A scale
is useful only if it improves the positive arm while retaining stable raw
returns; a smaller gap caused by degrading the no-shift baseline is not a
success criterion. This is a calibration pilot, not a new BAPR paper result.
