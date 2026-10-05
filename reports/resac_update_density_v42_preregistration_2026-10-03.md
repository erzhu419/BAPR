# RE-SAC Update-Density Diagnostic

V41 source-equation alignment did not restore the positive norm treatment:
switching full -201.3 +/-43.7, no-shift2142.6 +/-625.4, paired full-minus-zero
-2343.9 [-3836.7,-851.2], wins0/3. Fixed-budget actors/audits validate.
The source `experiment/configs/resac/halfcheetah.yaml` and DSAC
`rlkit/core/vec_online_rl_algorithm.py` execute1000 updates per1000 interactions,
whereas V41 executes250 per4000. This remaining difference is tested next.

Fresh development seeds112003/112023/112043; four arms, all Torch-equation-aligned:
low_full, low_no_shift, high_full, high_no_shift. Only updates_per_iter250/4000
and weight_reg0.01/0 differ. Positive sign, OOD0.01/BC0.001/LCB-2, ten critics,
source initialization/LayerNorm/alpha ordering, actor period2/clipping1 remain fixed.
Unchanged spring HalfCheetah actuator polarity, noise0.02, dwell250, horizon1000.
Common pre-iteration random warmup10000, then250 iterations of4000 interactions:
1,010,000 total steps. Low/high critic updates62,500/1,000,000, actor updates
31,250/500,000. Equal interactions, explicitly unequal compute. No early stopping,
checkpoint selection, historical pooling or post-hoc extension of this cohort.

Primary diagnostic: paired switching density-by-norm interaction
(high_full-high_no_shift)-(low_full-low_no_shift), averaged over the unchanged
strict event windows then three seeds. Also report all four planned paired
contrasts, stationary returns, finite losses, Q/bonus/alpha and time samples.
A positive interaction alone does not establish usable full-regularized control;
report absolute full returns and their gap to the same-density no-shift arm.
Scheduler only: high-density GPU smoke, twelve independent trainers, three CPU
audits, one aggregate. Small actor bundles and JSON only; checkpoints/replay
remain remote. Resume only the new arm's own checkpoint, preserving RNG/Adam.
Next: use the interaction to assess whether update density contributes to failure.

Limitations: three development seeds and reused evaluation windows; one-million
interaction pilot, not V41's8.4M budget. Blocked sampling remains unchanged, unlike
the source's online interleaving. FP32/Brax/polarity remain different from the
original Gym/AMP experiment. This is not a paper reproduction or BAPR confirmation.
