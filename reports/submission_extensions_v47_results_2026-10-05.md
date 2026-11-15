# Publication-extension results

All 72 logical scheduler jobs, including four aggregations, completed. The 24 audits
were synchronized automatically; this analysis pulled no checkpoints or datasets.
The frozen protocol is in `submission_extensions_v47_preregistration_2026-10-05.md`.
All intervals below are paired 95% t intervals over policy-seed means.

## Reacher: independent estimator initializations

| Estimator seed | BAPR switching return | Matched conditioned SAC | Paired improvement [95% CI] | Wins |
|---|---:|---:|---:|---:|
|95103|-7.015|-14.626|7.611 [5.736, 9.486]|5/5|
|95121|-7.094|-13.839|6.745 [5.467, 8.023]|5/5|
|95139|-6.918|-15.085|8.167 [6.290, 10.044]|5/5|

Robust SAC averaged -26.274 and oracle/native reference -4.824. All three
estimators retain positive gains against their own information-matched controls.
The five existing policy seeds are crossed with three estimator initializations;
the independent inference unit remains five policies, not fifteen pairs.

## Reacher: non-diagonal coordinates

Five new policy seeds and three new estimators tested four predefined signed
permutations, including two channel permutations. The preselected estimator
118003 gave BAPR -6.436, matched conditioned SAC -11.109, robust SAC -25.828
and oracle -4.986. The primary gain was 4.674 [2.920, 6.427], wins 5/5;
the gain over robust was 19.393 [17.653, 21.132], wins 5/5. The other two
estimators gave -6.413 and -6.449. Oracle beat every fixed context in all seeds.
The primary estimator detected 223/225 switches, with median delay 12 steps
among detected switches; two switches remained unresolved.

## HalfCheetah: source trainer and calibration sensitivity

| Source-comparison route | Switching return | BAPR minus route [95% CI] | Wins |
|---|---:|---:|---:|
|BAPR|3632.630|--|--|
|Original Torch RE-SAC, positive weight_reg=0.01|-195.463|3828.094 [2969.360, 4686.827]|5/5|
|Same source trainer, weight_reg=0|2010.103|1622.527 [615.885, 2629.169]|5/5|
|Oracle|3827.995|--|--|

Both source arms completed 8.4M interactions and 525,000 critic calls;
524,774-524,790 critic optimizer steps and all 262,500 actor updates ran.
The positive sign is preserved. In the final seed109003 log, the full arm's
Q mean/std were 9735.2/134.1 versus 141.0/0.9 without the shift. The failure
is not an unfinished run or extensive AMP skipping, nor is it exclusive to
the earlier JAX port. This single log does not isolate the failure's cause.

The separate nine-policy calibration sensitivity audit gave switching returns
2702.761, 2718.497 and 2773.305 versus oracle 3298.097 and uncompensated
369.205. Every estimator beat uncompensated and lost to oracle in all nine
policies. This recipe retained stationary return around 3255 versus oracle
3313, but had 23-32 unresolved switches out of 405 per estimator.

## Accounting And Limits

All audits retained the full 200/1000-step horizons, three switching windows,
finite returns and zero native terminations. Applicable oracle/native returns
were identical. Reacher actors used 1.6M interactions/100,000 updates; each
estimator used 178k training/82k validation transitions and 4500 updates.
Identically regenerated calibration data are not independent new information.

The source comparison preserves the original trainer/networks on the shared
Brax benchmark, not the original Gym-paper collector/protocol. Its actor
budget differs from BAPR's 525,000 updates, and BAPR has additional estimator
training cost. The no-shift arm remains an ensemble algorithm, not SAC.
The HalfCheetah calibration audit uses historical policies and a different
recipe, not multi-initialization confirmation of the original two-stage model.
The coordinate extension covers known box-preserving signed permutations,
not arbitrary gains, delays or irreversible faults. Cohorts are not pooled;
the original ten-seed 8/10 composite gate remains FAIL, with 7/10 successes.

The manuscript now includes the estimator-robustness and independent
coordinate-extension evidence, the source comparison and both HalfCheetah
limitations. The compact-result checker and 21 focused tests pass; the
26-page PDF compiles without LaTeX warnings and its new tables were inspected.
The frozen algorithm is unchanged. No further jobs were submitted.
