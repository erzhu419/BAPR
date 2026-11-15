# Fresh ten-seed BAPR deployment confirmation preregistration

## Scope

This experiment tests the deployment performance of the frozen BAPR teacher-estimator-student pipeline on the HalfCheetah actuator-polarity benchmark. It does not test sample efficiency, equal end-to-end construction cost, or universal superiority across nonstationary environments.

The experiment cannot launch unless the separately registered checkpoint-only mechanism audit writes a valid `MECHANISM_PASS` artifact. Failure of that gate ends this confirmation branch without training another model family.

## Frozen cohort

- Model seeds: `52021, 52127, 52237, 52349, 52457, 52567, 52679, 52783, 52889, 52999`.
- Event seeds: `152021, 152127, 152237, 152349, 152457`.
- The prior five model seeds and all of their returns are excluded.
- Methods: frozen-pipeline BAPR, SAC, recurrent ESCP, and released-B0 RE-SAC.
- Each single-controller baseline receives 5.6M environment steps and 350k updates.
- BAPR reuses the frozen teacher bank, estimator, robust fallback, training streams, and selection rule. Only student initialization is new; its complete construction budget is reported separately.

## Evaluation

Every model is evaluated deterministically on the same five event streams. Each event contains four stationary mode audits and a causal switching audit with a strict 1000-step horizon and fixed 250-step dwell. BAPR receives only observation, commanded action, reward, next observation, and its causal posterior; true mode, executed action, gain, and switch clock remain forbidden.

## Registered decision

Three one-sided paired hypotheses form one family: BAPR exceeds SAC, recurrent ESCP, and released-B0 RE-SAC in switching return. Holm-adjusted `p <= 0.05`, a positive Bonferroni simultaneous one-sided lower bound, at least 8/10 seed-slot wins, stationary retention of at least 95%, and no positive switching-termination gap are required for every named baseline comparison.

The per-seed strongest-baseline envelope is reported only as a stress-test diagnostic because choosing a different comparator after observing each seed is not a preregistered baseline-method hypothesis.

## Execution

The scheduler DAG contains 40 independent GPU training jobs, 40 CPU-only checkpoint audits, and one CPU aggregation job. GPU jobs are not hard-pinned and may use `local`, `jtl110gpu`, `jtl110gpu2`, `jtl311linux`, or `node007`. CPU audits use `node001` through `node006`. Checkpoint resume and result-file dependencies are mandatory; Slurm and auto-adopt are not used.

