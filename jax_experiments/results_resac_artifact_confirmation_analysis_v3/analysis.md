# RE-SAC released-artifact B0 confirmation

This is a fresh five-seed confirmation at the released controller budget: 2,000 iterations, 4,000 environment steps per iteration (8M total), and 250 updates per eligible iteration. SAC and RE-SAC share the same continuous gravity task stream.

Evaluation uses three independently generated held-out 40-task streams per training seed, deterministic-mean actions, strict 1,000-step horizons, and a 500-step switching sequence.

## HalfCheetah-v2

| Method | Stationary OOD | Worst quartile | Switching |
|---|---:|---:|---:|
| SAC | 1216.8 +/- 620.8 | -27.3 +/- 223.7 | 225.0 +/- 134.5 |
| RESAC | 925.4 +/- 330.3 | 250.7 +/- 418.8 | 348.5 +/- 323.9 |

Paired RE-SAC minus SAC:
- stationary mean: -291.4 +/- 519.1; 1/5 seed wins.
- worst quartile: +278.1 +/- 295.5; 4/5 seed wins.
- switching: +123.6 +/- 312.8; 3/5 seed wins.

Released-result cross-check (different legacy evaluation stream):
- SAC 4610.2 +/- 1100.2; RE-SAC 5327.9 +/- 1752.9.
- Paired delta +717.7 (4/5 wins); paper worst-quartile SAC -266, RE-SAC 740.

Numerical/provenance pass: **True**.

## Ant-v2

| Method | Stationary OOD | Worst quartile | Switching |
|---|---:|---:|---:|
| SAC | 656.7 +/- 48.8 | 177.8 +/- 46.2 | 302.7 +/- 153.2 |
| RESAC | 722.1 +/- 30.3 | 278.6 +/- 79.5 | 511.1 +/- 195.5 |

Paired RE-SAC minus SAC:
- stationary mean: +65.4 +/- 78.0; 4/5 seed wins.
- worst quartile: +100.8 +/- 81.3; 5/5 seed wins.
- switching: +208.4 +/- 78.0; 5/5 seed wins.

Released-result cross-check (different legacy evaluation stream):
- SAC 3912.0 +/- 972.3; RE-SAC 3866.4 +/- 637.9.
- Paired delta -45.7 (2/5 wins); paper worst-quartile SAC 1093, RE-SAC 1208.

Numerical/provenance pass: **True**.

## Interpretation boundary

Global numerical pass: **True**.

The fresh strict audit is the comparison used for current claims. The released values are a provenance cross-check only; agreement or disagreement must be interpreted together with the task-stream and evaluation-protocol differences. ESCP is deliberately excluded until the original recurrent history encoder is reproduced.
