# Persistent actuator-polarity oracle-headroom screen

## Scientific question

The current `structured_channel` HalfCheetah benchmark does not provide
reproducible mode-conditioned control headroom after a mature robust policy is
trained. This screen asks whether BAPR's controller interface can exploit a
persistent transition regime whose optimal actions are genuinely conflicting.
It tests the benchmark before training any learned estimator.

## Environment

The new `actuator_polarity` family has four persistent modes. Each mode
reverses a different approximately half of the actuator channels:

1. low-index half;
2. high-index half;
3. even-index channels;
4. odd-index channels.

Gravity, morphology, reward, and absolute actuator authority remain fixed.
All modes receive the same independent Gaussian execution noise with standard
deviation 0.02. A mode remains fixed for 250 steps; no robot parameter or mode
is resampled per transition. This models a persistent actuator calibration or
wiring regime, not observation noise or per-step morphology changes.

## Sealed screen

- Environments: HalfCheetah, Ant, Hopper, and Walker2d.
- Independent training seeds: 8, 16, and 24.
- Arms: equal-budget `RegimeSAC` robust and true-mode oracle.
- Robust receives an all-zero context through the same architecture.
- Oracle receives the exact current mode one-hot.
- Budget per arm: 1400 iterations, 5.6M transitions, 350,000 updates.
- Evaluation: three held-out event streams per training seed, five stationary
  episodes per mode, and five strict 1000-step switching episodes.

The 24 GPU training jobs fit within one 36-job batch. GPU placement is
unbound across `local`, `jtl110gpu`, `jtl110gpu2`, and `node007`;
`jtl311linux` is excluded. CPU audits use only `node001-node006`, are
file-gated by their exact bundle manifests, and cannot launch before training
completes.

## Promotion gate

An environment passes only if:

1. oracle switching return improves by at least 15%;
2. oracle worst-mode stationary return improves by at least 15%;
3. paired 95% intervals across the three policy seeds are above zero for both;
4. at least three of four stationary modes improve;
5. switching termination increases by no more than five percentage points.

At least three of four environments must pass. Only then may a causal
probabilistic mode estimator be trained. Failure leaves this family as a
positive-control attempt and does not authorize gate, posterior, or BAPR
hyperparameter tuning.

## Scheduler launch

The complete file-gated graph was submitted at high priority:

- GPU training: `t52036-t52059`;
- strict CPU audits: `t52060-t52083`;
- aggregate decision: `t52084`.

The first dispatch launched 21 of 24 producers across `local`,
`jtl110gpu`, `jtl110gpu2`, and `node007`. After startup VRAM grace expired,
the scheduler placed the three remaining Walker2d oracle tasks automatically;
all 24 producers are now running and unpinned. Audit and analysis tasks are
blocked on their exact producer manifests.

The first live log confirms `CudaDevice(id=0)`, `actuator_polarity`,
250-step dwell, the correct robust context arm, and a fresh 1400-iteration
budget. No task may run on `jtl311linux`.

## Final result (2026-07-27)

All 24 equal-budget controllers reached iter 1400 / 5.6M transitions, all 24
strict audits completed, and aggregate `t52084` passed artifact and pairing
validation. The preregistered global gate fails with 0/4 passing environments:

| Environment | Robust switching | Oracle switching | Relative gain | Robust worst | Oracle worst | Relative gain | Mode wins | Formal result |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 738.8 | 2633.9 | +256.5% | 138.2 | 1348.2 | +875.6% | 4/4 | fail CI |
| Ant | 1244.8 | 2304.5 | +85.1% | 663.9 | 1602.2 | +141.3% | 4/4 | fail CI |
| Hopper | 2671.3 | 2750.4 | +3.0% | 168.1 | 192.1 | +14.3% | 3/4 | fail effect |
| Walker2d | 2287.5 | 2302.6 | +0.7% | 176.6 | 200.9 | +13.8% | 2/4 | fail effect |

The formal failure has two different causes. HalfCheetah and Ant show large
descriptive controller headroom: oracle switching and worst-mode returns beat
robust in all three independent policy seeds, all four mode means improve, and
none of the nine paired event-stream comparisons reverses sign. Event-stream
variation is small; policy-training seed variation dominates. With only three
independent policy seeds, however, the paired t intervals are wide and cross
zero. These are promising but underpowered positive-control candidates, not
confirmed environments.

Hopper and Walker2d are genuine failures of this severity setting. Effects are
small and change sign across policy seeds. Both robust and oracle also
terminate in 100% of stationary and switching evaluations, so their return
comparison is not an acceptable adaptation-positive benchmark even though the
relative termination gap is zero. Future screens need an absolute survival
criterion in addition to the relative termination criterion.

This result does not evaluate a learned BAPR estimator. It evaluates whether
perfect current-mode information can help an otherwise identical controller.
The benchmark implementation is temporally coherent: actuator polarity remains
fixed for a 250-step dwell, morphology and gravity do not change, and only
independent Gaussian actuator noise with standard deviation 0.02 is sampled
per transition. The result therefore does not support a per-step morphology or
observation-noise explanation.

## Admissible next steps

1. Reuse the completed HalfCheetah and Ant checkpoints for a no-training
   cross-context audit: true, zero, fixed 0-3, cyclically wrong, and shuffled
   context must be compared inside the same process and event stream.
2. Add delayed-oracle cases at 1, 5, 10, 25, and 50 steps after each switch.
   A causal estimator cannot match an instantaneous oracle before observing a
   transition; learned inference is justified only if useful headroom survives
   realistic delay.
3. If both checks pass, preregister a new five-policy-seed confirmation for
   HalfCheetah and Ant. The current three seeds remain exploratory and must not
   be combined with event seeds as if there were nine independent policies.
4. Redesign Hopper and Walker2d before retraining. Use milder, invertible
   actuator calibration transforms and require absolute stationary survival
   before comparing returns.
5. Only after controller headroom and causal-delay tolerance are confirmed may
   BAPR train a heteroscedastic transition-likelihood model, sticky
   semi-Markov posterior, and posterior-conditioned policy. Gate tuning,
   BOCD/LCB variants, and another residual bank remain blocked.
