# Regime-adapter multi-seed confirmation protocol

## Development result

The frozen-base independent-adapter screen isolated HalfCheetah controller
interference from mode-estimation error. Every adapter branch starts from the
same audited 5.6M-transition robust actor, freezes that base, and independently
trains its residual policy, critic, target critic, alpha, optimizer, and replay.
One four-adapter bank and the robust continuation each receive 2.8M additional
transitions and 175k updates in aggregate.

On development training seed 8, residual cap `0.50` with the fixed identity map
`[0,1,2,3]` passed the sealed screen:

| Metric | Continued robust | Identity adapter bank | Gain |
|---|---:|---:|---:|
| Mean stationary return | 2196.9 | 2583.0 | +386.2 (+17.6%) |
| Mean switching return | 2212.0 | 2569.0 | +357.0 (+16.1%) |

The identity bank won all five paired switching event streams, beat the best
single fixed adapter, and added no terminations. This establishes controller
headroom under true mode but is only a one-training-seed development result.

## Frozen confirmation

No hyperparameter, controller map, or per-seed calibration may change in this
round:

- environment: `HalfCheetah-v2`, `structured_channel`;
- residual cap: `0.50`;
- true-mode routing: identity map `[0,1,2,3]`;
- policy-training seeds: `8,16,24,32,40`;
- primary untouched seeds: `16,24,32,40`;
- sealed event seeds: `76100,76200,76300,76400,76500`;
- deterministic policy mean, forced 1000-step horizon, dwell 250;
- five stationary episodes per mode and five switching episodes per event;
- equal aggregate post-fork budgets as in the development screen.

Seed 8 is retained only as a descriptive development row. It is excluded from
the primary interval and win count. The five event streams are averaged within
each policy-training seed; the four independently trained holdout policies are
the inferential units.

The audit compares continued robust, the frozen pre-fork base, identity routing,
and every fixed adapter `0-3` in the same process and paired environment stream.
There is no calibration task and no confirmation result can alter the routing
map.

## Promotion gate

Training a causal learned router is permitted only if all hold on untouched
training seeds `16,24,32,40`:

1. identity routing improves mean stationary return by at least 5% over robust;
2. identity routing improves mean switching return by at least 10% over robust;
3. paired two-sided 95% t intervals for both improvements lie above zero;
4. identity beats robust on both metrics in all four seeds;
5. identity beats each seed's best fixed adapter on stationary and switching
   return, with paired intervals above zero and four of four wins;
6. its switching termination rate is never more than 0.05 above robust.

A pass proves that independently specialized controllers plus privileged mode
routing have reproducible value. It does not by itself prove learned online
adaptation. A failure blocks estimator training and points back to controller
variance or insufficient specialization headroom.

## Scheduler graph

The graph contains 20 new GPU branches for seeds `16,24,32,40`, 25 CPU-only
sealed audits, and one CPU aggregate. The five seed-8 training bundles are
reused only after hash validation. GPU branches are portable across available
GPU nodes and retain checkpoint-managed resume. Their 2300 MB claim is based on
the observed 0.82-2.09 GB development peak plus a small margin. Audits are
restricted to `node001-node006` and wait for all five validated training bundle
manifests and checkpoint files. Only training tasks are explicitly dispatched;
the scheduler releases audits and aggregation when their files exist.

The graph was submitted atomically as `t51224-t51269`: training
`t51224-t51243`, audits `t51244-t51268`, and aggregate `t51269`. The first
bulk start reproduced two development-era operational effects. Concurrent JAX
cold starts on node007 exhausted its pthread limit, while the cross-GPU forward
diagnostic observed at most `1.72e-6` action difference and `0.119%` global Q
difference despite exact copied parameter blocks. The diagnostic bounds were
therefore documented as action `5e-6` and global Q `0.2%`; structural hashes,
zero residual initialization, and zero inserted context coordinates remain
exact. Scheduler child retries preserve the same signature, checkpoint paths,
budget, and configuration. No algorithm setting was changed.

Three signatures exhausted the automatic retry budget because their retries
were repeatedly returned to node007: seed-32 robust, seed-40 mode-0, and
seed-24 robust. Checkpoint-safe replacements `t51294`, `t51296`, and `t51298`
exclude node007 but remain portable across local and the three jtl GPU nodes.
All 20 unique training signatures subsequently reached `running`; the five
seed-8 audits completed while the other 20 audits remained correctly blocked
on their own seed's producer files.

## Final result

All 25 training bundles and 25 sealed audits completed and passed content-hash,
identity, and budget validation. The local validator initially rejected every
audit because remote manifests recorded the remote absolute workspace root.
The manifest SHA256 and sizes were identical. Provenance comparison now uses
the workspace-relative `jax_experiments/...` key, remains hash-strict, and is
portable across scheduler nodes. Independent local aggregation then reproduced
the scheduler result.

| Seed | Role | Robust stationary | Identity stationary | Robust switching | Identity switching |
|---:|---|---:|---:|---:|---:|
| 8 | development | 2200.0 | 2559.3 | 2225.5 | 2580.3 |
| 16 | holdout | 2302.4 | 2468.7 | 2266.7 | 2313.4 |
| 24 | holdout | 1960.8 | 1607.9 | 1949.9 | 1616.0 |
| 32 | holdout | 2425.3 | 1984.8 | 2323.0 | 1838.8 |
| 40 | holdout | 2431.6 | 2284.6 | 2286.3 | 2147.7 |

On untouched seeds, identity routing is `-8.5%` stationary and `-10.3%`
switching relative to the equally aggregated robust continuation, with only
one of four wins. The promotion gate fails and learned-router training remains
blocked.

This failure does not mean that persistent mode information lacks control
value. Relative to the frozen 5.6M-step base, identity routing improves
stationary return by `+299.8`, 95% CI `[+63.6,+535.9]`, and switching return by
`+236.1`, 95% CI `[+95.8,+376.4]`, winning all four holdout seeds. The correct
mode adapter is the best stationary expert in 13/16 holdout mode rows. The
single robust controller simply uses the 2.8M continuation transitions more
efficiently: its improvement over the frozen base is `+493.3` stationary and
`+463.6` switching. Independent adapters recover only 60.8% and 50.9% of those
gains because each critic/optimizer receives 0.7M transitions after the fork.

The next experiment is therefore an explicitly compute-unmatched diagnostic,
not a paper result: train every adapter from the common iter-1400 source to the
same 2.8M post-fork per-controller budget as robust. The compact iter-1575
adapter bundles intentionally omit replay, so extending them would introduce a
second replay reset. A clean diagnostic instead reboots once at the original
fork and trains continuously for 700 iterations. If that upper bound remains
weak, independent residual optimization is invalid. If it becomes strong, the
final algorithm must preserve a shared all-mode critic/backbone while keeping
mode-specific residual heads, rather than diluting data across four critics.
