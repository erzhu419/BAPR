# Frozen-base independent-adapter fork protocol

## Question

The event-grouped cross-context audit showed that HalfCheetah true context
beats zero and every fixed context, but its shared conditional controller is
still much worse than a separately trained robust controller. This screen asks
whether the failure is caused by negative transfer inside the shared
conditional actor/critic rather than by absent adaptation headroom.

## Common-controller rebootstrap

Every branch starts from the same completed `structured_channel`
HalfCheetah robust RegimeSAC checkpoint at iteration 1399, 5.6M transitions,
and 350k updates. The source actor was trained with an all-zero four-vector.
Its zero-context function is copied exactly into the observation-only base of
each BAPRRegime residual policy. The source critic and target critic are
embedded into the wider five-coordinate context input with all new context
weights set to zero.

This is not claimed to be an exact trajectory continuation. Every branch starts
with an empty replay buffer and reset optimizer states. Bootstrap validation requires exact equality of every copied parameter block and
all inserted context coordinates to be zero. Deterministic actions must agree to
absolute tolerance `1e-5`. The Q forward probe uses absolute tolerance `1e-3`
plus global-scale relative tolerance `2e-3` because CUDA GEMM changes its
float32 reduction path after zero context columns widen the input; the exact
parameter-layout tests remain the binding mapping check.

## Matched treatment budget

The robust branch continues for 700 iterations, or 2.8M new transitions and
175k updates. For each residual cap (`0.25`, `0.50`, `1.00`), four independent
fixed-mode branches each run 175 iterations, or 0.7M new transitions and
43,750 updates. Thus one four-adapter bank receives 2.8M aggregate new
transitions and 175k aggregate updates, equal to robust continuation.

Each adapter has its own policy residual, critic, target critic, alpha,
optimizer, replay, and scheduler task. The copied base actor is frozen and its
parameter hash must be unchanged at completion. The residual starts at exact
zero and must update. MuJoCo regularization remains off; the positive RE-SAC
regularization sign used by the bus implementation is untouched.

## Calibration and sealed evaluation

Training seed 8 is a development screen, not a final statistical claim.
Calibration disturbance streams are `74100` and `74200`. For each physical
mode, a robust-inclusive utility map selects among the frozen base and four
adapters by mean strict-horizon stationary return; an adapter is ineligible if
its termination rate exceeds the base by more than `0.02`.

Sealed streams `75100-75500` compare:

- continued robust controller;
- frozen 5.6M-step base;
- identity mode-to-adapter routing;
- the frozen robust-inclusive utility map;
- each of the four fixed adapters.

All cases use deterministic policy means, a 1000-step forced horizon, five
episodes per stationary mode, five switching streams, dwell 250, and paired
environment randomness inside each event task.

## Development promotion rule

A residual cap may expand to five independent training seeds only when the
utility-routed bank satisfies all of the following on the sealed development
streams:

1. at least 10% switching gain over the 8.4M-step robust continuation;
2. at least 5% mean stationary gain;
3. at least four of five paired switching-stream wins;
4. switching return above every fixed adapter;
5. termination rate no more than `0.05` above robust.

The five event streams share one trained controller and are not treated as
independent policy-training replicates. Passing this screen permits a five-seed
controller confirmation. Learned mode estimation remains blocked until that
confirmation passes.




## Scheduler launch

The preregistered graph was submitted as `t51149-t51180`: 13 GPU training
branches, three CPU calibrations, 15 sealed CPU audits, and one aggregate. Only
the 13 training branches were dispatched. Calibration, audit, and aggregate
tasks remain file-gated and cannot run before their producers publish validated
bundles.

Initial GPU startup exposed two operational issues rather than algorithm
failures. Five adapter attempts were rejected by an over-strict CUDA forward
comparison: copied actor outputs were exact, while the mathematically identical
widened critic GEMM differed by at most 0.0967 on a Q scale of 127.6. The
parameter mapping was exact; the runtime diagnostic now uses the documented
global-scale tolerance. Automatic retries are `t51183,t51184,t51188,t51189`. Retry `t51195`
confirmed the numerical fix but reproduced `LLVM ERROR: pthread_create failed`
after reaching training on the still-loaded node007. GPU children now
force single-thread XLA CPU, BLAS, JAX, and TF pools. The final retry
`t51199` excludes node007 but remains unpinned across local, jtl110gpu,
jtl110gpu2, and jtl311linux; it launched on jtl110gpu2. No training semantics,
checkpoint boundary, or controller budget changed.

The later five-seed confirmation observed the same hardware-only effect at a
slightly larger scale: maximum action difference `1.72e-6` and maximum global
Q difference `0.119%`. The forward diagnostic therefore uses the documented
`1e-5`/`0.2%` bounds above. Exact copied parameter hashes and zero inserted
coordinates remain unchanged and continue to reject any structural mismatch.
