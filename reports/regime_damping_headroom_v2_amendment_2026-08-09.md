# Persistent-damping headroom v2 pre-training amendment

The v1 environment design and scheduler DAG were registered before training. All 24 v1 GPU jobs (`t79103` through `t79126`) then exited in argument parsing because the isolated family `joint_damping_fault` was not included in the generic training CLI choice list. No environment was constructed, no optimizer step ran, and no checkpoint was written.

Version 2 changes only the isolated training entry so the already registered family string is accepted. The following remain byte-for-byte inherited from v1: physical damping modes, perturbation magnitude, per-step action noise, four environments, robust/oracle roles, three training seeds, three event seeds, 5.6M-step/350k-update budget, strict evaluation, and oracle-headroom decision threshold.

V2 uses new result, bundle, audit, analysis, registration, and scheduler-signature namespaces. The failed v1 task IDs and v1 registration hash are retained in the v2 registration. This is an engineering repair before training, not an environment or hypothesis change.

