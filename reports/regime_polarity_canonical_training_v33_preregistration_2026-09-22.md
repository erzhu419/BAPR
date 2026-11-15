# Canonical-controller training ablation

This exploratory development experiment tests whether the current controller
recipe wastes interaction budget by first learning a multimode compromise. It
does not alter the V32 result or its failed composite decision.

Five new policy seeds (89003, 89021, 89039, 89057, 89079) compare four SAC
recipes on the frozen HalfCheetah actuator-polarity benchmark:

| Recipe | Training path | Policy interactions |
|---|---|---:|
| warmstart_mode0 | 5.6M multimode source + 2.8M fixed mode-0 fine-tune | 8.4M |
| robust_multimode | multimode from random initialization | 8.4M |
| direct_mode0 | fixed mode 0 from random initialization | 8.4M |
| canonicalized_multimode | persistent modes, with the physics action interface transformed to mode-0 coordinates | 8.4M |

All policies use the same optimizer/update budget. The canonicalized interface
is enabled only during training and only for the actuator-polarity family. It
keeps actor, critic, target action, and replay action in the same reference
coordinate system. Strict evaluation disables this interface and restores the
true hidden actuator regimes.

Each canonical recipe is evaluated with no compensation, true-mode
compensation, and the same frozen v5 causal estimator. Three new stationary
event streams and three new switching schedules are used. Selection uses causal
switching return, not oracle return alone.

A new recipe is eligible to replace warmstart_mode0 only if it:

1. has a higher mean paired causal switching return;
2. wins at least four of five paired policy seeds;
3. retains at least 95% of stationary oracle return in at least four seeds; and
4. passes exact true-mode compensation trajectory equivalence.

Among eligible recipes, the one with the highest causal switching mean is
selected. This rule selects a development recipe; it is not a confirmatory
efficacy claim. The frozen estimator is not retrained, and ESCP/RE-SAC are not
retrained in this diagnostic.
