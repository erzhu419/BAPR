# V19 specialist stability diagnostic

All required v19 producers, amended CPU audits, and the aggregate are complete.
The ten failed scheduler records are superseded lineages: two specialist
producers were replaced, and eight audits were rerun through amendment 1 after
their original entry point tried to read a GPU-local protocol signature. No
scientific arm is missing.

The preregistered result is negative. `full_state` improves mean safe switching
return over `actor_only_control` by `+262.2`, wins `2/3` policy seeds and `7/9`
switching events, raises the worst-seed robust-relative gain from `+43.3%` to
`+53.8%`, and passes the robust-relative cell gate on all three seeds. It still
fails selection because its minimum stationary specialist retention relative
to actor-only is `85.2%`, below the frozen `95%` requirement. `critic_warmup`
is weaker: `+63.5` mean switching delta, `1/3` seed wins, `3/9` event wins, and
a lower worst-seed gain (`+35.2%`). It is closed.

The failure is localized rather than a missing adaptation signal. Full-state
initialization passes all `12/12` robust-relative stationary mode cells and
enables all four specialists in every seed. Its losses against actor-only are
concentrated in seed `82021` modes 2 and 3 (`85.2%` and `90.9%` retention),
seed `82039` mode 2 (`91.8%`), and seed `82003` mode 3 (`91.6%`). Pooled across
all seeds, full-state is stronger in modes 0, 1, and 2 and weaker only in mode
3.

Remote training logs show that a copied critic often gives full-state a much
stronger controller immediately after the fork, while subsequent actor updates
can erode that behavior. For example, seed `82021` mode 2 starts its logged
fine-tuning evaluations far above actor-only, then declines substantially over
the remaining updates. A fresh-critic warmup does not repair this, so the next
question is actor-update stability after a valid full-controller warm start,
not another posterior, gate, residual-cap, or environment change.

The next fresh-seed screen retains the v19 decision rule and compares final
full-state training with validation-selected full-state policies and a
validation-selected arm whose actor and temperature update every second critic
step. The v18 and v19 holdouts remain untouched.
