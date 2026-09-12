# Hopper and Walker2d structured-channel headroom result

This is an environment-capacity screen, not a learned BAPR result. 
Each role has the same 5.6M-step budget; the oracle receives the true 
mode and the robust arm receives zero context.

| Env | Robust / oracle switch | Gain | Robust / oracle worst | Gain | Mode wins | Seed wins switch/worst | Max termination | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Hopper | 2424.0 / 2463.9 | +1.6% | 127.7 / 167.7 | +31.3% | 2/4 | 2/3 | 100.0% | FAIL |
| Walker2d | 2186.7 / 2141.0 | -2.1% | 162.3 / 149.8 | -7.7% | 2/4 | 1/1 | 100.0% | FAIL |

## Decision

Passing environments: `[]`.

No BAPR transfer is authorized; retain both environments as headroom or survival negatives without severity tuning.
