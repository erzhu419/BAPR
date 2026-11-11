# Compact publication-extension evidence

`summary.json` contains all 24 policy audits reduced to per-episode returns,
mode accuracy, detected delays and unresolved-switch counts, plus completed
budgets, source-trainer configuration and paired intervals. It excludes states,
calibration arrays, replay buffers and checkpoints. This package reproduces
the statistics, not fresh environment rollouts or model training.

Three estimator seeds are crossed factors; five policy seeds are the independent
units in each Reacher/source study. The nine-policy HalfCheetah audit is a
historical, different-recipe sensitivity study. Source actor opportunities
are 262,500 versus BAPR's 525,000 updates; estimator training costs are separate.
The frozen protocol and detailed interpretation are in the two V47 reports.
No historical cohorts or acceptance decisions are replaced.

From the repository root, `python scripts/check_manuscript_results.py` validates
this package alongside the original confirmation packages, recomputing all
switching contrasts and checking the registered cohorts and completed budgets.

With SciPy installed, recompute every switching contrast from this package:

```python
import json
import math
from pathlib import Path
import statistics as st
from scipy.stats import t

data = json.loads(Path("summary.json").read_text())
for suite, study in data["suites"].items():
    for name, saved in study["comparisons"].items():
        left, right = name.split("-minus-")
        delta = [st.fmean(a["routes"][left]["switching_returns"])
                 - st.fmean(a["routes"][right]["switching_returns"])
                 for a in study["audits"]]
        mean = st.fmean(delta)
        radius = t.ppf(.975, len(delta) - 1) * st.stdev(delta) / math.sqrt(len(delta))
        assert abs(mean - saved["mean"]) < 1e-8
        assert max(abs(a-b) for a, b in zip(
            [mean-radius, mean+radius], saved["ci95"])) < 1e-8
        assert sum(x > 0 for x in delta) == saved["wins"]
        print(suite, name, mean, [mean-radius, mean+radius], saved["wins"])
```

The repository's `scripts/summarize_submission_extensions_v47.py` rebuilds
this package from the original local audits and validates protocol coverage.
Detection-delay medians must include the unresolved counts alongside them;
the medians alone describe only successfully detected switches.
