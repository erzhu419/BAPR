# Conflict-fallback router holdout confirmation preregistration

V15 freezes the v14 confirm-3 router without further parameter selection. The
controller banks, robust-inclusive utility maps, expected-action estimator,
conflict margin 3.0, posterior exit confidence 0.80, and minimum one robust
fallback action are unchanged. Three unused event seeds provide new explicit
switch orders.

The primary strict gate is unchanged: the selected router must gain at least
10% over matched robust SAC, recover at least 70% of safe-oracle headroom, win
all three events, and have zero termination on at least four of five policy
seeds. The causal-margin prerequisite also remains at four seeds under the
four-step privileged delayed oracle.

Because v14 selected this router against plain posterior MAP, confirmation also
requires the frozen router to beat plain MAP on at least four of five seeds and
have a positive paired mean return difference. No parameter is changed based
on v15 results. Passing v15 authorizes a final fresh-policy-bank comparison;
failure returns the project to switch-focused estimator training rather than
another router threshold sweep.
