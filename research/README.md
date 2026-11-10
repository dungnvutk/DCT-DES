# Published DCT reproduction and hospital-queue extension

## Run

Install the dependencies from the repository root:

```bash
python -m pip install -e ".[test]"
python -m research.paper_reproduction
python -m research.queue_comparison
```

The first script implements a 240-minute, time-indexed **integer-flow MILP**: casualty transport counts `x`, return availability `a`, and initial ambulance placement `f`. The objective applies the published class-specific survival function at **hospital arrival**; the first-immediate comparator uses the source-code prioritization weights (1000, 0.01). The model uses the published six configurations (2 sites, 7 hospitals, 40 casualties, 21/14/7 ambulances), per-hospital capacities [4,6,3,8,3,8,8], and the published scenario travel-time matrices. SciPy/HiGHS replaces the Windows GLPK executable in the authors' reference implementation.

The second script **replays the optimized schedules through a simplified event-driven treatment-availability model**. It assumes two treatment teams per hospital, deterministic 35-minute treatment for immediate casualties and 20 minutes for delayed casualties, first-available-team scheduling and the same published survival functions evaluated at *treatment start* (rather than hospital arrival). Those treatment assumptions are NOT present in the original paper and MUST NOT be described as clinically validated. The queue-aware alternative computes hospital-wait estimates from the current schedule, resolves the MILP using arrival+estimated-wait survival coefficients, and keeps only iterations that improve deterministic replay. This is an iterative sensitivity analysis, not a globally optimal integrated hospital-care MILP.

## Initial executed comparison

See [results/initial_comparison.csv](results/initial_comparison.csv). All scores are expected survivors out of 40 casualties, **not** empirical survival counts.

| Scenario | Severity | Original transport DCT | DCT replay with queues | Queue-informed replay |
|---|---|---:|---:|---:|
| Urban | Low | 29.60557 | 27.33212 | 27.33212 |
| Urban | Moderate | 18.74686 | 15.65685 | 16.25148 |
| Urban | High | 6.30576 | 5.65773 | 5.65773 |
| Highway | Low | 26.88252 | 24.23183 | 24.23183 |
| Highway | Moderate | 15.43045 | 14.28552 | 14.28552 |
| Highway | High | 4.22435 | 3.60263 | 3.84037 |

Reproduction of the authors' aggregate DCT scores matches the published two-decimal results. **The queue-aware numbers depend heavily on the invented team counts and treatment durations** and are therefore exploratory. All replay results are deterministic; there are no confidence intervals or stochastic replications in this table. It would be misleading to call the queue-informed improvement a demonstrated clinical benefit.

For research-grade evidence, validate the patient-care resource assumptions with clinical experts, re-run with varied team staffing and treatment-time distributions, validate actual ambulance itineraries, introduce dynamic arrival/capacity events and multiple stochastic replications, and report confidence intervals plus solver gaps.

## Original source
Kilanitis et al., *Optimal Transportation of Casualties to Hospitals after Multi-Disaster Mass Casualty Incidents*, and https://github.com/stathiscerth/ambulance_routing_model .
