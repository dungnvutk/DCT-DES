# DCT-DES: Dynamic Casualty Transportation + Discrete-Event Simulation

**Research prototype — not for clinical, emergency, or operational dispatch use.**

This repository contains a minimal runnable simulation–optimization framework inspired by the casualty-transportation research question of Kilanitis and collaborators. It is **not a reproduction of their exact DCT MILP**: it uses a smaller, rolling decision-epoch binary MILP and synthetic clinical parameters.

## Features

- Synthetic multi-site casualty arrivals with immediate/delayed triage classes
- Reusable ambulances, collection-point repositioning delays, stochastic outbound and return travel
- Two receiving hospitals with limited admissions and staffed treatment teams
- Event-driven hospital arrivals, scheduled treatment starts, treatment completions and ambulance returns
- Rolling MILP dispatch solved with SciPy/HiGHS, or an immediate-first comparison policy
- Fixed-seed policy comparisons across stochastic replications
- Expected-survivor metric computed using **illustrative, unvalidated** exponential time-to-treatment survival curves

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[test]"
pytest -q
```

## Run policy comparison

```bash
python -m dct_des --replications 20 --seed 42 --travel-cv 0.15
```

Run only the MILP policy:

```bash
python -m dct_des --policy milp --replications 10
```

For patient-level event records:

```python
from dct_des import run_simulation
result = run_simulation(policy="milp", seed=42)
print(result["expected_survivors"])
print(result["events"][:5])
```

## Modeling assumptions

1. Each ambulance transports one casualty per trip, then returns to the original collection point.
2. Ambulances can reposition between collection points using a fixed 15-minute penalty. This is a simplification.
3. Hospital capacity means total admissions over the simulation horizon, not instantaneous beds. Each hospital maintains a future team-availability heap to schedule treatment starts.
4. New casualty arrivals and ambulance returns trigger MILP reruns; previously dispatched missions are not canceled.
5. The MILP chooses feasible ambulance/patient/hospital triples to maximize estimated **marginal survival benefit** versus a 25-minute delay, based on the earliest currently reserved treatment-team availability. It is a surrogate, **not** the exact mathematical objective in the original paper.
6. Arrival, transport, and treatment events run in continuous simulated time; survival at treatment start is used for the research metric.
7. Hospital treatment times and casualty arrivals are synthetic. Treatment durations are deterministic, while travel is random with configurable coefficient of variation.
8. There is no realistic patient-specific treatment selection, clinical deterioration process, prehospital intervention, operating room, or staff shift model. The immediate-first comparator is itself implemented by the same allocation MILP with priority weights and is not independently validated.

## Extending the framework

- Replace the synthetic survival model with validated injury-specific functions.
- Add scenario files, explicit beds, staff schedules, operating rooms, and specialty resource compatibility.
- Model true patient queues and preemption / urgency-based scheduling.
- Estimate hospital waiting times from DES or sample-average approximation and incorporate them into a richer time-indexed MILP.
- Include alternative discharge, transfer, cancellation, and response policies.
- Add CI, larger-scenario stress tests, confidence intervals, and trace-level validations.

### Reference

Kilanitis et al., *Optimal Transportation of Casualties to Hospitals after Multi-Disaster Mass Casualty Incidents*. Please consult the original publication and the authors' code when claiming methodological replication.

**Do not interpret simulated expected survivors as actual clinical survival predictions.**
