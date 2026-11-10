"""Event-driven evacuation simulator with rolling-horizon MILP dispatch.

Prototype only: synthetic survival curves, synthetic data, not a clinical tool.
Times are minutes. The model is independent of the Kilanitis paper's code.
"""
from dataclasses import dataclass, field
import heapq
import math
import random

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix


@dataclass
class Patient:
    id: str
    site: str
    arrival: float
    severity: str
    duration: float
    status: str = "not_arrived"
    hospital_arrival: float | None = None
    treatment_start: float | None = None


@dataclass
class Ambulance:
    id: str
    site: str
    available: bool = True
    trips: int = 0


@dataclass
class Hospital:
    id: str
    teams: int
    capacity: int
    team_available: list[float] = field(default_factory=list)
    admissions: int = 0

    def __post_init__(self):
        if self.teams <= 0 or self.capacity <= 0:
            raise ValueError("Hospital teams and capacity must be positive")
        self.team_available = [0.0] * self.teams
        heapq.heapify(self.team_available)


def survival(severity, treatment_start, arrival):
    """Illustrative time-to-treatment survival, NOT calibrated to clinical data."""
    rates = {"immediate": 0.035, "delayed": 0.012}
    return math.exp(-rates[severity] * max(0.0, treatment_start - arrival))


def default_scenario():
    return {
        "patients": [
            {"id": f"P{i}", "site": "S1" if i % 2 else "S2",
             "arrival": float((i // 6) * 12),
             "severity": "immediate" if i % 3 == 0 else "delayed",
             "duration": 28.0 if i % 3 == 0 else 15.0}
            for i in range(1, 25)
        ],
        "ambulances": [
            {"id": f"A{i}", "site": "S1" if i % 2 else "S2"}
            for i in range(1, 7)
        ],
        "hospitals": [
            {"id": "H1", "teams": 2, "capacity": 14},
            {"id": "H2", "teams": 3, "capacity": 18},
        ],
        "travel": {"S1": {"H1": 12.0, "H2": 22.0},
                   "S2": {"H1": 24.0, "H2": 11.0}},
    }


def _dispatch(now, waiting, free, hospitals, travel, policy):
    """Binary MILP: select ambulance/patient/hospital triples.

    Constrains ambulance, patient and hospital capacity per decision epoch.
    """
    if not waiting or not free:
        return []
    choices = []
    for ai, ambulance in enumerate(free):
        for pi, patient in enumerate(waiting):
            for hi, hospital in enumerate(hospitals):
                if hospital.admissions >= hospital.capacity:
                    continue
                reposition = 0 if ambulance.site == patient.site else 15
                eta = now + reposition + travel[patient.site][hospital.id]
                start = max(eta, hospital.team_available[0])
                if policy == "milp":
                    # Estimated marginal loss avoided by dispatching now.
                    score = (survival(patient.severity, start, patient.arrival)
                             - survival(patient.severity, start + 25, patient.arrival))
                elif policy == "immediate_first":
                    score = (100 if patient.severity == "immediate" else 10) - 0.001 * eta
                else:
                    raise ValueError("Unknown policy")
                choices.append((ai, pi, hi, score))
    if not choices:
        return []
    n = len(choices)
    m = len(free) + len(waiting) + len(hospitals)
    matrix = lil_matrix((m, n), dtype=float)
    for idx, (ai, pi, hi, _) in enumerate(choices):
        matrix[ai, idx] = 1
        matrix[len(free) + pi, idx] = 1
        matrix[len(free) + len(waiting) + hi, idx] = 1
    limits = [1] * (len(free) + len(waiting))
    limits += [h.capacity - h.admissions for h in hospitals]
    result = milp(
        -np.array([c[3] for c in choices]),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), 0, limits),
        options={"time_limit": 10.0},
    )
    if result.x is None:
        raise RuntimeError(f"Dispatch MILP failed: {result.message}")
    return [(free[ai], waiting[pi], hospitals[hi])
            for selected, (ai, pi, hi, _) in zip(result.x, choices)
            if selected > 0.5]


def run_simulation(scenario=None, policy="milp", seed=1, travel_cv=0.15,
                   horizon=480.0):
    """Event queue handles arrivals, transport, ambulance reuse and treatment.

    Reoptimizes when patient arrivals or ambulance returns alter availability.
    Queues are represented by reserved future trauma-team time slots.
    """
    if policy not in ("milp", "immediate_first"):
        raise ValueError("Unknown policy")
    if not 0 <= travel_cv <= 1 or horizon <= 0:
        raise ValueError("Invalid travel coefficient of variation or horizon")
    data = default_scenario() if scenario is None else scenario
    patients = [Patient(**p) for p in data["patients"]]
    ambulances = [Ambulance(**a) for a in data["ambulances"]]
    hospitals = [Hospital(**h) for h in data["hospitals"]]
    travel = data["travel"]
    if any(p.arrival < 0 or p.duration <= 0 for p in patients):
        raise ValueError("Invalid patient arrival or duration")
    rng = random.Random(seed)
    events, log = [], []
    sequence = 0

    def schedule(t, event_type, payload):
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (t, sequence, event_type, payload))

    def journey(average):
        if average < 0:
            raise ValueError("Negative travel time")
        if travel_cv == 0:
            return average
        sigma = math.sqrt(math.log(1 + travel_cv ** 2))
        return average * rng.lognormvariate(-sigma * sigma / 2, sigma)

    for patient in patients:
        schedule(patient.arrival, "release", patient)

    while events:
        now, _, kind, obj = heapq.heappop(events)
        if now > horizon:
            break
        if kind == "release":
            obj.status = "waiting"
            log.append({"time": now, "event": "casualty_arrival", "patient": obj.id})
        elif kind == "hospital_arrival":
            patient, ambulance, hospital, origin = obj
            patient.hospital_arrival = now
            patient.status = "hospital"
            free_at = heapq.heappop(hospital.team_available)
            start = max(now, free_at)
            patient.treatment_start = start
            heapq.heappush(hospital.team_available, start + patient.duration)
            log.append({"time": now, "event": "hospital_arrival",
                        "patient": patient.id, "hospital": hospital.id,
                        "treatment_start": start})
            schedule(start, "treatment_start", patient)
            schedule(start + patient.duration, "treatment_end", patient)
            schedule(now + journey(travel[origin][hospital.id]),
                     "ambulance_return", (ambulance, origin))
        elif kind == "ambulance_return":
            ambulance, origin = obj
            ambulance.available = True
            ambulance.site = origin
            log.append({"time": now, "event": "ambulance_return",
                        "ambulance": ambulance.id})
        elif kind == "treatment_start":
            obj.status = "treating"
            log.append({"time": now, "event": "treatment_start", "patient": obj.id})
        elif kind == "treatment_end":
            obj.status = "treated"
            log.append({"time": now, "event": "treatment_end", "patient": obj.id})

        waiting = [p for p in patients if p.status == "waiting"]
        available = [a for a in ambulances if a.available]
        for ambulance, patient, hospital in _dispatch(
                now, waiting, available, hospitals, travel, policy):
            ambulance.available = False
            ambulance.trips += 1
            patient.status = "in_transit"
            hospital.admissions += 1
            reposition = 0 if ambulance.site == patient.site else 15
            t = now + reposition + journey(travel[patient.site][hospital.id])
            schedule(t, "hospital_arrival", (patient, ambulance, hospital, patient.site))
            log.append({"time": now, "event": "dispatch",
                        "patient": patient.id, "ambulance": ambulance.id,
                        "hospital": hospital.id})

    treated = [p for p in patients
               if p.treatment_start is not None and p.treatment_start <= horizon]
    return {
        "policy": policy, "seed": seed, "patients": len(patients),
        "treated": len(treated),
        "expected_survivors": round(sum(
            survival(p.severity, p.treatment_start, p.arrival) for p in treated), 4),
        "mean_wait_minutes": round(sum(
            p.treatment_start - p.hospital_arrival for p in treated) / len(treated), 3)
        if treated else None,
        "ambulance_trips": sum(a.trips for a in ambulances),
        "events": log,
    }
