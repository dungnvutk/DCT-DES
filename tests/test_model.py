import pytest
from dct_des import run_simulation, default_scenario


def test_deterministic_and_unique_dispatches():
    result = run_simulation(seed=3, travel_cv=0)
    assert result == run_simulation(seed=3, travel_cv=0)
    dispatched = [e["patient"] for e in result["events"] if e["event"] == "dispatch"]
    assert len(dispatched) == len(set(dispatched))
    assert result["ambulance_trips"] == len(dispatched)


@pytest.mark.parametrize("policy", ["milp", "immediate_first"])
def test_policy_feasible(policy):
    scenario = default_scenario()
    result = run_simulation(scenario, policy=policy, travel_cv=0)
    assert 0 <= result["expected_survivors"] <= len(scenario["patients"])
    assert result["treated"] <= len(scenario["patients"])
    assert result["mean_wait_minutes"] is None or result["mean_wait_minutes"] >= 0
    dispatches = [e for e in result["events"] if e["event"] == "dispatch"]
    for hospital in scenario["hospitals"]:
        assert sum(e["hospital"] == hospital["id"] for e in dispatches) <= hospital["capacity"]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        run_simulation(travel_cv=-0.1)
    with pytest.raises(ValueError):
        run_simulation(policy="unknown", travel_cv=0)
