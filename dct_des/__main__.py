import argparse
import json
from statistics import mean
from .model import run_simulation


def main():
    parser = argparse.ArgumentParser(description="DCT-DES research prototype")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--travel-cv", type=float, default=0.15)
    parser.add_argument("--policy", choices=["both", "milp", "immediate_first"], default="both")
    args = parser.parse_args()
    if args.replications < 1:
        parser.error("replications must be >= 1")
    policies = ["immediate_first", "milp"] if args.policy == "both" else [args.policy]
    summary = {}
    for policy in policies:
        runs = [run_simulation(policy=policy, seed=args.seed + i,
                               travel_cv=args.travel_cv)
                for i in range(args.replications)]
        summary[policy] = {
            "replications": len(runs),
            "mean_expected_survivors": round(mean(r["expected_survivors"] for r in runs), 3),
            "mean_treated": round(mean(r["treated"] for r in runs), 2),
            "mean_ambulance_trips": round(mean(r["ambulance_trips"] for r in runs), 2),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
