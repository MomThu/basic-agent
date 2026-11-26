"""Make a small fake microservice dataset with known answers, for the mini agent.

The system:
    frontend -> checkout -> payment
    frontend -> catalog
    checkout -> db

Each case is one hour (00:00-00:59) with one row per service per minute: cpu (%), memory (%),
latency (ms). One fault starts at a random minute between 00:20 and 00:35.

Faults (what changes on the faulty service):
    CPU fault      cpu +50, latency +60
    memory fault   memory +35, latency +40
    network delay  latency +250 (cpu and memory unchanged)

Propagation: when a service gets slower, each service that calls it gets slower 1-2 minutes
later, by 1.3x the extra time (it waits for the answer and retries).

Traps for the agent:
    1. Symptom trap: the caller at the top (frontend) shows the BIGGEST latency increase,
       but it is only a victim. The root cause is the service whose problem starts FIRST.
    2. Noise trap: every value has random noise, and in each case one healthy service has a
       short 2-minute cpu spike before the real fault starts.

15 cases = 5 services x 3 reasons, so every component and every reason appears exactly once.

Output: data/system.json (who calls whom), data/case_01.csv ... case_15.csv, data/truth.json

Run:  python make_data.py
"""

import json
import os
import random

import pandas as pd

CALLS = {"frontend": ["checkout", "catalog"], "checkout": ["payment", "db"],
         "payment": [], "catalog": [], "db": []}
CALLERS = {s: [c for c, callees in CALLS.items() if s in callees] for s in CALLS}

# normal cpu %, memory %, latency ms
NORMAL = {"frontend": (30, 45, 120), "checkout": (25, 50, 80), "payment": (20, 40, 40),
          "catalog": (15, 35, 30), "db": (35, 60, 15)}          
# noise (std) of cpu, memory, latency
NOISE = (2.0, 1.5, 5.0)                                          
EFFECT = {"CPU fault": (50, 0, 60), "memory fault": (0, 35, 40), "network delay": (0, 0, 250)}
MINUTES = 60


def make_case(root, reason, rng):
    """
    Make one case with a fault on the given root service, for the given reason.
    Return a dataframe with the metrics, and the ground truth.
    """
    # minute the fault starts
    start = rng.randint(20, 35)                                   
    extra = {s: {"cpu": [0.0] * MINUTES, "memory": [0.0] * MINUTES, "latency": [0.0] * MINUTES}
             for s in CALLS}

    # 1. the fault on the root service
    d_cpu, d_mem, d_lat = EFFECT[reason]
    for t in range(start, MINUTES):
        extra[root]["cpu"][t] += d_cpu
        extra[root]["memory"][t] += d_mem
        extra[root]["latency"][t] += d_lat

    # 2. propagation: callers get slower 1-2 minutes later, by 1.3x the extra latency
    todo = [root]
    while todo:
        s = todo.pop()
        for caller in CALLERS[s]:
            lag = rng.choice([1, 2])
            for t in range(lag, MINUTES):
                extra[caller]["latency"][t] += 1.3 * extra[s]["latency"][t - lag]
            todo.append(caller)

    # 3. noise trap: a short cpu spike on a healthy service, before the fault
    decoy = rng.choice([s for s in CALLS if s != root])
    t0 = rng.randint(10, start - 4)
    for t in (t0, t0 + 1):
        extra[decoy]["cpu"][t] += 25

    # 4. normal level + noise + extra
    rows = []
    for t in range(MINUTES):
        for s in CALLS:
            values = [max(0.0, base + rng.gauss(0, sd) + extra[s][k][t])
                      for base, sd, k in zip(NORMAL[s], NOISE, ("cpu", "memory", "latency"))]
            rows.append({"time": f"00:{t:02d}", "service": s, "cpu": round(values[0], 1),
                         "memory": round(values[1], 1), "latency": round(values[2], 1)})
    truth = {"component": root, "reason": reason, "time": f"00:{start:02d}"}
    return pd.DataFrame(rows), truth


def main(out="data", seed=42):
    rng = random.Random(seed)                                     # fixed seed: same data every time
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "system.json"), "w") as f:
        json.dump(CALLS, f, indent=2)
    pairs = [(s, r) for s in CALLS for r in EFFECT]               # all 15 combinations
    rng.shuffle(pairs)
    truth = {}
    for i, (root, reason) in enumerate(pairs, 1):
        df, truth[f"case_{i:02d}"] = make_case(root, reason, rng)
        df.to_csv(os.path.join(out, f"case_{i:02d}.csv"), index=False)
    with open(os.path.join(out, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print(f"wrote {len(truth)} cases to {out}/")


if __name__ == "__main__":
    main()