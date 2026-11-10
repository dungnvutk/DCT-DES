"""Illustrative deterministic hospital-queue comparison, not paper validation."""
import json
import heapq
from collections import defaultdict
from .paper_reproduction import solve, survival

def replay(schedule, severity, teams=2, immediate_minutes=35, delayed_minutes=20):
    slots={h:[0.0]*teams for h in range(7)}
    for h in slots:heapq.heapify(slots[h])
    score=0.0; waits=[]
    for p in sorted(schedule,key=lambda x:(x["arrival"],x["hospital"],x["class"],x["departure"])):
        h=p["hospital"]; arrival=p["arrival"]
        start=max(arrival,heapq.heappop(slots[h]))
        heapq.heappush(slots[h],start+(immediate_minutes if p["class"]==0 else delayed_minutes))
        waits.append(start-arrival)
        score+=survival(p["class"],start,severity)
    return {"expected_survivors":round(score,5),
            "mean_wait":round(sum(waits)/len(waits),3) if waits else 0,
            "max_wait":round(max(waits),3) if waits else 0}

def forecast_wait(schedule,teams=2,immediate_minutes=35,delayed_minutes=20):
    grouped=defaultdict(list)
    for p in schedule:grouped[p["hospital"]].append(p)
    forecast=[]
    for h in range(7):
        booked=sorted(grouped[h],key=lambda p:(p["arrival"],p["departure"]))
        slots=[0.0]*teams;heapq.heapify(slots);k=0;waits=[]
        for t in range(240):
            while k<len(booked) and booked[k]["arrival"]<=t:
                p=booked[k];k+=1
                start=max(p["arrival"],heapq.heappop(slots))
                heapq.heappush(slots,start+(immediate_minutes if p["class"]==0 else delayed_minutes))
            waits.append(max(0,min(slots)-t))
        forecast.append(waits)
    return forecast

def compare(scenario,severity,teams=2):
    if teams<=0:raise ValueError("teams must be positive")
    immediate=solve(scenario,severity,"immediate_first")
    dct=solve(scenario,severity,"dct")
    selected=dct
    highest=replay(selected["schedule"],severity,teams)["expected_survivors"]
    iterations=[]
    for step in range(3):
        waiting=forecast_wait(selected["schedule"],teams)
        candidate=solve(scenario,severity,"dct",wait_forecast=waiting)
        observed=replay(candidate["schedule"],severity,teams)["expected_survivors"]
        iterations.append({"iteration":step+1,"expected_survivors":observed})
        if observed>highest+1e-7:
            highest=observed;selected=candidate
        else:break
    return {"scenario":scenario,"severity":severity,"teams_per_hospital":teams,
            "transport_immediate":immediate["expected_survivors"],
            "transport_dct":dct["expected_survivors"],
            "queue_immediate":replay(immediate["schedule"],severity,teams),
            "queue_dct":replay(dct["schedule"],severity,teams),
            "queue_informed":replay(selected["schedule"],severity,teams),
            "iterations":iterations}

if __name__=="__main__":
    for scenario in (1,2):
        for severity in ("low","moderate","high"):
            print(json.dumps(compare(scenario,severity)),flush=True)
