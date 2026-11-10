"""Reproduce Kilanitis et al. (2026) six MILP scenarios (SciPy/HiGHS).
Run: python -m research.paper_reproduction
This is a research benchmark, not operational medical software.
"""
import json
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

TRAVEL = {
  1: np.array([[7,5],[11,12],[5,6],[5,6],[4,5],[5,4],[8,9]]).T,
  2: np.array([[18,25],[21,23],[15,21],[15,22],[15,21],[16,22],[16,17]]).T,
}
CAP = [4,6,3,8,3,8,8]
PARAMS = {
  'low':[(.56,91,1.58),(.81,160,2.41)],
  'moderate':[(.24,47,1.3),(.76,138,2.17)],
  'high':[(.09,17,1.01),(.57,61,2.03)],
}
SETTINGS = {'low':(5,15,21),'moderate':(10,10,14),'high':(15,5,7)}

def survival(s, t, severity):
    b0,b1,b2=PARAMS[severity][s]
    return b0/(1+(t/b1)**b2)

def solve(scenario=1, severity='high', policy='dct', horizon=240,
          time_limit=30, wait_forecast=None):
    """Time-indexed integer casualty flows, return flows and initial asset allocation.

    wait_forecast is an optional [7, 240] hospital congestion lookup. Without
    it the objective exactly matches the transportation-only survival equation.
    """
    d=TRAVEL[scenario]
    immediate,delayed,number_ambulances=SETTINGS[severity]
    xs=[];ars=[];fs=[];objective=[];upper=[]
    for t in range(horizon+1):
        for i in range(2):
            for h in range(7):
                for s in range(2):
                    xs.append((i,h,s,t))
                    delay=wait_forecast[h][min(239,t+int(d[i,h]))] if wait_forecast is not None else 0
                    score=survival(s,t+d[i,h]+delay,severity)
                    weight=(1000 if s==0 else .01) if policy=='immediate_first' else 1
                    objective.append(-weight*score)
                    upper.append(min(immediate if s==0 else delayed,number_ambulances))
    nx=len(xs)
    for t in range(horizon+1):
        for i in range(2):
            for h in range(7):
                ars.append((i,h,t));objective.append(0)
                upper.append(number_ambulances if t>=d[i,h] else 0)
    na=len(ars)
    for a in range(number_ambulances):
        origin=a%7
        for i in range(2):
            fs.append((a,i,int(d[i,origin])));objective.append(0);upper.append(1)
    ix={v:j for j,v in enumerate(xs)}
    ia={v:nx+j for j,v in enumerate(ars)}
    iff={v:nx+na+j for j,v in enumerate(fs)}
    rows=[];cols=[];vals=[];lower=[];limits=[]
    def row(terms,low,high):
        k=len(lower);lower.append(low);limits.append(high)
        for j,value in terms:
            rows.append(k);cols.append(j);vals.append(value)
    for h in range(7):
        row(((ix[x],1) for x in xs if x[1]==h),-np.inf,CAP[h])
    for i in range(2):
        for s in range(2):
            row(((ix[x],1) for x in xs if x[0]==i and x[2]==s),
                -np.inf,immediate if s==0 else delayed)
    for a in range(number_ambulances):
        row(((iff[z],1) for z in fs if z[0]==a),1,1)
    for t in range(horizon+1):
        for i in range(2):
            terms=[]
            for h in range(7):
                for s in range(2):terms.append((ix[(i,h,s,t)],1))
                terms.append((ia[(i,h,t)],-1))
            for a in range(number_ambulances):
                z=(a,i,t)
                if z in iff:terms.append((iff[z],-1))
            row(terms,-np.inf,0)
    for h in range(7):
        for u in range(horizon+1):
            terms=[]
            for i in range(2):
                t=u-int(d[i,h])
                if t>=0:
                    for s in range(2):terms.append((ix[(i,h,s,t)],-1))
                ret=u+int(d[i,h])
                if ret<=horizon:terms.append((ia[(i,h,ret)],1))
            if terms:row(terms,-np.inf,0)
    matrix=coo_matrix((vals,(rows,cols)),
                      shape=(len(lower),len(objective))).tocsr()
    start=time.monotonic()
    result=milp(np.array(objective),
        integrality=np.ones(len(objective)),
        bounds=Bounds(np.zeros(len(objective)),np.array(upper)),
        constraints=LinearConstraint(matrix,np.array(lower),np.array(limits)),
        options={'time_limit':time_limit,'mip_rel_gap':0.0})
    runtime=time.monotonic()-start
    if result.x is None:
        raise RuntimeError(f'No feasible dispatch plan: {result.message}')
    schedule=[{'site':i,'hospital':h,'class':s,'departure':t,
               'arrival':t+int(d[i,h])}
              for j,(i,h,s,t) in enumerate(xs)
              for _ in range(int(round(result.x[j])))]
    expected=sum(survival(p['class'],p['arrival'],severity) for p in schedule)
    return {'scenario':scenario,'severity':severity,'policy':policy,
            'status':result.message,'seconds':round(runtime,2),
            'expected_survivors':round(expected,5),
            'transported':len(schedule),'mip_gap':float(result.mip_gap),
            'schedule':schedule}

if __name__=='__main__':
    for scenario in (1,2):
        for severity in ('low','moderate','high'):
            for policy in ('immediate_first','dct'):
                result=solve(scenario,severity,policy)
                result.pop('schedule')
                print(json.dumps(result))
