import numpy as np, json, time
from math import comb
from scipy.optimize import linprog
import doelp
rng=np.random.default_rng(20261110)
sizes=[2,3,4,5,6,8,10,15,20,30,40,50]
R=20
out=[]
for n in sizes:
    m=n  # square-ish; also m = 2n below
    for mfac in (1,2):
        m=mfac*n
        rec=dict(n=n,m=m,ok_cen=0,stc=[],t_cen=[],ok_csa=0,ok_evd=0,evd_done=0,st=[],pv=[],runs=[],sys=[],t_csa=[],t_sx=[],t_evd=[],t_hi=[])
        k=0
        while k<R:
            A=rng.uniform(-0.2,1.0,(m,n)); A[A<0.05]=0.0  # sparse-ish, some negatives
            A[rng.integers(0,m,n),np.arange(n)]=rng.uniform(.2,1,n)  # each column has a positive entry
            b=rng.uniform(1,10,m); c=rng.uniform(1,10,n)
            hi=linprog(-c,A_ub=A,b_ub=b,method="highs")
            if hi.status!=0: continue
            U=np.full(n,1e4)
            t=time.perf_counter(); cs=doelp.csa_solve(c,A,b,U=U); t1=time.perf_counter()-t
            t=time.perf_counter(); sx=doelp.simplex_tableau(c,A,b); t2=time.perf_counter()-t
            s0=0.5*b.min()/np.maximum(A,0).sum(1).max(); x0=np.full(n,s0)
            t=time.perf_counter(); cc=doelp.csa_solve(c,A,b,U=U,x0=x0); t4=time.perf_counter()-t
            rec["ok_cen"]+= abs(cc["z"]+hi.fun)<=1e-6*max(1,abs(hi.fun)); rec["stc"].append(cc["stages2"]); rec["t_cen"].append(t4)
            rec["ok_csa"]+= abs(cs["z"]+hi.fun)<=1e-6*max(1,abs(hi.fun))
            rec["st"].append(cs["stages2"]+cs["stages1"]); rec["pv"].append(sx["pivots1"]+sx["pivots2"])
            rec["t_csa"].append(t1); rec["t_sx"].append(t2)
            nsys=comb(m+2*n,n)
            if nsys<=3e4:
                t=time.perf_counter(); ev=doelp.evd_solve(c,A,b,U=U); t3=time.perf_counter()-t
                rec["evd_done"]+=1; rec["ok_evd"]+= int(abs(ev["z"]+hi.fun)<=1e-6*max(1,abs(hi.fun)))
                rec["runs"].append(ev["runs"]); rec["sys"].append(ev["systems"]); rec["t_evd"].append(t3)
            k+=1
        s={kk:(float(np.mean(v)) if isinstance(v,list) and v else v) for kk,v in rec.items()}
        s["stc_max"]=int(max(rec["stc"])); s["st_max"]=int(max(rec["st"])); s["pv_max"]=int(max(rec["pv"]))
        s["nsys_theory"]=float(comb(m+2*n,n)); s={k:(int(v) if isinstance(v,(np.integer,)) else v) for k,v in s.items()}; s["same_path"]=int(sum(a==b for a,b in zip(rec["st"],rec["pv"])))
        out.append(s); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items()},flush=True)
json.dump(out,open("results/e2.json","w"),indent=1)
