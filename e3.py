"""E3: unknown objective coefficients observed with noise. Constraints known.
S1 pick-the-winner on extreme vertices; S2 DOE model-based (EVD + centroid, OLS, CSA);
S3 DOE model-based with a D-optimal subset of vertices (same run budget)."""
import numpy as np, json
import doelp, problems as pr
rng=np.random.default_rng(7)
SIMS=2000
names=["P1 Reddy Mikks (Ex. 2.1-1)","P8 Production (Ex. 2.4-2)","P7 Bank loan (Ex. 2.4-1)"]
sigmas=[0.01,0.05,0.10,0.20,0.40]
reps=[1,2,4]

def dopt_subset(C, size, it=200):
    D=np.column_stack([np.ones(len(C)),C]); N=len(C)
    idx=list(range(min(size,N)))
    best=np.linalg.det(D[idx].T@D[idx]) if size>=D.shape[1] else 0
    # greedy exchange
    improved=True
    while improved:
        improved=False
        for a in range(len(idx)):
            for j in range(N):
                if j in idx: continue
                t=idx.copy(); t[a]=j
                d=np.linalg.det(D[t].T@D[t])
                if d>best*(1+1e-9):
                    best=d; idx=t; improved=True
    return idx

out=[]
for name in names:
    p=pr.get(name); c=p["c"]; k=len(c)
    ev=doelp.evd_solve(c,p["G"],p["h"],p["E"],p["e"])
    V=ev["vertices"]; zv=V@c; zs=zv.max(); zmin=zv.min(); rng_z=zs-zmin
    cen=V.mean(0); C=np.vstack([V,cen]); N0=len(C)
    iopt=np.argmax(zv); xopt=V[iopt]
    di=dopt_subset(C,k+1+1)
    deff_full=doelp.d_efficiency(np.column_stack([np.ones(N0),C]))
    deff_sub=doelp.d_efficiency(np.column_stack([np.ones(len(di)),C[di]]))
    for r in reps:
        B=N0*r
        alloc=np.full(len(di),B//len(di)); alloc[:B%len(di)]+=1
        Xs=np.repeat(C[di],alloc,axis=0)
        Xf=np.repeat(C,r,axis=0)
        for s in sigmas:
            sd=s*rng_z
            hit=np.zeros(3); loss=np.zeros((3,SIMS))
            for t in range(SIMS):
                # S1 & S2 share the same observations
                y=Xf@c+rng.normal(0,sd,len(Xf))
                means=y.reshape(N0,r).mean(1)[:len(V)]
                x1=V[np.argmax(means)]
                b0,bh,_,_=doelp.fit_first_order(Xf,y)
                r2=doelp.csa_solve(bh,p["G"],p["h"],p["E"],p["e"])
                x2=r2["x"]
                y3=Xs@c+rng.normal(0,sd,len(Xs))
                b0,bh3,_,_=doelp.fit_first_order(Xs,y3)
                x3=doelp.csa_solve(bh3,p["G"],p["h"],p["E"],p["e"])["x"]
                for q,x in enumerate((x1,x2,x3)):
                    l=(zs-c@x)/rng_z
                    loss[q,t]=l; hit[q]+= l<1e-6
            rec=dict(problem=name,k=k,vertices=len(V),N_design=N0,budget=B,r=r,sigma=s,
                     hit=(hit/SIMS).tolist(),loss_mean=(loss.mean(1)*100).tolist(),
                     loss_p95=(np.percentile(loss,95,axis=1)*100).tolist(),
                     deff_full=deff_full,deff_sub=deff_sub,n_sub=len(di))
            out.append(rec)
            print(name[:3],"r",r,"sig",s,"hit",np.round(hit/SIMS,3),"loss%",np.round(loss.mean(1)*100,3),flush=True)
json.dump(out,open("results/e3.json","w"),indent=1)
