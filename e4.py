"""Worked example: P8 production, unknown coefficients, EVD+centroid, r=2, sigma=10% of range."""
import numpy as np, json
from scipy import stats
import doelp, problems as pr
rng=np.random.default_rng(2026)
p=pr.get("P8 Production (Ex. 2.4-2)"); c=p["c"]; k=4
ev=doelp.evd_solve(c,p["G"],p["h"]); V=ev["vertices"]; zv=V@c
sd=0.10*(zv.max()-zv.min())
C=np.vstack([V,V.mean(0)]); r=2; X=np.repeat(C,r,axis=0)
beta0=-37000+0  # true intercept of net profit
y=beta0+X@c+rng.normal(0,sd,len(X))
b0,b,s,D=doelp.fit_first_order(X,y)
N,pp=D.shape
XtXi=np.linalg.inv(D.T@D); beta=np.concatenate([[b0],b]); se=np.sqrt(np.diag(XtXi))*s
tcrit=stats.t.ppf(.975,N-pp)
yhat=D@beta; sse=((y-yhat)**2).sum(); sst=((y-y.mean())**2).sum(); ssr=sst-sse
# pure error from replicates
groups=y.reshape(len(C),r); sspe=((groups-groups.mean(1,keepdims=True))**2).sum(); dfpe=len(C)*(r-1)
sslof=sse-sspe; dflof=(N-pp)-dfpe
Flof=(sslof/dflof)/(sspe/dfpe); plof=1-stats.f.cdf(Flof,dflof,dfpe)
Freg=(ssr/(pp-1))/(sse/(N-pp)); preg=1-stats.f.cdf(Freg,pp-1,N-pp)
res=doelp.csa_solve(b,p["G"],p["h"])
x=res["x"]; true_at=c@x-37000
out=dict(sd=sd,N=N,coef=beta.tolist(),se=se.tolist(),ci_lo=(beta-tcrit*se).tolist(),ci_hi=(beta+tcrit*se).tolist(),
 true=[-37000]+c.tolist(),R2=ssr/sst,R2adj=1-(sse/(N-pp))/(sst/(N-1)),
 anova=dict(ssr=ssr,dfr=pp-1,sse=sse,dfe=N-pp,F=Freg,p=preg,sslof=sslof,dflof=dflof,sspe=sspe,dfpe=dfpe,Flof=Flof,plof=plof),
 x_doe=x.tolist(),pred=float(b0+b@x),true_at_x=float(true_at),true_opt=64625.0,stages=res["stages2"],vertices=len(V))
json.dump(out,open("results/e4.json","w"),indent=1)
for kk,v in out.items(): print(kk,v)
