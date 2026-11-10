import numpy as np, json
from scipy.optimize import linprog
import doelp, problems as pr
rows=[]
for name in pr.P:
    p=pr.get(name)
    ev=doelp.evd_solve(p["c"],p["G"],p["h"],p["E"],p["e"])
    cs=doelp.csa_solve(p["c"],p["G"],p["h"],p["E"],p["e"])
    sx=doelp.simplex_tableau(p["c"],p["G"],p["h"],p["E"],p["e"])
    hi=linprog(-p["c"],A_ub=p["G"],b_ub=p["h"],A_eq=p["E"],b_eq=p["e"],method="highs")
    rep=lambda z: pr.report(p,z) if z is not None else None
    r=dict(name=name,kind=p["kind"],
      highs=("unbounded" if hi.status==3 else rep(-hi.fun)),
      evd_status=ev["status"],evd_z=rep(ev.get("z")),evd_x=np.round(ev.get("x"),4).tolist(),evd_runs=ev["runs"],evd_sys=ev["systems"],evd_alt=ev.get("n_alt"),
      csa_status=cs["status"],csa_z=rep(cs.get("z")),csa_x=np.round(cs["x"],4).tolist(),csa_st1=cs["stages1"],csa_st2=cs["stages2"],csa_drops=cs["drops"],csa_alt=cs["alt_optima"],
      sx_status=sx["status"],sx_z=rep(sx.get("z")),sx_p1=sx["pivots1"],sx_p2=sx["pivots2"])
    rows.append(r); print(r)
json.dump(rows,open("results/e1.json","w"),indent=1)
