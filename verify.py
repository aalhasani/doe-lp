"""Compare a fresh rerun (results/*.json) with the values reported in the manuscript.
Run after e1.py, e2.py, e3.py, e4.py.  Prints PASS/FAIL per check."""
import json, sys
ok = True
def check(name, got, want, tol=1e-6):
    global ok
    good = abs(got - want) <= tol * max(1.0, abs(want))
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:.6g}, paper {want:.6g}")

e1 = {r["name"][:2]: r for r in json.load(open("results/e1.json"))}
for k, z in {"P1": 21, "P2": 437.6470588, "P3": 3.4, "P4": 18, "P5": 10, "P7": 0.99648, "P8": 64625}.items():
    check(f"E1 {k} EVD optimum", float(e1[k]["evd_z"]), z, 1e-6)
    check(f"E1 {k} CSA optimum", float(e1[k]["csa_z"]), z, 1e-6)
print(("PASS" if e1["P6"]["evd_status"] == e1["P6"]["csa_status"] == "unbounded" else "FAIL"), " E1 P6 unbounded flagged")

e2 = json.load(open("results/e2.json"))
check("E2 instances where CSA path = simplex path", sum(d["same_path"] for d in e2), 480, 0)
check("E2 CSA correct (all 480)", sum(d["ok_csa"] for d in e2), 480, 0)
d50 = [d for d in e2 if d["n"] == 50 and d["m"] == 50][0]
check("E2 mean pivots n=m=50", d50["pv"], 12.65, 1e-3)
check("E2 interior-start stages n=m=50", d50["stc"], 53.0, 1e-3)

e3 = json.load(open("results/e3.json"))
def loss(p, s, q): return [d for d in e3 if d["problem"].startswith(p) and d["r"] == 2 and d["sigma"] == s][0]["loss_mean"][q]
for p, vals in {"P1": (2.04, 0.91, 0.63), "P8": (1.63, 0.31, 0.29), "P7": (4.21, 2.16, 1.38)}.items():
    for q, v in enumerate(vals):
        check(f"E3 {p} sigma=10% loss S{q+1} (%)", round(loss(p, 0.10, q), 2), v, 1e-9)

e4 = json.load(open("results/e4.json"))
check("E4 R^2", round(e4["R2"], 3), 0.938, 1e-9)
check("E4 lack-of-fit p", round(e4["anova"]["plof"], 2), 0.82, 1e-9)
check("E4 true profit of DOE plan", round(e4["true_at_x"]), 64045, 0)
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - send me the output")
sys.exit(0 if ok else 1)
