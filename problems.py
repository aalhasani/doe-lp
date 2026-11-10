"""Benchmark LPs from Taha, Operations Research: An Introduction, 10th Global Ed. (2017),
written as  max c'x  s.t.  Gx <= h, Ex = e, x >= 0.  'const' is added to the reported objective;
'sense' = -1 marks an original minimization (reported value = -z)."""
import numpy as np

P = {
    "P1 Reddy Mikks (Ex. 2.1-1)": dict(
        c=[5, 4], G=[[6, 4], [1, 2], [-1, 1], [0, 1]], h=[24, 6, 1, 2], sense=1, kind="Baseline"),
    "P2 Diet (Ex. 2.2-2)": dict(
        c=[-.3, -.9], G=[[-1, -1], [.21, -.30], [-.03, .01]], h=[-800, 0, 0], sense=-1,
        kind="Minimization, >= rows"),
    "P3 Equality (Ex. 3.4-1)": dict(
        c=[-4, -1], G=[[-4, -3], [1, 2]], h=[-6, 4], E=[[3, 1]], e=[3], sense=-1,
        kind="Equality + >= rows"),
    "P4 Degenerate (Ex. 3.5-1)": dict(
        c=[3, 9], G=[[1, 4], [1, 2]], h=[8, 4], sense=1, kind="Degenerate optimum"),
    "P5 Alternative optima (Ex. 3.5-2)": dict(
        c=[2, 4], G=[[1, 2], [1, 1]], h=[5, 4], sense=1, kind="Alternative optima"),
    "P6 Unbounded (Ex. 3.5-3)": dict(
        c=[2, 1], G=[[1, -1], [2, 0]], h=[10, 40], sense=1, kind="Unbounded"),
    "P7 Bank loan (Ex. 2.4-1)": dict(
        c=[.026, .0509, .0864, .06875, .078],
        G=[[1, 1, 1, 1, 1], [.4, .4, .4, -.6, -.6], [.5, .5, -.5, 0, 0],
           [.06, .03, -.01, .01, -.02]], h=[12, 0, 0, 0], sense=1, kind="Ratio constraints, n=5"),
    "P8 Production (Ex. 2.4-2)": dict(
        c=[45, 60, 30, 18],
        G=[[.30, .30, .25, .15], [.25, .35, .30, .10], [.45, .50, .40, .22], [.15, .15, .10, .05],
           [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        h=[1000, 1000, 1000, 1000, 800, 750, 600, 500], const=-37000, sense=1,
        kind="Engineering case, n=4"),
}


def get(name):
    p = dict(P[name])
    p["c"] = np.array(p["c"], float); p["G"] = np.array(p["G"], float); p["h"] = np.array(p["h"], float)
    if "E" in p:
        p["E"] = np.array(p["E"], float); p["e"] = np.array(p["e"], float)
    else:
        p["E"] = None; p["e"] = None
    p.setdefault("const", 0.0)
    return p


def report(p, z):
    return p["sense"] * (z + p["const"]) if p["sense"] == 1 else -(z) + 0 * p["const"]
