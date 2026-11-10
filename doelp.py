"""
doelp.py - Design-of-Experiments (DOE) routes to the optimum of a linear programming model.

Problem form (all methods):
    maximize   z = c'x
    subject to G x <= h     (inequalities; nonnegativity and factor ranges are added internally)
               E x  = e     (equalities, optional)
               0 <= x <= U  (U = practical factor ranges; U_j = inf allowed for the simplex only)

Methods
-------
1. evd_solve      : Extreme-Vertices Design (McLean & Anderson, 1966). Enumerates every vertex of the
                    constrained experimental region, "runs" the response at each, selects the best.
                    Exact; reports all alternative optima; flags unboundedness by region expansion.
2. csa_solve      : Constrained Steepest Ascent (Box-Wilson RSM path of steepest ascent with an
                    active-set projection and a ratio test). A two-stage version handles regions
                    where the origin is infeasible (stage I minimizes artificial violations).
3. fit_first_order: OLS fit of a first-order response-surface model from designed runs (noisy case).
4. simplex_tableau: Textbook two-phase tableau simplex (Dantzig rule, Bland fallback) used only to count
                    pivots for comparison.
"""
from itertools import combinations
import numpy as np

TOL = 1e-9


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def build_region(G, h, n, U=None):
    """Stack user inequalities with nonnegativity and (finite) upper bounds. Returns (A, b, tags)."""
    rows, rhs, tags = [], [], []
    if G is not None and len(G):
        for i, (g, hh) in enumerate(zip(np.atleast_2d(G), np.atleast_1d(h))):
            rows.append(np.asarray(g, float)); rhs.append(float(hh)); tags.append(("con", i))
    for j in range(n):
        r = np.zeros(n); r[j] = -1.0
        rows.append(r); rhs.append(0.0); tags.append(("lb", j))
    if U is not None:
        for j in range(n):
            if np.isfinite(U[j]):
                r = np.zeros(n); r[j] = 1.0
                rows.append(r); rhs.append(float(U[j])); tags.append(("ub", j))
    return np.array(rows), np.array(rhs), tags


def default_ranges(G, h, E, e, n, factor=10.0):
    """Practical factor ranges when none are given: 10 x the largest |rhs| (at least 10)."""
    vals = [abs(v) for v in list(np.atleast_1d(h if h is not None else [])) +
            list(np.atleast_1d(e if e is not None else []))]
    big = factor * max([1.0] + vals)
    return np.full(n, big)


# ----------------------------------------------------------------------------------------------
# 1. Extreme-vertices design
# ----------------------------------------------------------------------------------------------
def extreme_vertices(G, h, E=None, e=None, U=None, tol=1e-7):
    """All vertices of {Gx<=h, Ex=e, 0<=x<=U}. Returns (V, active_sets, n_systems)."""
    G = None if G is None else np.atleast_2d(np.asarray(G, float))
    n = (G.shape[1] if G is not None else np.atleast_2d(E).shape[1])
    A, b, tags = build_region(G, h, n, U)
    Eq = np.zeros((0, n)) if E is None else np.atleast_2d(np.asarray(E, float))
    eq = np.zeros(0) if e is None else np.atleast_1d(np.asarray(e, float))
    k = n - Eq.shape[0]
    V, act, nsys = [], [], 0
    for S in combinations(range(len(b)), k):
        M = np.vstack([Eq, A[list(S)]]); r = np.concatenate([eq, b[list(S)]])
        nsys += 1
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, r)
        if np.all(A @ x <= b + tol) and (Eq.shape[0] == 0 or np.allclose(Eq @ x, eq, atol=tol)):
            if not any(np.allclose(x, v, atol=1e-7) for v in V):
                V.append(x); act.append([tags[s] for s in S])
    return np.array(V), act, nsys


def evd_solve(c, G, h, E=None, e=None, U=None, expand=10.0, response=None):
    """Extreme-vertices design route to the LP optimum (maximization).
    response(x) -> observed z; default = exact c'x."""
    c = np.asarray(c, float); n = len(c)
    if U is None:
        U = default_ranges(G, h, E, e, n)
    resp = response if response is not None else (lambda x: float(c @ x))
    V, act, nsys = extreme_vertices(G, h, E, e, U)
    if len(V) == 0:
        return dict(status="infeasible", runs=0, systems=nsys)
    z = np.array([resp(v) for v in V])
    best = z.max()
    ties = np.where(z >= best - 1e-7 * max(1, abs(best)))[0]
    i = ties[0]
    # unboundedness check: optimum touches an artificial range -> expand range and re-run
    touches = [j for j in range(n) if abs(V[i][j] - U[j]) < 1e-7]
    status = "optimal"
    if touches:
        V2, _, nsys2 = extreme_vertices(G, h, E, e, U * expand)
        z2 = max(resp(v) for v in V2)
        nsys += nsys2
        if z2 > best * (1 + 1e-6) + 1e-9:
            status = "unbounded"
    return dict(status=status, x=V[i], z=best, n_alt=len(ties), alt=V[ties], runs=len(V),
                systems=nsys, vertices=V, zv=z)


# ----------------------------------------------------------------------------------------------
# 2. Constrained steepest ascent (RSM path with active set)
# ----------------------------------------------------------------------------------------------
def _independent(rows, new):
    if len(rows) == 0:
        return np.linalg.norm(new) > TOL
    M = np.vstack(rows + [new])
    return np.linalg.matrix_rank(M, tol=1e-9) == len(rows) + 1


def _csa_core(c, A, b, Eq, eq, x, max_stages=10000, tol=1e-9, path=None):
    """Steepest ascent of c'x from feasible x over {Ax<=b, Eq x = eq}.
    Returns x, status, stages (moves along the path), drops (constraint releases), W, lam."""
    n = len(c)
    W = []  # indices of inequality rows in working set
    eq_rows = [r for r in Eq]
    base = list(eq_rows)
    for i in np.where(np.abs(A @ x - b) <= 1e-9 * (1 + np.abs(b)))[0]:
        if _independent(base + [A[j] for j in W], A[i]):
            W.append(int(i))
    stages = drops = 0
    blocked_bland = False
    for it in range(max_stages):
        N = np.array(base + [A[j] for j in W]) if (len(base) + len(W)) else np.zeros((0, n))
        if N.shape[0]:
            lam, *_ = np.linalg.lstsq(N.T, c, rcond=None)
            d = c - N.T @ lam
        else:
            lam = np.zeros(0); d = c.copy()
        if np.linalg.norm(d) > 1e-9 * (1 + np.linalg.norm(c)):
            # ratio test along the steepest-ascent direction projected on the current face
            Ad = A @ d
            slack = b - A @ x
            cand = [(max(slack[i], 0.0) / Ad[i], i) for i in range(len(b))
                    if i not in W and Ad[i] > 1e-12]
            if not cand:
                return x, "unbounded", stages, drops, W, lam
            t = min(v for v, _ in cand)
            x = x + t * d
            stages += 1
            if path is not None:
                path.append(x.copy())
            # add the first independent blocking row(s)
            for v, i in sorted(cand):
                if v <= t + 1e-12 and _independent(base + [A[j] for j in W], A[i]):
                    W.append(i); break
            continue
        lam_in = lam[len(base):]
        if len(lam_in) == 0 or lam_in.min() >= -1e-9:
            return x, "optimal", stages, drops, W, lam_in
        # release a constraint whose multiplier is negative (most negative; Bland if cycling)
        neg = [k for k in range(len(lam_in)) if lam_in[k] < -1e-9]
        if it > 50 * (n + len(b)):
            blocked_bland = True
        k = min(neg, key=lambda k: W[k]) if blocked_bland else int(np.argmin(lam_in))
        W.pop(k); drops += 1
    return x, "stage_limit", stages, drops, W, lam


def csa_solve(c, G, h, E=None, e=None, U=None, x0=None, expand=10.0, path=None):
    """Two-stage constrained steepest ascent. Stage I (if needed) reaches the feasible region by
    steepest ascent on minus the sum of artificial deviations; stage II climbs c'x."""
    c = np.asarray(c, float); n = len(c)
    if U is None:
        U = default_ranges(G, h, E, e, n)
    G2 = None if G is None else np.atleast_2d(np.asarray(G, float))
    A, b, tags = build_region(G2, h, n, U)
    Eq = np.zeros((0, n)) if E is None else np.atleast_2d(np.asarray(E, float))
    eq = np.zeros(0) if e is None else np.atleast_1d(np.asarray(e, float))
    st1 = 0
    if x0 is None:
        x0 = np.zeros(n)
    feasible = np.all(A @ x0 <= b + 1e-9) and (len(eq) == 0 or np.allclose(Eq @ x0, eq))
    if not feasible:
        # stage I: variables (x, a_in, a_eq); constraints Ax - a_in <= b ; Eq x + s*a_eq = eq
        viol = np.maximum(A @ x0 - b, 0)
        need = np.where(viol > 1e-12)[0]
        sgn = np.sign(eq - Eq @ x0); sgn[sgn == 0] = 1
        p, q = len(need), len(eq)
        N1 = n + p + q
        A1 = np.zeros((len(b) + p + q, N1)); b1 = np.zeros(len(b) + p + q)
        A1[:len(b), :n] = A; b1[:len(b)] = b
        for k, i in enumerate(need):
            A1[i, n + k] = -1.0
        for k in range(p + q):                       # artificials >= 0
            A1[len(b) + k, n + k] = -1.0
        E1 = np.zeros((q, N1)); E1[:, :n] = Eq
        for k in range(q):
            E1[k, n + p + k] = sgn[k]
        c1 = np.zeros(N1); c1[n:] = -1.0
        y0 = np.concatenate([x0, viol[need], np.abs(eq - Eq @ x0)])
        y, s1, st1, d1, _, _ = _csa_core(c1, A1, b1, E1, eq, y0)
        if c1 @ y < -1e-7:
            return dict(status="infeasible", stages1=st1)
        x0 = y[:n]
    if path is not None:
        path.append(np.array(x0, float).copy())
    x, status, st2, drops, W, lam = _csa_core(c, A, b, Eq, eq, x0, path=path)
    out = dict(status=status, x=x, z=float(c @ x), stages1=st1, stages2=st2, drops=drops)
    # alternative optima: an active inequality with zero multiplier at a nondegenerate optimum
    out["alt_optima"] = bool(status == "optimal" and len(lam) and np.any(np.abs(lam) < 1e-9))
    # unboundedness: optimum sits on an artificial factor range -> expand and compare
    touches = [j for j in range(n) if abs(x[j] - U[j]) < 1e-7]
    if status == "optimal" and touches:
        r2 = csa_solve(c, G, h, E, e, U * expand, x0=None, expand=1.0) if expand > 1 else None
        if r2 is not None and r2["z"] > out["z"] * (1 + 1e-6) + 1e-9:
            out["status"] = "unbounded"
    return out


# ----------------------------------------------------------------------------------------------
# 3. First-order model fit (noisy / unknown coefficients)
# ----------------------------------------------------------------------------------------------
def fit_first_order(X, y):
    """OLS fit y = b0 + b'x. Returns (b0, b, residual_sd, design_matrix)."""
    X = np.asarray(X, float)
    D = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = y - D @ beta
    dof = max(len(y) - D.shape[1], 1)
    return beta[0], beta[1:], float(np.sqrt(res @ res / dof)), D


def d_efficiency(D):
    """D-efficiency (%) of a model matrix D (N x p)."""
    N, p = D.shape
    det = np.linalg.det(D.T @ D)
    return 100.0 * max(det, 0) ** (1.0 / p) / N


# ----------------------------------------------------------------------------------------------
# 4. Textbook two-phase tableau simplex (for pivot counts)
# ----------------------------------------------------------------------------------------------
def simplex_tableau(c, G, h, E=None, e=None, max_iter=100000):
    """max c'x, Gx<=h, Ex=e, x>=0. Two-phase method, Dantzig entering rule, Bland fallback.
    Returns dict(status, x, z, pivots1, pivots2)."""
    c = np.asarray(c, float); n = len(c)
    G = np.zeros((0, n)) if G is None else np.atleast_2d(np.asarray(G, float))
    h = np.zeros(0) if h is None else np.atleast_1d(np.asarray(h, float))
    E = np.zeros((0, n)) if E is None else np.atleast_2d(np.asarray(E, float))
    e = np.zeros(0) if e is None else np.atleast_1d(np.asarray(e, float))
    m1, m2 = len(h), len(e)
    # rows: G x + s = h (flip if h<0 -> surplus + artificial), E x + a = e
    rows, rhs, basis = [], [], []
    ncols = n + m1  # x, slack/surplus
    art_cols = []
    Arows = []
    for i in range(m1):
        r = np.zeros(ncols); r[:n] = G[i]; r[n + i] = 1.0; bb = h[i]
        if bb < 0:
            r = -r; bb = -bb
            Arows.append((r, bb, True))
        else:
            Arows.append((r, bb, False))
    for i in range(m2):
        r = np.zeros(ncols); r[:n] = E[i]; bb = e[i]
        if bb < 0:
            r = -r; bb = -bb
        Arows.append((r, bb, True))
    na = sum(1 for _, _, a in Arows if a)
    T = np.zeros((m1 + m2, ncols + na + 1))
    k = 0
    for i, (r, bb, a) in enumerate(Arows):
        T[i, :ncols] = r; T[i, -1] = bb
        if a:
            T[i, ncols + k] = 1.0; basis.append(ncols + k); art_cols.append(ncols + k); k += 1
        else:
            basis.append(n + i)
    total = ncols + na

    def run(obj, allowed, maxit):
        piv = 0
        bland = False
        seen = set()
        while piv < maxit:
            cb = obj[basis]
            red = obj[:total] - cb @ T[:, :total]
            cand = [j for j in allowed if red[j] > 1e-10]
            if not cand:
                return "optimal", piv
            key = tuple(basis)
            if key in seen:
                bland = True
            seen.add(key)
            j = min(cand) if bland else max(cand, key=lambda j: red[j])
            col = T[:, j]
            rat = [(T[i, -1] / col[i], basis[i], i) for i in range(len(basis)) if col[i] > 1e-10]
            if not rat:
                return "unbounded", piv
            tmin = min(r[0] for r in rat)
            ties = [r for r in rat if r[0] <= tmin + 1e-12]
            i = min(ties, key=lambda r: r[1])[2]
            T[i] /= T[i, j]
            for r in range(len(basis)):
                if r != i:
                    T[r] -= T[r, j] * T[i]
            basis[i] = j
            piv += 1
        return "limit", piv

    p1 = 0
    if na:
        obj1 = np.zeros(total); obj1[art_cols] = -1.0
        s, p1 = run(obj1, list(range(total)), max_iter)
        if T[[i for i in range(len(basis)) if basis[i] in art_cols], -1].sum() > 1e-7:
            return dict(status="infeasible", pivots1=p1, pivots2=0)
    obj2 = np.zeros(total); obj2[:n] = c
    allowed = [j for j in range(total) if j not in art_cols]
    s, p2 = run(obj2, allowed, max_iter)
    x = np.zeros(total)
    for i, bi in enumerate(basis):
        x[bi] = T[i, -1]
    return dict(status=s, x=x[:n], z=float(c @ x[:n]), pivots1=p1, pivots2=p2)
