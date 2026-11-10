# DOE routes to the LP optimum: reproducible code

Requires Python 3.10+.  Install: pip install -r requirements.txt

    python e1.py    # Table II: eight textbook LPs (Taha, 10th ed.)
    python e2.py    # Fig. 2: 480 random LPs (seed 20261110), ~1 min
    python e3.py    # Fig. 3, Table III: noisy coefficients (seed 7), ~10 min
    python e4.py    # Table IV: worked ANOVA example (seed 2026)
    python figs.py  # Figs. 1-3 (PNG + PDF) in results/
    python verify.py # checks the rerun against the numbers in the paper

doelp.py    : extreme-vertices design, constrained steepest ascent, OLS fit, tableau simplex
problems.py : the eight test problems in max-form

Funded by Umm Al-Qura University, Saudi Arabia, grant number 26UQU4400075GSSR05. License: MIT.
