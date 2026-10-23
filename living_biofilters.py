#!/usr/bin/env python3
"""
living_biofilters.py - all the Python code for the living-biofilter study in one file.

Odibo, A., Janpum, C., Pombubpa, N., Monshupanee, T., Incharoensakdi, A.,
Rehman, Z. U., & In-na, P. (2024). Microalgal-bacterial immobilized co-culture
as living biofilters for nutrient recovery from synthetic wastewater and their
potential as biofertilizers. Bioresource Technology, 398, 130509.
https://doi.org/10.1016/j.biortech.2024.130509

WHAT IS INSIDE
  Part 1  Equations: cell viability, degradation, decision matrix, nutrient
          removal and absorption rate, chlorophyll a, soil N/P/K, arcsine transform
  Part 2  Statistics: Shapiro-Wilk -> ANOVA + Tukey or Kruskal-Wallis +
          Mann-Whitney U, factorial ANOVA, mixed (repeated-measures) ANOVA,
          compact letter display
  Part 3  Figures in the layout of Figs. 2-6
  Part 4  Example data generator (SIMULATED, see the note below)
  Part 5  Statistics pipeline for Sections 3.1-3.8
  Part 6  Kinetic bioreactor model (Monod-Droop algae, Monod bacteria, light,
          luxury P uptake, fast sorption stage, oxygen balance)
  Part 7  Calibration of the model to the 7-day nutrient-removal data
  Part 8  Bioreactor runs + C. vulgaris photobioreactor (BG-11) scenario
  Part 9  The COMSOL Multiphysics model equations solved in 1D through the gel
  Part 10 Self-tests

INSTALL
  pip install numpy pandas scipy statsmodels pingouin matplotlib

RUN
  python living_biofilters.py all                     # everything, on example data
  python living_biofilters.py analysis                # statistics + Figs. 2-6
  python living_biofilters.py bioreactor              # fit model, scenarios
  python living_biofilters.py comsol1d                # spatial (gel) model preview
  python living_biofilters.py test                    # self-tests
  python living_biofilters.py example-data            # write data/example/*.csv
  python living_biofilters.py analysis --data data/raw     # your own measurements
  python living_biofilters.py <command> --help        # options for each command

INPUT DATA
  Five CSV files in one folder: toxicity.csv, degradation.csv, nutrients.csv,
  plants.csv, soil.csv. Run "example-data" once to see the exact columns, then
  put your raw measurements in the same layout in e.g. data/raw/.

  NOTE: the example data are random values (n = 3, fixed seed) drawn around the
  means and SDs reported in the paper. They let the code run out of the box but
  are NOT the original measurements. Use your raw data for real results.

OUTPUT
  results/python/      processed tables, statistics_report.txt, figures
  results/bioreactor/  fitted parameters, model-vs-data, model figures,
                       parameters_fitted.txt (for COMSOL: Parameters > Load from File)

License: MIT.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict, replace
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path.cwd()
DEFAULT_DATA = ROOT / "data" / "example"

ANALYSIS_DOC = """Statistics pipeline (paper Sections 3.1-3.8).

    python living_biofilters.py analysis
    python living_biofilters.py analysis --data data/raw
    python living_biofilters.py analysis --paper-methods   # tests exactly as in the paper
"""
BIOREACTOR_DOC = """Kinetic bioreactor model: fit to the nutrient trials and run scenarios.

    python living_biofilters.py bioreactor
    python living_biofilters.py bioreactor --data data/raw
    python living_biofilters.py bioreactor --skip-fit --pbr-days 21 --pbr-volume 10
"""
COMSOL1D_DOC = """The COMSOL model equations solved in 1D through the gel layer.

    python living_biofilters.py comsol1d
    python living_biofilters.py comsol1d --f-gel 0.05      # denser gel
"""


def ensure_data(folder):
    """Create the example data the first time if the default folder is missing."""
    folder = Path(folder)
    if not (folder / "nutrients.csv").exists():
        if folder.resolve() == DEFAULT_DATA.resolve():
            make_example_data(folder)
        else:
            sys.exit(f"No input data found in {folder}")



# ============================================================================
# PART 1 - Equations from the paper (Sections 2.3-2.13)
# ============================================================================

# --------------------------------------------------------------------------
# 2.3 Toxicity test - % cell viability
# --------------------------------------------------------------------------
def cell_viability(sample_od_d0, sample_od_d7, control_od_d0, control_od_d7,
                   blank_d0=0.0, blank_d7=0.0):
    """% cell viability (OD750), adapted from He et al. (2006).

    viability = [(S7 - blank7) - (S0 - blank0)] / [(C7 - blank7) - (C0 - blank0)] x 100
    """
    sample_growth = (sample_od_d7 - blank_d7) - (sample_od_d0 - blank_d0)
    control_growth = (control_od_d7 - blank_d7) - (control_od_d0 - blank_d0)
    return sample_growth / control_growth * 100.0


# --------------------------------------------------------------------------
# 2.4 Degradation test
# --------------------------------------------------------------------------
def degradation(dry_weight_d0, dry_weight_d7):
    """% degradation = (W0 - W7) / W0 x 100"""
    return (dry_weight_d0 - dry_weight_d7) / dry_weight_d0 * 100.0


# --------------------------------------------------------------------------
# 2.5 Nutrient removal
# --------------------------------------------------------------------------
def removal_efficiency(c_initial, c_final):
    """% nutrient removal efficiency = (Ci - Cf) / Ci x 100"""
    return (c_initial - c_final) / c_initial * 100.0


def absorption_rate(c_initial, c_final, volume_l, time_day, biomass_g):
    """Nutrient absorption rate (mg day^-1 g_biomass^-1)

    rate = (Ci - Cf) x V / (T x m)
    Ci, Cf : mg/L   V : working volume (L)   T : days   m : initial dry biomass (g)
    """
    time_day = np.asarray(time_day, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (c_initial - c_final) * volume_l / (time_day * biomass_g)
    return np.where(time_day > 0, rate, np.nan)


# --------------------------------------------------------------------------
# 2.10 Chlorophyll a per dried plant biomass (modified Sartory & Grobbelaar 1984)
# --------------------------------------------------------------------------
def chlorophyll_a(a630, a645, a663, a750, dry_weight_g, extract_volume_ml=1.0):
    """Chl a per dried plant biomass (ug / g).

    Chl a = [11.62(A663-A750) - 2.16(A645-A750) - 0.1(A630-A750)] / dry weight

    The bracket gives ug/mL of extract. The paper divides it directly by dry
    weight (extract_volume_ml = 1). Pass the real extract volume (e.g. 1.5 mL)
    to convert to ug of chlorophyll per g of biomass.
    """
    conc = (11.62 * (a663 - a750)
            - 2.16 * (a645 - a750)
            - 0.10 * (a630 - a750))
    return conc * extract_volume_ml / dry_weight_g


# --------------------------------------------------------------------------
# 2.12 Soil N, P, K
# --------------------------------------------------------------------------
def total_n(titrant_volume, acid_normality, soil_weight_kg):
    """Kjeldahl total N (mg/kg) = V x N x 14.01 x 100 / W"""
    return titrant_volume * acid_normality * 14.01 * 100.0 / soil_weight_kg


def total_p(p_conc_mg_ml, v_sample, acid_molarity, v_naoh, v_molybdate,
            dilution_pct, soil_weight_kg):
    """Alkaline-oxidation total P (mg/kg)

    = C x (V1/M) x (V2/V3) x 1000 / ((100 - H) x W)
    """
    return (p_conc_mg_ml * (v_sample / acid_molarity) * (v_naoh / v_molybdate)
            * 1000.0 / ((100.0 - dilution_pct) * soil_weight_kg))


def total_k(od_k, soil_weight_kg, standard_conc=100.0):
    """Flame-photometer total K (mg/kg) = OD x 1.2 x 100 / W"""
    return od_k * 1.2 * standard_conc / soil_weight_kg


# --------------------------------------------------------------------------
# 2.13 Transform used on percentage data before statistics
# --------------------------------------------------------------------------
def arcsine_transform(percent):
    """asin(sqrt(p)) with p as a proportion; values clipped to [0, 1].

    Viability can exceed 100 %, so values are clipped before transforming.
    """
    p = np.clip(np.asarray(percent, dtype=float) / 100.0, 0.0, 1.0)
    return np.arcsin(np.sqrt(p))


# --------------------------------------------------------------------------
# 2.4 Decision matrix used to pick the binder / ratio
# --------------------------------------------------------------------------
def decision_matrix(viability, degradation_pct, w_viability=0.5, w_degradation=0.5):
    """Score conditions: high viability is good, low degradation is good.

    Each criterion is min-max scaled to 0-1, then combined with the weights.
    Returns scores in the input order (higher is better).
    """
    v = np.asarray(viability, dtype=float)
    d = np.asarray(degradation_pct, dtype=float)

    def scale(x):
        rng = np.ptp(x)
        return np.zeros_like(x) if rng == 0 else (x - x.min()) / rng

    return w_viability * scale(v) + w_degradation * (1.0 - scale(d))



# ============================================================================
# PART 2 - Statistics (Section 2.13)
# ============================================================================

ALPHA = 0.05


def is_normal(df, value, group, alpha=ALPHA):
    """True if every group passes Shapiro-Wilk (groups with n < 3 are skipped)."""
    for _, g in df.groupby(group, observed=True):
        x = g[value].dropna()
        if len(x) >= 3 and np.ptp(x) > 0:
            if stats.shapiro(x).pvalue < alpha:
                return False
    return True


def compact_letters(groups, pvals, alpha=ALPHA):
    """Compact letter display: groups sharing a letter are not significantly different.

    groups : ordered list of group names
    pvals  : dict {(g1, g2): p} for each pair
    """
    def p_of(a, b):
        return pvals.get((a, b), pvals.get((b, a), 1.0))

    # insert-and-absorb algorithm (Piepho 2004)
    letter_sets = [set(groups)]
    for a, b in combinations(groups, 2):
        if p_of(a, b) >= alpha:
            continue
        new_sets = []
        for s in letter_sets:
            if a in s and b in s:
                new_sets.append(s - {a})
                new_sets.append(s - {b})
            else:
                new_sets.append(s)
        # absorb: drop sets that are subsets of another set
        new_sets = [s for i, s in enumerate(new_sets)
                    if not any(s < t or (s == t and j < i)
                               for j, t in enumerate(new_sets) if j != i)]
        letter_sets = new_sets

    # order letters by first group they appear in
    letter_sets.sort(key=lambda s: min(groups.index(g) for g in s))
    out = {g: "" for g in groups}
    for i, s in enumerate(letter_sets):
        for g in s:
            out[g] += "abcdefghijklmnopqrstuvwxyz"[i]
    return out


def compare_groups(df, value, group, method="auto", transform=None, alpha=ALPHA):
    """One-factor comparison across groups.

    method    : "auto" (pick by normality), "anova" or "kruskal"
    transform : None or "arcsine" (for % data)
    Returns a dict with test name, statistic, df, p, pairwise table and letters.
    """
    d = df[[group, value]].dropna().copy()
    if transform == "arcsine":
        d[value] = arcsine_transform(d[value])

    groups = list(pd.unique(d[group]))
    samples = [d.loc[d[group] == g, value].values for g in groups]

    if method == "auto":
        method = "anova" if is_normal(d, value, group, alpha) else "kruskal"

    pvals = {}
    if method == "anova":
        f, p = stats.f_oneway(*samples)
        stat_name, dfree = "F", (len(groups) - 1, len(d) - len(groups))
        tk = pairwise_tukeyhsd(d[value], d[group], alpha=alpha)
        tab = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
        for _, r in tab.iterrows():
            pvals[(r["group1"], r["group2"])] = float(r["p-adj"])
        pairwise = tab
        test = "One-way ANOVA + Tukey HSD"
        statistic = f
    else:
        h, p = stats.kruskal(*samples)
        stat_name, dfree = "H", len(groups) - 1
        rows = []
        for (ga, xa), (gb, xb) in combinations(zip(groups, samples), 2):
            u, pu = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            pvals[(ga, gb)] = pu
            rows.append({"group1": ga, "group2": gb, "U": u, "p": pu})
        pairwise = pd.DataFrame(rows)
        test = "Kruskal-Wallis + pairwise Mann-Whitney U"
        statistic = h

    return {
        "test": test,
        "stat_name": stat_name,
        "statistic": float(statistic),
        "df": dfree,
        "p": float(p),
        "pairwise": pairwise,
        "letters": compact_letters(groups, pvals, alpha),
    }


def factorial_anova(df, formula):
    """Multi-factor ANOVA (Type II), e.g. 'viability ~ C(binder) * C(inoculum)'."""
    model = smf.ols(formula, data=df).fit()
    return anova_lm(model, typ=2)


def mixed_anova(df, dv, within, between, subject):
    """Two-way repeated-measures (mixed) ANOVA: time (within) x group (between)."""
    import pingouin as pg
    return pg.mixed_anova(data=df, dv=dv, within=within, between=between,
                          subject=subject)


def format_result(name, res):
    """Readable one-line summary + letters."""
    dfs = res["df"] if isinstance(res["df"], int) else f"{res['df'][0]}, {res['df'][1]}"
    sig = "significant" if res["p"] < ALPHA else "not significant"
    letters = ", ".join(f"{k}: {v}" for k, v in res["letters"].items())
    return (f"{name}\n  {res['test']}: {res['stat_name']} = {res['statistic']:.3f}, "
            f"DF = {dfs}, P = {res['p']:.4f} ({sig})\n  Letters -> {letters}\n")



# ============================================================================
# PART 3 - Figures in the layout of Figs. 2-6
# ============================================================================

TREATMENT_COLORS = ["#F28030", "#FFC20E", "#6AA84F", "#990000", "#8B6508"]
RATIO_COLORS = {"CV": "#7F7F7F", "BS": "#B0B0B0", "3:1": "#C00000", "1:1": "#FFC000", "1:3": "#548235"}
PLANT_COLORS = {"Morning glory": "#FFC000", "Sunflower": "#C00000", "Cucumber": "#548235"}


def _bar_with_letters(ax, labels, means, sds, colors, letters=None):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, color=colors, capsize=3, edgecolor="none")
    ax.set_xticks(x, labels, fontsize=8)
    if letters:
        top = np.nanmax(np.asarray(means) + np.asarray(sds))
        for xi, m, s, lab in zip(x, means, sds, labels):
            ax.text(xi, m + s + 0.03 * top, letters.get(lab, ""), ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def fig_binder_screen(summary, value, ylabel, letters, path):
    """Fig. 2 (a,b) or (e,f): one panel per binder; summary has binder, inoculum, mean, sd."""
    binders = list(summary["binder"].unique())
    fig, axes = plt.subplots(len(binders), 1, figsize=(5.5, 3.2 * len(binders)), sharey=True)
    axes = np.atleast_1d(axes)
    order = ["CV", "BS", "3:1", "1:1", "1:3"]
    for ax, b in zip(axes, binders):
        s = summary[summary["binder"] == b].set_index("inoculum").reindex(order).dropna(how="all")
        _bar_with_letters(ax, list(s.index), s["mean"], s["sd"],
                          [RATIO_COLORS[i] for i in s.index], letters.get(b))
        ax.axvline(1.5, ls="--", color="grey", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(b, loc="right", fontsize=10)
    axes[-1].set_xlabel("Single culture  |  Mixing ratio (CV/BS)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_nutrients(summary, path):
    """Fig. 3: removal efficiency (a,b) and absorption rate (c,d) over time."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    styles = {("3:1", "I"): ("#C00000", "-", "s"), ("3:1", "S"): ("#C00000", "--", "s"),
              ("1:1", "I"): ("#2E75B6", "-", "o"), ("1:1", "S"): ("#2E75B6", "--", "o")}
    panels = [("nh4_removal_pct", "% NH$_4^+$ removal efficiency", "a"),
              ("po4_removal_pct", "% PO$_4^{3-}$ removal efficiency", "b"),
              ("nh4_rate", "NH$_4^+$ absorption rate (mg day$^{-1}$ g$^{-1}$)", "c"),
              ("po4_rate", "PO$_4^{3-}$ absorption rate (mg day$^{-1}$ g$^{-1}$)", "d")]
    for ax, (col, ylab, tag) in zip(axes.flat, panels):
        for (ratio, system), (c, ls, mk) in styles.items():
            s = summary[(summary.ratio == ratio) & (summary.system == system)].sort_values("day")
            if col.endswith("rate"):
                s = s[s.day > 0]
            ax.errorbar(s.day, s[f"{col}_mean"], yerr=s[f"{col}_sd"], color=c, ls=ls,
                        marker=mk, mfc=c if system == "I" else "white", capsize=2,
                        label=f"{ratio}_CV/BS_{system}")
        ax.set_xlabel("Time (Day)")
        ax.set_ylabel(ylab)
        ax.set_title(f"({tag})", loc="left", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_plant_lengths(summary, letters, treatments, path):
    """Fig. 4: grouped bars of plant-part lengths, one panel per plant."""
    parts = ["total_height_cm", "leaf_cm", "branch_cm", "stem_cm", "root_cm"]
    part_names = ["Total plant growth", "Leaf", "Branch", "Stem", "Root"]
    plants = list(summary["plant"].unique())
    fig, axes = plt.subplots(len(plants), 1, figsize=(10, 3.6 * len(plants)))
    w = 0.8 / len(treatments)
    for ax, plant, tag in zip(np.atleast_1d(axes), plants, "abc"):
        s = summary[summary.plant == plant].set_index("treatment")
        for i, (trt, col) in enumerate(zip(treatments, TREATMENT_COLORS)):
            x = np.arange(len(parts)) + (i - (len(treatments) - 1) / 2) * w
            m = [s.loc[trt, f"{p}_mean"] for p in parts]
            sd = [s.loc[trt, f"{p}_sd"] for p in parts]
            ax.bar(x, m, w, yerr=sd, color=col, capsize=2, label=trt)
            for xi, mi, si, p in zip(x, m, sd, parts):
                lab = letters.get((plant, p), {}).get(trt, "")
                ax.text(xi, mi + si + 0.3, lab, ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(parts)), part_names)
        ax.set_ylabel("Plant length (cm)")
        ax.set_title(f"({tag}) {plant}", loc="right", fontsize=10)
        ax.legend(fontsize=7, frameon=False, ncol=1, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_grid(summary, variables, ylabels, letters, treatments, path):
    """Figs. 5 and 6: 3 x 3 grid, rows = variables, columns = plants (as in the paper)."""
    plants = list(summary["plant"].unique())
    fig, axes = plt.subplots(len(plants), len(variables),
                             figsize=(4.2 * len(variables), 3.0 * len(plants)))
    short = ["Control", "Chemical\nfertilizer", "Biofilter\n(CV/BS 3:1)",
             "Biofilter\n(CV/BS 1:1)", "Biofilter\ncontrol"]
    tags = iter("abcdefghi")
    for j, (var, ylab) in enumerate(zip(variables, ylabels)):
        for i, plant in enumerate(plants):
            ax = axes[i, j]
            s = summary[summary.plant == plant].set_index("treatment").reindex(treatments)
            lt = {sh: letters.get((plant, var), {}).get(t, "") for sh, t in zip(short, treatments)}
            _bar_with_letters(ax, short, s[f"{var}_mean"], s[f"{var}_sd"],
                              [PLANT_COLORS[plant]] * len(treatments), lt)
            ax.tick_params(axis="x", labelsize=6)
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(f"({next(tags)}) {plant}", loc="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)



# ============================================================================
# PART 4 - Example data (SIMULATED around the published means/SDs)
# ============================================================================

def make_example_data(out_dir):
    OUT = Path(out_dir)
    rng = np.random.default_rng(2024)
    OUT.mkdir(parents=True, exist_ok=True)
    N = 3
    TREATMENTS = ["Control", "Chemical fertilizer", "Biofilter (CV/BS 3:1)",
                  "Biofilter (CV/BS 1:1)", "Biofilter control"]
    PLANTS = ["Morning glory", "Sunflower", "Cucumber"]


    def draw(mean, sd, n=N, lo=None):
        x = rng.normal(mean, sd, n)
        return np.clip(x, lo, None) if lo is not None else x


    # ---------------------------------------------------------------- toxicity
    viab = {  # Fig. 2a/b, % cell viability (mean, sd)
        "GG":    {"CV": (138, 20), "BS": (187, 10), "3:1": (148, 8), "1:1": (110, 25), "1:3": (52, 12)},
        "GG/CG": {"CV": (210, 6),  "BS": (240, 50), "3:1": (175, 8), "1:1": (219, 40), "1:3": (104, 8)},
    }
    rows = []
    blank0, blank7, ctrl_growth = 0.050, 0.052, 0.40
    for binder, d in viab.items():
        for inoc, (m, s) in d.items():
            culture = "single" if inoc in ("CV", "BS") else "co-culture"
            for r in range(1, N + 1):
                c0 = 0.80 + rng.normal(0, 0.01)
                c7 = c0 + ctrl_growth + rng.normal(0, 0.01)
                s0 = 0.80 + rng.normal(0, 0.01)
                v = rng.normal(m, s)
                s7 = s0 + (blank7 - blank0) + v / 100 * ((c7 - blank7) - (c0 - blank0))
                rows.append(dict(binder=binder, culture=culture, inoculum=inoc, replicate=r,
                                 sample_od_d0=s0, sample_od_d7=s7,
                                 control_od_d0=c0, control_od_d7=c7,
                                 blank_d0=blank0, blank_d7=blank7))
    pd.DataFrame(rows).round(4).to_csv(OUT / "toxicity.csv", index=False)

    # ------------------------------------------------------------- degradation
    degr = {  # Fig. 2e/f, % degradation
        "GG":    {"CV": (4.7, 3.5), "BS": (3.5, 2.6), "3:1": (5.1, 3.0), "1:1": (5.3, 3.4), "1:3": (4.1, 2.0)},
        "GG/CG": {"CV": (3.9, 1.2), "BS": (2.2, 1.4), "3:1": (3.3, 1.4), "1:1": (3.0, 0.4), "1:3": (4.4, 0.4)},
    }
    rows = []
    for binder, d in degr.items():
        for inoc, (m, s) in d.items():
            culture = "single" if inoc in ("CV", "BS") else "co-culture"
            for r in range(1, N + 1):
                w0 = 0.200 + rng.normal(0, 0.005)
                pct = max(rng.normal(m, s), 0.1)
                rows.append(dict(binder=binder, culture=culture, inoculum=inoc, replicate=r,
                                 dry_weight_d0=w0, dry_weight_d7=w0 * (1 - pct / 100)))
    pd.DataFrame(rows).round(5).to_csv(OUT / "degradation.csv", index=False)

    # --------------------------------------------------------------- nutrients
    CI_NH4, CI_PO4 = 67.1, 29.9           # mg/L, Section 2.5
    days = [0, 1, 3, 5, 7]
    traj = {  # Fig. 3a/b, % removal at each day; initial biomass (g)
        ("1:1", "I"): ([0, 56, 72, 79, 86.7], [0, 58, 96, 98, 99.3], 0.100),
        ("3:1", "I"): ([0, 38, 65, 72, 81.0], [0, 48, 91, 96, 97.5], 0.130),
        ("1:1", "S"): ([0, 38, 43, 49.5, 54.3], [0, 40, 46, 51, 54.6], 0.135),
        ("3:1", "S"): ([0, 36, 40, 46.5, 50.0], [0, 42, 45, 47, 49.0], 0.186),
    }
    rows = []
    for (ratio, system), (nh4, po4, m) in traj.items():
        for r in range(1, N + 1):
            biomass = m * (1 + rng.normal(0, 0.02))
            for day, rn, rp in zip(days, nh4, po4):
                jn = 0 if day == 0 else rng.normal(0, 0.8)
                jp = 0 if day == 0 else rng.normal(0, 0.4)
                rows.append(dict(ratio=ratio, system=system, replicate=r, day=day,
                                 nh4_mg_l=CI_NH4 * (1 - min(rn + jn, 100) / 100),
                                 po4_mg_l=CI_PO4 * (1 - min(rp + jp, 100) / 100),
                                 volume_l=0.05, initial_biomass_g=biomass))
    pd.DataFrame(rows).round(4).to_csv(OUT / "nutrients.csv", index=False)

    # ------------------------------------------------------------------ plants
    # Fig. 4 lengths (cm): total, leaf, branch, stem, root
    lengths = {
        "Morning glory": [(20.5, 21, 23, 27.5, 21), (9.6, 6, 8.6, 7.5, 6), (2.3, 3, 4, 3.2, 2.7),
                          (2.4, 2.3, 2.6, 2.5, 2.4), (7, 6.7, 8, 13.4, 5.3)],
        "Sunflower":     [(11.5, 17.2, 22.8, 23.3, 17.3), (3.7, 3.7, 5.5, 4.3, 4.3), (4.5, 3.8, 2, 2.1, 1.2),
                          (6.3, 9.5, 12, 11, 7.3), (3.3, 4.3, 6, 6.8, 4.3)],
        "Cucumber":      [(12.3, 8.2, 13.4, 12.4, 13.1), (3.8, 3.1, 3.8, 3.5, 4.5), (1.6, 1.6, 2.1, 1.4, 1.5),
                          (2.7, 2.2, 3, 2.8, 2.4), (5, 3.2, 6.8, 5.8, 5.4)],
    }
    # Fig. 5 biomass (g), pH, Chl a (ug/g): (mean, sd) per treatment
    biomass = {"Morning glory": [(1.3, .1), (1.2, .5), (0.9, .2), (1.3, .1), (1.1, .3)],
               "Sunflower":     [(1.5, .05), (1.3, .9), (1.5, .2), (1.5, .1), (1.6, .1)],
               "Cucumber":      [(0.9, .4), (1.7, .1), (2.0, .3), (1.7, .5), (1.9, .2)]}
    chla = {"Morning glory": [(2.0, 1.0), (2.9, 2.6), (9.5, 4.4), (2.5, 1.5), (20.0, 11.3)],
            "Sunflower":     [(19.1, 2.5), (11.6, 4.8), (10.2, 2.6), (6.0, 3.5), (9.5, 1.6)],
            "Cucumber":      [(8.5, 2.8), (19.4, 12.4), (6.3, 4.3), (12.4, .67), (8.5, 2.8)]}
    rows = []
    for plant in PLANTS:
        for t_i, trt in enumerate(TREATMENTS):
            for r in range(1, N + 1):
                parts = [max(draw(m[t_i], 0.15 * m[t_i], 1)[0], 0.1) for m in lengths[plant]]
                dw = max(draw(*biomass[plant][t_i], 1)[0], 0.2)
                chl = max(draw(*chla[plant][t_i], 1)[0], 0.5)
                a750, a630, a645 = 0.010, 0.030, 0.050
                # back-solve A663 so the paper's equation returns `chl`
                a663 = a750 + (chl * dw + 2.16 * (a645 - a750) + 0.1 * (a630 - a750)) / 11.62
                rows.append(dict(plant=plant, treatment=trt, replicate=r,
                                 total_height_cm=parts[0], leaf_cm=parts[1], branch_cm=parts[2],
                                 stem_cm=parts[3], root_cm=parts[4], dry_biomass_g=dw,
                                 A630=a630, A645=a645, A663=a663, A750=a750))
    pd.DataFrame(rows).round(4).to_csv(OUT / "plants.csv", index=False)

    # -------------------------------------------------------------------- soil
    ph = {"Morning glory": [(10.2, .2), (8.9, .4), (9.6, .6), (9.3, .2), (9.4, .2)],
          "Sunflower":     [(7.6, .4), (7.4, .2), (8.0, .3), (8.1, .3), (8.2, .05)],
          "Cucumber":      [(8.2, .05), (7.5, .2), (8.4, .1), (8.2, .1), (8.2, .1)]}
    npk = {  # (N, P, K) mg/kg means; Fig. 6
        "Morning glory": [(6.1, .22, .3), (7.0, .82, 1.9), (5.9, .24, .2), (6.0, .22, .3), (6.0, .26, .4)],
        "Sunflower":     [(6.0, .28, .2), (6.6, .82, 1.7), (6.0, .24, .2), (6.0, .26, .4), (6.1, .31, .6)],
        "Cucumber":      [(6.2, .40, .5), (6.2, .50, .5), (6.1, .30, .2), (6.2, .30, .04), (5.8, .30, .7)],
    }
    W = 0.015            # 15 g soil
    ACID_N = 0.01        # 0.01 M HCl
    V1, M, V2, V3, H = 3.0, 0.87, 20.0, 2.0, 0.0
    rows = []
    for plant in PLANTS:
        for t_i, trt in enumerate(TREATMENTS):
            n_m, p_m, k_m = npk[plant][t_i]
            for r in range(1, N + 1):
                n_val = max(draw(n_m, 0.04 * n_m, 1)[0], 0.01)
                p_val = max(draw(p_m, 0.15 * p_m, 1)[0], 0.005)
                k_val = max(draw(k_m, 0.25 * k_m, 1)[0], 0.005)
                rows.append(dict(
                    plant=plant, treatment=trt, replicate=r,
                    soil_ph=draw(*ph[plant][t_i], 1)[0],
                    soil_weight_kg=W,
                    n_titrant_volume=n_val * W / (ACID_N * 14.01 * 100),
                    n_acid_normality=ACID_N,
                    p_conc_mg_ml=p_val * (100 - H) * W / ((V1 / M) * (V2 / V3) * 1000),
                    p_v_sample=V1, p_acid_molarity=M, p_v_naoh=V2, p_v_molybdate=V3,
                    p_dilution_pct=H,
                    k_od=k_val * W / (1.2 * 100)))
    pd.DataFrame(rows).to_csv(OUT / "soil.csv", index=False, float_format="%.6g")

    print(f"Example data written to {OUT}")
    return OUT



# ============================================================================
# PART 5 - Statistics pipeline (Sections 3.1-3.8)
# ============================================================================

TREATMENTS = ["Control", "Chemical fertilizer", "Biofilter (CV/BS 3:1)",
              "Biofilter (CV/BS 1:1)", "Biofilter control"]
CO_RATIOS = ["3:1", "1:1", "1:3"]
# Section 2.13: morning glory was normal (ANOVA); cucumber & sunflower were not (Kruskal-Wallis)
PAPER_METHOD = {"Morning glory": "anova", "Sunflower": "kruskal", "Cucumber": "kruskal"}


def mean_sd(df, keys, cols):
    g = df.groupby(keys, observed=True, sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return out.reset_index()


def cmd_analysis(argv=None):
    ap = argparse.ArgumentParser(description=ANALYSIS_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT / "data" / "example", type=Path)
    ap.add_argument("--out", default=ROOT / "results" / "python", type=Path)
    ap.add_argument("--paper-methods", action="store_true",
                    help="use ANOVA/Kruskal-Wallis per plant as in the paper instead of "
                         "choosing by Shapiro-Wilk")
    ap.add_argument("--chl-extract-ml", type=float, default=1.0,
                    help="ethanol extract volume (mL) for Chl a; 1.0 reproduces the paper")
    a = ap.parse_args(argv)
    ensure_data(a.data)

    fig_dir = a.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = []

    def log(text):
        print(text)
        report.append(text)

    # ------------------------------------------------------ 3.1 toxicity
    tox = pd.read_csv(a.data / "toxicity.csv")
    tox["viability_pct"] = cell_viability(tox.sample_od_d0, tox.sample_od_d7,
                                             tox.control_od_d0, tox.control_od_d7,
                                             tox.blank_d0, tox.blank_d7)
    tox.to_csv(a.out / "toxicity_processed.csv", index=False)
    log("=" * 70 + "\n3.1 CELL VIABILITY (toxicity test)\n" + "=" * 70)
    log(factorial_anova(tox, "viability_pct ~ C(binder) * C(inoculum)").round(4).to_string())
    tox_letters = {}
    for b, g in tox.groupby("binder", sort=False):
        res = compare_groups(g[g.inoculum.isin(CO_RATIOS)], "viability_pct", "inoculum")
        tox_letters[b] = res["letters"]
        log(format_result(f"Viability, {b} binder, mixing ratios", res))

    # --------------------------------------------------- 3.2 degradation
    deg = pd.read_csv(a.data / "degradation.csv")
    deg["degradation_pct"] = degradation(deg.dry_weight_d0, deg.dry_weight_d7)
    deg.to_csv(a.out / "degradation_processed.csv", index=False)
    log("=" * 70 + "\n3.2 DEGRADATION OF LIVING BIOFILTERS\n" + "=" * 70)
    deg["degradation_asin"] = arcsine_transform(deg.degradation_pct)
    log(factorial_anova(deg, "degradation_asin ~ C(binder) * C(inoculum)").round(4).to_string())
    deg_letters = {}
    for b, g in deg.groupby("binder", sort=False):
        res = compare_groups(g[g.inoculum.isin(CO_RATIOS)], "degradation_pct", "inoculum",
                                transform="arcsine")
        deg_letters[b] = res["letters"]
        log(format_result(f"Degradation, {b} binder, mixing ratios", res))

    # ------------------------------------------------- decision matrix
    v = tox.groupby(["binder", "inoculum"]).viability_pct.mean()
    d = deg.groupby(["binder", "inoculum"]).degradation_pct.mean()
    dm = pd.concat([v, d], axis=1).reset_index()
    dm = dm[dm.inoculum.isin(CO_RATIOS)].copy()
    dm["score"] = decision_matrix(dm.viability_pct, dm.degradation_pct)
    dm = dm.sort_values("score", ascending=False)
    dm.to_csv(a.out / "decision_matrix.csv", index=False)
    log("Decision matrix (higher score = better binder/ratio):\n" + dm.round(3).to_string(index=False) + "\n")

    tox_sum = tox.groupby(["binder", "inoculum"], sort=False).viability_pct.agg(["mean", "std"])
    tox_sum = tox_sum.rename(columns={"std": "sd"}).reset_index()
    fig_binder_screen(tox_sum, "viability_pct", "% Cell viability", tox_letters,
                            fig_dir / "fig2ab_cell_viability.png")
    deg_sum = deg.groupby(["binder", "inoculum"], sort=False).degradation_pct.agg(["mean", "std"])
    deg_sum = deg_sum.rename(columns={"std": "sd"}).reset_index()
    fig_binder_screen(deg_sum, "degradation_pct", "% Degradation", deg_letters,
                            fig_dir / "fig2ef_degradation.png")

    # ------------------------------------------------- 3.3 nutrients
    nut = pd.read_csv(a.data / "nutrients.csv")
    ci = nut[nut.day == 0].set_index(["ratio", "system", "replicate"])[["nh4_mg_l", "po4_mg_l"]]
    nut = nut.join(ci, on=["ratio", "system", "replicate"], rsuffix="_initial")
    for n in ("nh4", "po4"):
        nut[f"{n}_removal_pct"] = removal_efficiency(nut[f"{n}_mg_l_initial"], nut[f"{n}_mg_l"])
        nut[f"{n}_rate"] = absorption_rate(nut[f"{n}_mg_l_initial"], nut[f"{n}_mg_l"],
                                              nut.volume_l, nut.day, nut.initial_biomass_g)
    nut["group"] = nut.ratio + "_CV/BS_" + nut.system
    nut["subject"] = nut.group + "_r" + nut.replicate.astype(str)
    nut.to_csv(a.out / "nutrients_processed.csv", index=False)
    nut_sum = mean_sd(nut, ["ratio", "system", "day"],
                      ["nh4_removal_pct", "po4_removal_pct", "nh4_rate", "po4_rate"])
    nut_sum.to_csv(a.out / "nutrients_summary.csv", index=False)

    log("=" * 70 + "\n3.3 NUTRIENT REMOVAL\n" + "=" * 70)
    last = nut[nut.day == nut.day.max()]
    for n, lab in (("nh4", "NH4+"), ("po4", "PO4 3-")):
        log(f"Day {int(nut.day.max())} {lab} removal (mean +/- SD):\n"
            + last.groupby("group")[f"{n}_removal_pct"].agg(["mean", "std"]).round(2).to_string())
        res = compare_groups(last, f"{n}_removal_pct", "group", method="kruskal")
        log(format_result(f"{lab} removal on final day", res))
        rm = nut[nut.day > 0].copy()
        rm[f"{n}_asin"] = arcsine_transform(rm[f"{n}_removal_pct"])
        log(f"Two-way repeated-measures (mixed) ANOVA, {lab} removal (arcsine):\n"
            + mixed_anova(rm, f"{n}_asin", "day", "group", "subject").round(4).to_string() + "\n")
    fig_nutrients(nut_sum, fig_dir / "fig3_nutrient_removal.png")

    # ------------------------------------------------- 3.4-3.7 plants
    pl = pd.read_csv(a.data / "plants.csv")
    pl["chl_a_ug_g"] = chlorophyll_a(pl.A630, pl.A645, pl.A663, pl.A750,
                                       pl.dry_biomass_g, a.chl_extract_ml)
    soil = pd.read_csv(a.data / "soil.csv")
    soil["total_n_mg_kg"] = total_n(soil.n_titrant_volume, soil.n_acid_normality, soil.soil_weight_kg)
    soil["total_p_mg_kg"] = total_p(soil.p_conc_mg_ml, soil.p_v_sample, soil.p_acid_molarity,
                                       soil.p_v_naoh, soil.p_v_molybdate, soil.p_dilution_pct,
                                       soil.soil_weight_kg)
    soil["total_k_mg_kg"] = total_k(soil.k_od, soil.soil_weight_kg)
    pl = pl.merge(soil[["plant", "treatment", "replicate", "soil_ph",
                        "total_n_mg_kg", "total_p_mg_kg", "total_k_mg_kg"]],
                  on=["plant", "treatment", "replicate"], how="left")
    pl.to_csv(a.out / "plants_processed.csv", index=False)

    variables = {
        "total_height_cm": "Total plant height (cm)", "leaf_cm": "Leaf (cm)",
        "branch_cm": "Branch (cm)", "stem_cm": "Stem (cm)", "root_cm": "Root (cm)",
        "dry_biomass_g": "Dry biomass (g)", "soil_ph": "Soil pH",
        "chl_a_ug_g": "Chl a (ug/g plant biomass)", "total_n_mg_kg": "Soil total N (mg/kg)",
        "total_p_mg_kg": "Soil total P (mg/kg)", "total_k_mg_kg": "Soil total K (mg/kg)",
    }
    pl_sum = mean_sd(pl, ["plant", "treatment"], list(variables))
    pl_sum.to_csv(a.out / "plants_summary.csv", index=False)

    log("=" * 70 + "\n3.4-3.8 PLANT GROWTH, SOIL pH, CHLOROPHYLL a, SOIL N/P/K\n" + "=" * 70)
    letters, rows = {}, []
    for plant, g in pl.groupby("plant", sort=False):
        method = PAPER_METHOD.get(plant, "auto") if a.paper_methods else "auto"
        for var, lab in variables.items():
            res = compare_groups(g, var, "treatment", method=method)
            letters[(plant, var)] = res["letters"]
            log(format_result(f"{plant} - {lab}", res))
            rows.append(dict(plant=plant, variable=var, test=res["test"],
                             statistic=res["statistic"], df=str(res["df"]), p=res["p"],
                             **{f"letter_{k}": v for k, v in res["letters"].items()}))
    pd.DataFrame(rows).to_csv(a.out / "plants_statistics.csv", index=False)

    fig_plant_lengths(pl_sum, letters, TREATMENTS, fig_dir / "fig4_plant_lengths.png")
    fig_grid(pl_sum, ["dry_biomass_g", "soil_ph", "chl_a_ug_g"],
                   ["Biomass (g)", "pH values", "Chl a per weight of biomass (ug/g)"],
                   letters, TREATMENTS, fig_dir / "fig5_biomass_ph_chla.png")
    fig_grid(pl_sum, ["total_n_mg_kg", "total_p_mg_kg", "total_k_mg_kg"],
                   ["Total N (mg/kg soil)", "Total P (mg/kg soil)", "Total K (mg/kg soil)"],
                   letters, TREATMENTS, fig_dir / "fig6_soil_npk.png")

    (a.out / "statistics_report.txt").write_text("\n".join(report))
    print(f"\nDone. Results in {a.out}")



# ============================================================================
# PART 6 - Kinetic bioreactor model (C. vulgaris + B. subtilis)
# ============================================================================

# molar-mass conversions between ion and element
NH4_TO_N = 14.007 / 18.038
PO4_TO_P = 30.974 / 94.971


@dataclass
class Params:
    # --- algae (C. vulgaris) ---
    mu_max_a: float = 1.5      # 1/d   max specific growth rate        (fitted)
    b_a: float = 0.05          # 1/d   decay
    K_Na: float = 1.0          # mg N/L  half-saturation, ammonium
    i_N_a: float = 0.08        # g N / g algae
    Q_min: float = 0.002       # g P / g algae  minimum P quota
    Q_max: float = 0.030       # g P / g algae  maximum P quota
    rho_max_P: float = 15.0    # mg P / (g algae d)  max P uptake    (fitted)
    K_P: float = 0.5           # mg P/L
    Y_O2_a: float = 1.3        # g O2 released / g algae grown
    # --- light ---
    I0_lux: float = 2000.0     # lx at the culture surface (Section 2.1)
    lux_to_umol: float = 0.0135  # umol m-2 s-1 per lx (white LED, approx.)
    K_I: float = 20.0          # umol m-2 s-1  light half-saturation
    K_a: float = 0.15          # m2 / g  specific light attenuation of biomass
    light_fraction: float = 0.5  # 12:12 h light:dark
    # --- bacteria (B. subtilis) ---
    mu_max_b: float = 2.0      # 1/d   (fitted)
    b_b: float = 0.1           # 1/d
    K_G: float = 20.0          # mg glucose/L
    K_O: float = 0.5           # mg O2/L
    K_Nb: float = 1.0          # mg N/L
    K_Pb: float = 0.1          # mg P/L
    Y_G: float = 0.5           # g biomass / g glucose
    i_N_b: float = 0.12        # g N / g bacteria
    i_P_b: float = 0.02        # g P / g bacteria
    # --- fast sorption / precipitation stage ---
    q_max_N: float = 5.0       # mg N / g biomass   (fitted)
    q_max_P: float = 3.0       # mg P / g biomass   (fitted)
    K_s: float = 5.0           # mg/L
    k_s: float = 10.0          # 1/d  (fast, approaches equilibrium within hours)
    # --- gas exchange ---
    kLa: float = 5.0           # 1/d  (closed bottle on orbital shaker, 40 rpm)
    O_sat: float = 8.2         # mg/L at 25 C

    def to_dict(self):
        return asdict(self)


@dataclass
class Reactor:
    """Geometry and operating conditions."""
    mode: str = "immobilized"          # "immobilized", "suspended" or "pbr"
    V_L: float = 0.050                 # L liquid
    V_cells: float = 0.0032            # L volume the cells occupy (gel layer 4 x 8 x 0.1 cm)
    depth_m: float = 0.001             # m light path through the cell layer
    bacteria: bool = True
    init: dict = field(default_factory=dict)

    @classmethod
    def wastewater(cls, mode, biomass_g, algae_fraction, Q0=0.01,
                   nh4_mg_l=67.1, po4_mg_l=29.9, glucose_mg_l=450.0):
        """Bottle of 50 mL synthetic wastewater (Section 2.5)."""
        if mode == "immobilized":
            v_cells, depth = 0.0032, 0.001       # 4 x 8 cm textile, ~1 mm gel
        else:
            v_cells, depth = 0.050, 0.0156       # cells in 50 mL, 1.56 cm liquid depth
        init = dict(Xa=biomass_g * algae_fraction, Qp=Q0,
                    Xb=biomass_g * (1 - algae_fraction),
                    N=nh4_mg_l * NH4_TO_N, P=po4_mg_l * PO4_TO_P,
                    G=glucose_mg_l, O=8.2, SN=0.0, SP=0.0)
        return cls(mode=mode, V_L=0.050, V_cells=v_cells, depth_m=depth,
                   bacteria=True, init=init)

    @classmethod
    def bg11_pbr(cls, volume_l=5.0, depth_m=0.08, x0_g_l=0.05, Q0=0.01):
        """C. vulgaris stock culture in BG-11 (Section 2.1): NaNO3 1.5 g/L, K2HPO4 0.04 g/L."""
        n_mg_l = 1500 * 14.007 / 84.995           # ~247 mg N/L (nitrate, treated as N source)
        p_mg_l = 40 * 30.974 / 174.18             # ~7.1 mg P/L
        init = dict(Xa=x0_g_l * volume_l, Qp=Q0, Xb=0.0, N=n_mg_l, P=p_mg_l, G=0.0, O=8.2, SN=0.0, SP=0.0)
        return cls(mode="pbr", V_L=volume_l, V_cells=volume_l, depth_m=depth_m,
                   bacteria=False, init=init)


STATE = ["Xa", "Qp", "Xb", "N", "P", "G", "O", "SN", "SP"]


def light_factor(p: Params, Xa_g, reactor: Reactor, t):
    """Photoperiod x depth-averaged light limitation."""
    on = (t % 1.0) < p.light_fraction
    if not on:
        return 0.0
    I0 = p.I0_lux * p.lux_to_umol
    conc = max(Xa_g, 0.0) / reactor.V_cells * 1000.0       # g/m3
    tau = p.K_a * conc * reactor.depth_m
    I_av = I0 if tau < 1e-9 else I0 * (1 - np.exp(-tau)) / tau
    return I_av / (p.K_I + I_av)


def rates(t, y, p: Params, r: Reactor):
    Xa, Qp, Xb, N, P, G, O, SN, SP = np.maximum(y, 0.0)
    Qp = max(Qp, p.Q_min * 1.0001)
    fI = light_factor(p, Xa, r, t)

    # algae
    mu_a = p.mu_max_a * fI * N / (p.K_Na + N) * (1 - p.Q_min / Qp)
    rho_P = p.rho_max_P * P / (p.K_P + P) * max(p.Q_max - Qp, 0) / (p.Q_max - p.Q_min)  # mg P/g/d
    # bacteria
    if r.bacteria and Xb > 0:
        mu_b = (p.mu_max_b * G / (p.K_G + G) * O / (p.K_O + O)
                * N / (p.K_Nb + N) * P / (p.K_Pb + P))
    else:
        mu_b = 0.0

    growth_a = mu_a * Xa                     # g/d
    growth_b = mu_b * Xb                     # g/d
    dXa = growth_a - p.b_a * Xa
    # quota dilution by growth + uptake (g P/g/d)
    dQp = rho_P / 1000.0 - mu_a * Qp
    dXb = growth_b - p.b_b * Xb

    VL = r.V_L
    X = Xa + Xb
    dSN = p.k_s * (p.q_max_N * X * N / (p.K_s + N) - SN)                    # mg/d
    dSP = p.k_s * (p.q_max_P * X * P / (p.K_s + P) - SP)
    dN = -(p.i_N_a * growth_a + p.i_N_b * growth_b) * 1000.0 / VL - dSN / VL  # mg/L/d
    dP = -(rho_P * Xa + p.i_P_b * growth_b * 1000.0) / VL - dSP / VL
    dG = -(growth_b / p.Y_G) * 1000.0 / VL
    # O2 demand of heterotrophic growth ~ (1 - Y_G)/Y_G * 1.07 g O2 / g biomass
    o2_resp = growth_b * (1 - p.Y_G) / p.Y_G * 1.07
    dO = (p.Y_O2_a * growth_a - o2_resp) * 1000.0 / VL + p.kLa * (p.O_sat - O)
    return [dXa, dQp, dXb, dN, dP, dG, dO, dSN, dSP]


def simulate(p: Params, r: Reactor, t_end=7.0, n_out=None):
    """Integrate the model; returns a dict of arrays (t in days)."""
    y0 = [r.init[k] for k in STATE]
    t_eval = np.linspace(0, t_end, n_out or int(t_end * 48) + 1)
    sol = solve_ivp(rates, (0, t_end), y0, args=(p, r), method="LSODA",
                    t_eval=t_eval, max_step=0.02, rtol=1e-6, atol=1e-9)
    out = {"t": sol.t}
    out.update({k: sol.y[i] for i, k in enumerate(STATE)})
    out["NH4_mg_l"] = out["N"] / NH4_TO_N
    out["PO4_mg_l"] = out["P"] / PO4_TO_P
    out["nh4_removal_pct"] = (1 - out["N"] / r.init["N"]) * 100
    out["po4_removal_pct"] = (1 - out["P"] / r.init["P"]) * 100 if r.init["P"] > 0 else 0
    out["Xa_g_l"] = out["Xa"] / r.V_L
    return out



# ============================================================================
# PART 7 - Calibration of the bioreactor model
# ============================================================================

ALGAE_FRACTION = {"3:1": 0.75, "1:1": 0.50, "1:3": 0.25}
MODE = {"I": "immobilized", "S": "suspended"}

# parameters estimated from data; everything else stays at the literature-type default
FIT = {"mu_max_a": (0.05, 10.0), "rho_max_P": (0.5, 200.0), "mu_max_b": (0.01, 20.0),
       "q_max_N": (0.1, 100.0), "q_max_P": (0.05, 50.0)}


def observed_means(nutrients: pd.DataFrame) -> pd.DataFrame:
    """Mean removal (%) per ratio, system and day, plus mean initial biomass and C0."""
    d = nutrients.copy()
    c0 = d[d.day == 0].groupby(["ratio", "system", "replicate"])[["nh4_mg_l", "po4_mg_l"]].first()
    d = d.join(c0, on=["ratio", "system", "replicate"], rsuffix="_0")
    d["nh4_removal_pct"] = (1 - d.nh4_mg_l / d.nh4_mg_l_0) * 100
    d["po4_removal_pct"] = (1 - d.po4_mg_l / d.po4_mg_l_0) * 100
    return (d.groupby(["ratio", "system", "day"], as_index=False)
              .agg(nh4_removal_pct=("nh4_removal_pct", "mean"),
                   po4_removal_pct=("po4_removal_pct", "mean"),
                   biomass_g=("initial_biomass_g", "mean"),
                   nh4_0=("nh4_mg_l_0", "mean"), po4_0=("po4_mg_l_0", "mean")))


def reactor_for(group: pd.DataFrame, system: str, ratio: str) -> Reactor:
    g0 = group.iloc[0]
    return Reactor.wastewater(MODE[system], g0.biomass_g, ALGAE_FRACTION[ratio],
                              nh4_mg_l=g0.nh4_0, po4_mg_l=g0.po4_0)


def predict(p: Params, obs: pd.DataFrame, system: str) -> pd.DataFrame:
    rows = []
    for ratio, g in obs[obs.system == system].groupby("ratio"):
        out = simulate(p, reactor_for(g, system, ratio), t_end=float(g.day.max()))
        for day in g.day:
            i = int(np.argmin(np.abs(out["t"] - day)))
            rows.append(dict(ratio=ratio, system=system, day=day,
                             nh4_pred=out["nh4_removal_pct"][i],
                             po4_pred=out["po4_removal_pct"][i]))
    return obs[obs.system == system].merge(pd.DataFrame(rows), on=["ratio", "system", "day"])


def fit_system(obs: pd.DataFrame, system: str, base: Params | None = None):
    """Least-squares fit (log-scaled) of FIT parameters for one system ('I' or 'S')."""
    base = base or Params()
    names = list(FIT)
    lo = np.log([FIT[n][0] for n in names])
    hi = np.log([FIT[n][1] for n in names])
    x0 = np.log([getattr(base, n) for n in names])

    def resid(logx):
        p = replace(base, **dict(zip(names, np.exp(logx))))
        pr = predict(p, obs, system)
        pr = pr[pr.day > 0]
        return np.concatenate([pr.nh4_pred - pr.nh4_removal_pct,
                               pr.po4_pred - pr.po4_removal_pct])

    res = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi),
                        diff_step=0.05, max_nfev=200)
    p = replace(base, **dict(zip(names, np.exp(res.x))))
    r = resid(res.x)
    stats = dict(rmse_pct=float(np.sqrt(np.mean(r ** 2))), n_points=int(r.size),
                 converged=bool(res.success))
    return p, stats



# ============================================================================
# PART 8 - Bioreactor runs: fit, figures, BG-11 photobioreactor
# ============================================================================

STYLE = {("3:1", "I"): ("#C00000", "-"), ("3:1", "S"): ("#C00000", "--"),
         ("1:1", "I"): ("#2E75B6", "-"), ("1:1", "S"): ("#2E75B6", "--")}


def write_comsol_params(params_by_system, path):
    """COMSOL parameter file: name, expression with unit, description."""
    p = params_by_system["I"]
    rows = [
        ("mu_max_a", f"{p.mu_max_a:.6g}[1/d]", "Algae max specific growth rate (fitted, immobilized)"),
        ("b_a", f"{p.b_a:.6g}[1/d]", "Algae decay rate"),
        ("K_Na", f"{p.K_Na:.6g}[g/m^3]", "Ammonium-N half-saturation, algae"),
        ("i_N_a", f"{p.i_N_a:.6g}", "N content of algae (g N/g)"),
        ("Q_min", f"{p.Q_min:.6g}", "Minimum algal P quota (g P/g)"),
        ("Q_max", f"{p.Q_max:.6g}", "Maximum algal P quota (g P/g)"),
        ("rho_max_P", f"{p.rho_max_P / 1000:.6g}[1/d]", "Max P uptake (g P/g algae/d, fitted)"),
        ("K_P", f"{p.K_P:.6g}[g/m^3]", "Phosphate-P half-saturation"),
        ("Y_O2_a", f"{p.Y_O2_a:.6g}", "O2 released per algae grown (g/g)"),
        ("I0", f"{p.I0_lux * p.lux_to_umol:.6g}", "Light at biofilter surface, umol m-2 s-1 (2000 lx)"),
        ("K_I", f"{p.K_I:.6g}", "Light half-saturation, umol m-2 s-1"),
        ("K_a", f"{p.K_a:.6g}[m^2/g]", "Specific light attenuation of biomass"),
        ("mu_max_b", f"{p.mu_max_b:.6g}[1/d]", "Bacteria max specific growth rate (fitted)"),
        ("b_b", f"{p.b_b:.6g}[1/d]", "Bacteria decay rate"),
        ("K_G", f"{p.K_G:.6g}[g/m^3]", "Glucose half-saturation"),
        ("K_O", f"{p.K_O:.6g}[g/m^3]", "O2 half-saturation"),
        ("K_Nb", f"{p.K_Nb:.6g}[g/m^3]", "Ammonium-N half-saturation, bacteria"),
        ("K_Pb", f"{p.K_Pb:.6g}[g/m^3]", "Phosphate-P half-saturation, bacteria"),
        ("Y_G", f"{p.Y_G:.6g}", "Bacterial yield on glucose (g/g)"),
        ("i_N_b", f"{p.i_N_b:.6g}", "N content of bacteria (g N/g)"),
        ("i_P_b", f"{p.i_P_b:.6g}", "P content of bacteria (g P/g)"),
        ("q_max_N", f"{p.q_max_N / 1000:.6g}", "Fast-sorption capacity N (g N/g biomass, fitted)"),
        ("q_max_P", f"{p.q_max_P / 1000:.6g}", "Fast-sorption capacity P (g P/g biomass, fitted)"),
        ("K_s", f"{p.K_s:.6g}[g/m^3]", "Sorption half-saturation"),
        ("k_s", f"{p.k_s:.6g}[1/d]", "Sorption rate constant"),
        ("kLa", f"{p.kLa:.6g}[1/d]", "O2 transfer coefficient, liquid"),
        ("O_sat", f"{p.O_sat:.6g}[g/m^3]", "O2 saturation at 25 C"),
    ]
    with open(path, "w") as fh:
        for name, expr, desc in rows:
            fh.write(f'{name} {expr} "{desc}"\n')


def cmd_bioreactor(argv=None):
    ap = argparse.ArgumentParser(description=BIOREACTOR_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT / "data" / "example", type=Path)
    ap.add_argument("--out", default=ROOT / "results" / "bioreactor", type=Path)
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--pbr-days", type=float, default=14.0)
    ap.add_argument("--pbr-volume", type=float, default=5.0, help="L")
    ap.add_argument("--pbr-depth", type=float, default=0.08, help="light path, m")
    ap.add_argument("--pbr-lux", type=float, default=2000.0)
    ap.add_argument("--pbr-mu-max", type=float, default=1.5,
                    help="C. vulgaris max growth rate in BG-11 (1/d); literature range ~1-2")
    a = ap.parse_args(argv)
    ensure_data(a.data)
    a.out.mkdir(parents=True, exist_ok=True)
    pfile = a.out / "fitted_parameters.json"

    obs = observed_means(pd.read_csv(a.data / "nutrients.csv"))

    # ------------------------------------------------------------ calibration
    if a.skip_fit and pfile.exists():
        saved = json.loads(pfile.read_text())
        params = {s: Params(**saved[s]["params"]) for s in ("I", "S")}
        print(f"Loaded parameters from {pfile}")
    else:
        params, saved = {}, {}
        for s in ("I", "S"):
            print(f"Fitting system {s} ...")
            params[s], st = fit_system(obs, s)
            saved[s] = {"params": params[s].to_dict(), "fit": st,
                        "fitted": list(FIT)}
            print(f"  RMSE = {st['rmse_pct']:.2f} % removal over {st['n_points']} points")
        pfile.write_text(json.dumps(saved, indent=2))
    write_comsol_params(params, a.out / "parameters_fitted.txt")

    table = pd.DataFrame({s: {k: getattr(params[s], k) for k in FIT} for s in params})
    table.columns = ["Immobilized (I)", "Suspended (S)"]
    table.to_csv(a.out / "fitted_parameters.csv")
    print("\nFitted parameters:\n" + table.round(4).to_string())

    # ------------------------------------------------------ model vs data
    sims, rows = {}, []
    for (ratio, system), g in obs.groupby(["ratio", "system"]):
        out = simulate(params[system], reactor_for(g, system, ratio), t_end=7.0)
        sims[(ratio, system)] = out
        pd.DataFrame(out).to_csv(a.out / f"simulation_{ratio.replace(':', 'to')}_{system}.csv", index=False)
        for _, r in g.iterrows():
            i = int(np.argmin(np.abs(out["t"] - r.day)))
            rows.append(dict(ratio=ratio, system=system, day=r.day,
                             nh4_obs=r.nh4_removal_pct, nh4_model=out["nh4_removal_pct"][i],
                             po4_obs=r.po4_removal_pct, po4_model=out["po4_removal_pct"][i]))
    pd.DataFrame(rows).round(3).to_csv(a.out / "model_vs_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, n, lab in zip(axes, ("nh4", "po4"), ("NH$_4^+$", "PO$_4^{3-}$")):
        for key, out in sims.items():
            c, ls = STYLE[key]
            g = obs[(obs.ratio == key[0]) & (obs.system == key[1])]
            ax.plot(out["t"], out[f"{n}_removal_pct"], color=c, ls=ls, label=f"{key[0]}_CV/BS_{key[1]} model")
            ax.plot(g.day, g[f"{n}_removal_pct"], "o" if key[1] == "I" else "s", color=c,
                    mfc=c if key[1] == "I" else "white")
        ax.set(xlabel="Time (day)", ylabel=f"% {lab} removal", ylim=(0, 105))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Kinetic model (lines) vs measured means (markers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "model_fit.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for key, out in sims.items():
        c, ls = STYLE[key]
        lab = f"{key[0]}_CV/BS_{key[1]}"
        axes[0, 0].plot(out["t"], out["Xa"] * 1000, color=c, ls=ls, label=lab)
        axes[0, 1].plot(out["t"], out["Qp"] * 100, color=c, ls=ls)
        axes[1, 0].plot(out["t"], out["O"], color=c, ls=ls)
        axes[1, 1].plot(out["t"], out["G"], color=c, ls=ls)
    for ax, yl in zip(axes.flat, ["Algal biomass (mg)", "Algal P quota (% DW)",
                                   "Dissolved O$_2$ (mg/L)", "Glucose (mg/L)"]):
        ax.set(xlabel="Time (day)", ylabel=yl)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "model_states.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------- photobioreactor scenario
    # axenic culture: literature kinetics, not the wastewater fit (which lumps stress effects)
    p_pbr = Params(mu_max_a=a.pbr_mu_max, I0_lux=a.pbr_lux, kLa=50.0, q_max_N=0.0, q_max_P=0.0)
    pbr = Reactor.bg11_pbr(volume_l=a.pbr_volume, depth_m=a.pbr_depth)
    out = simulate(p_pbr, pbr, t_end=a.pbr_days)
    pd.DataFrame(out).to_csv(a.out / "pbr_bg11_simulation.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(out["t"], out["Xa_g_l"], color="#548235")
    ax[0].set(xlabel="Time (day)", ylabel="C. vulgaris (g/L)",
              title=f"BG-11, {a.pbr_volume:g} L, {a.pbr_lux:g} lx, 12:12 h")
    ax[1].plot(out["t"], out["N"], label="N (mg/L)")
    ax[1].plot(out["t"], out["P"] * 10, label="P x10 (mg/L)")
    ax[1].set(xlabel="Time (day)", ylabel="Medium (mg/L)")
    ax[1].legend(frameon=False)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out / "pbr_bg11.png", dpi=300)
    plt.close(fig)
    print(f"\nPBR: {out['Xa_g_l'][0]:.3f} -> {out['Xa_g_l'][-1]:.3f} g/L after {a.pbr_days:g} days")
    print(f"Results in {a.out}")



# ============================================================================
# PART 9 - COMSOL model equations solved in 1D (no licence needed)
# ============================================================================

UNIT = {"[1/d]": 1 / 86400, "[g/m^3]": 1.0, "[m^2/g]": 1.0}

M = dict(N=14.007, P=30.974, G=180.16, O2=31.998, X=24.6, NH4=18.038, PO4=94.971)


def load_params(path):
    p = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+)\s+([-0-9.eE+]+)(\[[^\]]*\])?", line)
        if m:
            p[m.group(1)] = float(m.group(2)) * UNIT.get(m.group(3) or "", 1.0)
    return p


def cmd_comsol1d(argv=None):
    ap = argparse.ArgumentParser(description=COMSOL1D_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=ROOT / "results" / "bioreactor" / "parameters_fitted.txt")
    ap.add_argument("--f-gel", type=float, default=0.6)
    ap.add_argument("--t-b", type=float, default=1e-3, help="gel thickness, m")
    ap.add_argument("--n", type=int, default=20, help="grid cells through the gel")
    ap.add_argument("--f-algae", type=float, default=0.5)
    ap.add_argument("--m-bio", type=float, default=0.1, help="g")
    a = ap.parse_args(argv)
    if not Path(a.params).exists():
        print("No fitted parameters yet - running the bioreactor fit first.")
        cmd_bioreactor([])
    k = load_params(a.params)

    W, L, VL = 0.04, 0.08, 50e-6                       # m, m, m3
    tb, n = a.t_b, a.n
    Vbf = W * L * tb
    dy = tb / n
    y = (np.arange(n) + 0.5) * dy                      # cell centres (0 = bottom)
    A = W * L                                          # interface area
    D = a.f_gel * np.array([1.96e-9, 0.76e-9, 0.67e-9, 2.1e-9])  # N, P, G, O2

    c0 = np.array([67.1 / M["NH4"], 29.9 / M["PO4"], 450 / M["G"], k["O_sat"] / M["O2"]])
    Xa0 = a.m_bio * a.f_algae / Vbf / M["X"]
    Xb0 = a.m_bio * (1 - a.f_algae) / Vbf / M["X"]
    PA0 = 0.01 * Xa0 * M["X"] / M["P"]
    # state: gel solutes (4 x n), biomass (5 x n), liquid solutes (4)
    y0 = np.concatenate([np.repeat(c0, n), np.repeat([Xa0, Xb0, PA0, 0, 0], n), c0])

    def rhs(t, s):
        c = np.maximum(s[:4 * n].reshape(4, n), 0)
        b = s[4 * n:9 * n].reshape(5, n)
        cl = np.maximum(s[9 * n:], 0)
        Xa, Xb, PA, SN, SP = np.maximum(b, 0)
        Xa_g, Xb_g = Xa * M["X"], Xb * M["X"]
        N_g, P_g, G_g, O_g = c[0] * M["N"], c[1] * M["P"], c[2] * M["G"], c[3] * M["O2"]
        Qp = np.maximum(PA * M["P"] / np.maximum(Xa_g, 1e-6), k["Q_min"] * 1.0001)
        photo = 0.5 * (1 + np.tanh(np.sin(2 * np.pi * t / 86400) / 0.05))
        I_loc = k["I0"] * np.exp(-k["K_a"] * Xa_g * (tb - y))
        f_I = photo * I_loc / (k["K_I"] + I_loc)
        mu_a = k["mu_max_a"] * f_I * N_g / (k["K_Na"] + N_g) * (1 - k["Q_min"] / Qp)
        rho_P = (k["rho_max_P"] * P_g / (k["K_P"] + P_g)
                 * np.maximum(k["Q_max"] - Qp, 0) / (k["Q_max"] - k["Q_min"]))
        mu_b = (k["mu_max_b"] * G_g / (k["K_G"] + G_g) * O_g / (k["K_O"] + O_g)
                * N_g / (k["K_Nb"] + N_g) * P_g / (k["K_Pb"] + P_g))
        g_a, g_b = mu_a * Xa_g, mu_b * Xb_g
        r_SN = k["k_s"] * (k["q_max_N"] * (Xa_g + Xb_g) * N_g / (k["K_s"] + N_g) - SN * M["N"])
        r_SP = k["k_s"] * (k["q_max_P"] * (Xa_g + Xb_g) * P_g / (k["K_s"] + P_g) - SP * M["P"])
        R = np.vstack([-(k["i_N_a"] * g_a + k["i_N_b"] * g_b + r_SN) / M["N"],
                       -(rho_P * Xa_g + k["i_P_b"] * g_b + r_SP) / M["P"],
                       -(g_b / k["Y_G"]) / M["G"],
                       (k["Y_O2_a"] * g_a - g_b * (1 - k["Y_G"]) / k["Y_G"] * 1.07) / M["O2"]])
        # diffusion: no flux at bottom, Dirichlet-like coupling to liquid at top (half cell)
        flux = np.zeros((4, n + 1))                           # upward flux at n+1 faces
        flux[:, 1:-1] = -D[:, None] * np.diff(c, axis=1) / dy
        flux[:, -1] = -D * (cl - c[:, -1]) / (dy / 2)         # top face, gel -> liquid
        dc = -np.diff(flux, axis=1) / dy + R
        dcl = flux[:, -1] * A / VL
        dcl[3] += k["kLa"] * (k["O_sat"] / M["O2"] - cl[3])
        db = np.vstack([(g_a - k["b_a"] * Xa_g) / M["X"], (g_b - k["b_b"] * Xb_g) / M["X"],
                        rho_P * Xa_g / M["P"] - k["b_a"] * PA, r_SN / M["N"], r_SP / M["P"]])
        return np.concatenate([dc.ravel(), db.ravel(), dcl])

    days = np.array([0, 1, 3, 5, 7])
    sol = solve_ivp(rhs, (0, 7 * 86400), y0, method="BDF", t_eval=days * 86400,
                    max_step=600, rtol=1e-5, atol=1e-10)
    cl = sol.y[9 * n:]
    print(f"f_gel = {a.f_gel}, gel thickness = {tb*1e3:g} mm, CV/BS algae fraction = {a.f_algae}")
    print(" day   NH4+ removal %   PO4 3- removal %   O2 liquid mg/L")
    for i, d in enumerate(days):
        print(f"{d:4d}   {100*(1-cl[0,i]/c0[0]):14.1f}   {100*(1-cl[1,i]/c0[1]):16.1f}"
              f"   {cl[3,i]*M['O2']:14.2f}")



# ============================================================================
# PART 10 - Self-tests (values checked against the paper)
# ============================================================================
def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def cmd_test(argv=None):
    checks = []

    def check(name, ok):
        checks.append(ok)
        print(("PASS " if ok else "FAIL ") + name)

    check("removal efficiency 86.7 %", _close(removal_efficiency(67.1, 67.1 * (1 - 0.867)), 86.7))
    check("NH4+ absorption rate 18.7 (Fig. 3c, day 1)",
          _close(absorption_rate(67.1, 67.1 * 0.44, 0.05, 1, 0.1), 18.7, 0.2))
    check("PO4 3- absorption rate 8.7 (Fig. 3d, day 1)",
          _close(absorption_rate(29.9, 29.9 * 0.42, 0.05, 1, 0.1), 8.7, 0.1))
    check("absorption rate is NaN on day 0",
          bool(np.isnan(absorption_rate(67.1, 67.1, 0.05, 0, 0.1))))
    check("viability = 100 % for equal growth", _close(cell_viability(0.8, 1.2, 0.8, 1.2, .05, .05), 100))
    check("degradation", _close(degradation(0.200, 0.194), 3.0))
    check("chlorophyll a", _close(chlorophyll_a(0.01, 0.01, 1.01, 0.01, 2.0), 5.81))
    check("total K", _close(total_k(0.5, 0.015), 0.5 * 120 / 0.015))
    check("total P", _close(total_p(1, 3, 1, 20, 2, 0, 1), 300))
    check("arcsine clips > 100 %", _close(arcsine_transform(150), np.pi / 2))
    check("decision matrix prefers high viability / low degradation",
          int(np.argmax(decision_matrix([219, 110, 52], [3.0, 5.3, 4.1]))) == 0)
    df = pd.DataFrame({"g": np.repeat(list("ABCD"), 3),
                       "v": [86, 87, 86.5, 81, 82, 80, 54, 55, 54.5, 50, 49, 50.5]})
    res = compare_groups(df, "v", "g", method="kruskal")
    check("Kruskal-Wallis H = 10.38, DF = 3 (Section 3.3)",
          _close(res["statistic"], 10.38, 0.01) and res["df"] == 3)
    check("compact letters",
          compact_letters(["A", "B", "C"], {("A", "B"): .9, ("A", "C"): .001, ("B", "C"): .001})
          == {"A": "a", "B": "a", "C": "b"})

    p = replace(Params(), b_a=0.0, b_b=0.0)
    r = Reactor.wastewater("immobilized", 0.1, 0.5)
    o = simulate(p, r, t_end=3.0)
    n_tot = o["N"] * r.V_L + (p.i_N_a * o["Xa"] + p.i_N_b * o["Xb"]) * 1000 + o["SN"]
    p_tot = o["P"] * r.V_L + (o["Qp"] * o["Xa"] + p.i_P_b * o["Xb"]) * 1000 + o["SP"]
    check("bioreactor N mass balance", bool(np.allclose(n_tot, n_tot[0], rtol=1e-3)))
    check("bioreactor P mass balance", bool(np.allclose(p_tot, p_tot[0], rtol=1e-3)))
    o = simulate(Params(q_max_N=0, q_max_P=0), Reactor.bg11_pbr(), t_end=7.0)
    check("C. vulgaris grows in BG-11", o["Xa_g_l"][-1] > 2 * o["Xa_g_l"][0])

    print(f"\n{sum(checks)}/{len(checks)} tests passed")
    if not all(checks):
        sys.exit(1)


def cmd_example_data(argv=None):
    ap = argparse.ArgumentParser(description="Write the simulated example data.")
    ap.add_argument("--out", default=DEFAULT_DATA, type=Path)
    make_example_data(ap.parse_args(argv).out)


def cmd_all(argv=None):
    ensure_data(DEFAULT_DATA)
    for title, fn in [("STATISTICS", cmd_analysis), ("BIOREACTOR MODEL", cmd_bioreactor),
                      ("COMSOL EQUATIONS IN 1D", cmd_comsol1d), ("SELF-TESTS", cmd_test)]:
        print(f"\n{'#' * 70}\n# {title}\n{'#' * 70}")
        fn([])


COMMANDS = {
    "all": cmd_all,
    "analysis": cmd_analysis,
    "bioreactor": cmd_bioreactor,
    "comsol1d": cmd_comsol1d,
    "test": cmd_test,
    "example-data": cmd_example_data,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands:", ", ".join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 2)
    COMMANDS[sys.argv[1]](sys.argv[2:])
