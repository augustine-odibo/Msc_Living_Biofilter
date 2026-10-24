"""
Simulations and figures for the microalgae core model.
Run after microalgae_cobrapy_model.py is in the same folder:
    python microalgae_simulations.py
Writes figures (PNG) and results (CSV) to ./figures
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from cobra.flux_analysis import production_envelope, flux_variability_analysis
from cobra.io import write_sbml_model

from microalgae_cobrapy_model import build_model, set_mode

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

model = build_model()
write_sbml_model(model, os.path.join(OUT, "microalgae_core.xml"))

# ------------------------------------------------------------------ 1. modes
modes = ["autotrophic", "mixotrophic", "heterotrophic"]
rows, fluxes = [], {}
for mode in modes:
    with model:
        set_mode(model, mode)
        sol = model.optimize()
        fluxes[mode] = sol.fluxes.copy()
        rows.append({
            "mode": mode,
            "growth_1_per_h": sol.objective_value,
            "light_uptake": -sol.fluxes["EX_photon"],
            "acetate_uptake": -sol.fluxes["EX_ac"],
            "O2_exchange": sol.fluxes["EX_o2"],
            "CO2_exchange": sol.fluxes["EX_co2"],
            "RuBisCO_flux": sol.fluxes["CBB"] * 3,
            "respiration_flux": sol.fluxes["RESP"],
        })
modes_df = pd.DataFrame(rows)
modes_df.to_csv(os.path.join(OUT, "growth_modes.csv"), index=False)
pd.DataFrame(fluxes).to_csv(os.path.join(OUT, "flux_distributions.csv"))
print(modes_df.round(3).to_string(index=False))

labels = ["Autotrophic\n(light + CO2)", "Mixotrophic\n(light + acetate)", "Heterotrophic\n(dark + acetate)"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
bars = ax.bar(labels, modes_df["growth_1_per_h"], color=BLUE, width=0.55, edgecolor="white", linewidth=2)
for b, v in zip(bars, modes_df["growth_1_per_h"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", color=INK, fontsize=10)
ax.set_ylabel("Growth rate (1/h)")
ax.set_title("Mixotrophy gives the fastest growth", loc="left", fontsize=11, color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[1]
x = range(3); w = 0.36
b1 = ax.bar([i - w / 2 for i in x], modes_df["O2_exchange"], w, color=AQUA, label="O2", edgecolor="white", linewidth=2)
b2 = ax.bar([i + w / 2 for i in x], modes_df["CO2_exchange"], w, color=ORANGE, label="CO2", edgecolor="white", linewidth=2)
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("Exchange flux (mmol/gDW/h)\n< 0 uptake, > 0 release")
ax.set_title("Gas exchange flips between light and dark", loc="left", fontsize=11, color=INK, fontweight="bold")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_growth_modes.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 2. light curve
lights = list(range(0, 101, 5))
curve = []
for L in lights:
    row = {"light": L}
    for mode in ["autotrophic", "mixotrophic"]:
        with model:
            set_mode(model, mode, light=L)
            g = model.slim_optimize(error_value=0.0)
            row[mode] = max(g, 0.0)
    curve.append(row)
curve_df = pd.DataFrame(curve)
curve_df.to_csv(os.path.join(OUT, "light_response.csv"), index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(curve_df["light"], curve_df["autotrophic"], color=BLUE, linewidth=2, marker="o", markersize=4)
ax.plot(curve_df["light"], curve_df["mixotrophic"], color=ORANGE, linewidth=2, marker="o", markersize=4)
ax.text(101, curve_df["autotrophic"].iloc[-1], "  Autotrophic", color=INK, va="center")
ax.text(101, curve_df["mixotrophic"].iloc[-1], "  Mixotrophic\n  (10 acetate)", color=INK, va="center")
ax.set_xlim(0, 125)
ax.set_xlabel("Light uptake (mmol photons/gDW/h)")
ax.set_ylabel("Growth rate (1/h)")
ax.set_title("Growth rises with light and saturates; acetate adds about 0.26 1/h", loc="left", fontsize=11, color=INK, fontweight="bold")
ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_light_response.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 3. storage envelope
env_rows = []
with model:
    set_mode(model, "autotrophic", light=60)
    gmax = model.slim_optimize()
    for frac in [i / 10 for i in range(11)]:
        g = gmax * frac
        with model:
            model.reactions.BIOMASS.bounds = (g, g)
            model.objective = "DM_lipid"
            lip = model.slim_optimize(error_value=0.0)
            model.objective = "DM_starch"
            sta = model.slim_optimize(error_value=0.0)
        env_rows.append({"growth": g, "lipid_C": lip * 16, "starch_C": sta * 6})
env_df = pd.DataFrame(env_rows)
env_df.to_csv(os.path.join(OUT, "storage_envelope.csv"), index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(env_df["growth"], env_df["starch_C"], color=BLUE, linewidth=2, marker="o", markersize=4)
ax.plot(env_df["growth"], env_df["lipid_C"], color=ORANGE, linewidth=2, marker="o", markersize=4)
ax.text(env_df["growth"].iloc[4] + 0.004, env_df["starch_C"].iloc[4] + 0.25, "Starch (maximum)", color=INK)
ax.text(env_df["growth"].iloc[4] - 0.004, env_df["lipid_C"].iloc[4] - 0.55, "Lipid (maximum)", color=INK, ha="right")
ax.set_xlabel("Growth rate (1/h)")
ax.set_ylabel("Storage carbon flux (mmol C/gDW/h)")
ax.set_title("Slower growth frees carbon for starch or lipid", loc="left", fontsize=11, color=INK, fontweight="bold")
ax.set_ylim(bottom=0)
ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_storage_envelope.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 4. FVA
with model:
    set_mode(model, "autotrophic", light=60)
    fva = flux_variability_analysis(model, fraction_of_optimum=0.95)
fva.to_csv(os.path.join(OUT, "fva_autotrophic.csv"))

# ------------------------------------------------------------------ 5. flux map
NODES = {
    "Light": (0.6, 8.6), "CO2": (3.6, 8.6),
    "ATP + NADPH": (0.6, 6.6), "G3P\n(Calvin cycle)": (3.6, 6.6), "Starch": (6.6, 6.6),
    "O2 out": (0.6, 4.6), "Pyruvate": (3.6, 4.6), "Biomass": (6.6, 4.6),
    "TCA +\nrespiration": (0.6, 2.6), "Acetyl-CoA": (3.6, 2.6), "Acetate": (6.6, 2.6),
    "Lipid": (3.6, 0.6),
}
# (from, to, net flux function); a negative value reverses the arrow
EDGES = [
    ("Light", "ATP + NADPH", lambda f: 8 * f["LEF"] + 2 * f["CEF"]),
    ("ATP + NADPH", "O2 out", lambda f: f["LEF"]),
    ("CO2", "G3P\n(Calvin cycle)", lambda f: 3 * f["CBB"]),
    ("G3P\n(Calvin cycle)", "Starch", lambda f: f["STARCH_SYN"]),
    ("G3P\n(Calvin cycle)", "Pyruvate", lambda f: f["GLYC"] - f["GNG"]),
    ("G3P\n(Calvin cycle)", "Biomass", lambda f: 4 * f["BIOMASS"]),
    ("Pyruvate", "Biomass", lambda f: f["AAS"]),
    ("Pyruvate", "Acetyl-CoA", lambda f: f["PDH"] - f["PEPCK"]),
    ("Acetate", "Acetyl-CoA", lambda f: f["ACS"]),
    ("Acetyl-CoA", "TCA +\nrespiration", lambda f: f["TCA"]),
    ("Acetyl-CoA", "Lipid", lambda f: f["FAS"]),
]


def draw_map(ax, fl, title):
    ax.set_xlim(-0.4, 7.6); ax.set_ylim(-0.1, 9.3); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    vals = [fn(fl) for _, _, fn in EDGES]
    vmax = max(abs(v) for v in vals) or 1
    for (a, b, _), v in zip(EDGES, vals):
        on = abs(v) > 1e-6
        (x1, y1), (x2, y2) = NODES[a], NODES[b]
        if v < -1e-6:
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
            shrinkA=30, shrinkB=30, linewidth=1.2 + 5 * abs(v) / vmax if on else 1,
            color=BLUE if on else "#c9c8c3", linestyle="-" if on else (0, (3, 3)), zorder=1))
        if on:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = (0.42, 0) if abs(x2 - x1) < 0.1 else (0, 0.28)
            ax.text(mx + dx, my + dy, f"{abs(v):.2f}", fontsize=8.5, color=INK2, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=3)
    for name, (x, y) in NODES.items():
        ax.add_patch(FancyBboxPatch((x - 0.7, y - 0.38), 1.4, 0.76,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=SURF, edgecolor=INK2, linewidth=1, zorder=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=8.5, color=INK, zorder=3)


fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
draw_map(axes[0], fluxes["autotrophic"], "Autotrophic: light drives CO2 fixation")
draw_map(axes[1], fluxes["heterotrophic"], "Heterotrophic: acetate is the only carbon source")
fig.text(0.01, 0.012, "Numbers and arrow widths = net flux in mmol/gDW/h (light in mmol photons/gDW/h). Grey dashed = inactive.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(os.path.join(OUT, "fig1_flux_maps.png"), dpi=200)
plt.close(fig)

print("Saved figures and CSV files to", OUT)
