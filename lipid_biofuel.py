#!/usr/bin/env python3
"""
lipid_biofuel.py - tools for screening microalgae for biofuel: culture medium,
FACS/BODIPY gating, rDNA primers, lipid-droplet image analysis, Bligh & Dyer
lipid content, FAME profile -> biodiesel properties (EN 14214 / ASTM D6751),
GC run time, and statistics.

Based on the methods of: Pereira H, Barreira L, Custodio L, Alrokayan S,
Mouffouk F, Varela J, Abu-Salah KM, Ben-Hamadou R (2013). Isolation and fatty
acid profile of selected microalgae strains from the Red Sea for biofuel
production. Energies 6:2773-2783. doi:10.3390/en6062773 (CC-BY 3.0).
The published Table 2 FAME profiles and Figure 3 lipid contents are built in as
example data, so every command runs out of the box.

COMMANDS
  medium     ALGAL medium: stock recipe (g/L) and final concentrations
  facs       gate flow-cytometry events (Chl gate, BODIPY populations P1-P3)
  primers    GC %, Tm (nearest-neighbour), primer-dimer check, in-silico PCR
  droplets   count/measure BODIPY lipid droplets in a fluorescence image
  lipid      gravimetric (Bligh & Dyer) lipid content from tube weights
  fame       FAME profile -> SFA/MUFA/PUFA, iodine value, cetane number,
             CFPP, oxidation stability, viscosity, density, EN 14214 check
  gc         GC oven programme run time and temperature profile
  stats      one-way ANOVA + Tukey from raw data or from mean/SD/n
  methods    write the improved-methods document (Markdown)
  example    write example input files
  test       self-tests
  all        run everything on the example data

    python lipid_biofuel.py fame                         # paper's Table 2
    python lipid_biofuel.py fame --input my_fame.csv     # your GC-MS results
    python lipid_biofuel.py lipid --input my_weights.csv
    python lipid_biofuel.py facs --input events.csv      # or .fcs (pip install fcsparser)
    python lipid_biofuel.py <command> --help

Requires: numpy, pandas, scipy, matplotlib.   License: MIT.
"""
from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, ndimage

OUT = Path.cwd() / "results" / "lipid_biofuel"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    return OUT / name


# =============================================================================
# 1. Culture medium (Section 3.1: modified ALGAL medium, 1:1000 dilution)
# =============================================================================
ALGAL = [  # component, stock mM (as published), molar mass g/mol
    ("ZnCl2", 1, 136.29),
    ("ZnSO4.H2O", 1, 179.47),
    ("MnCl2.4H2O", 1, 197.91),
    ("Na2MoO4.2H2O", 0.1, 241.95),
    ("CoCl2.6H2O", 0.1, 237.93),
    ("CuSO4.5H2O", 0.1, 249.69),
    ("Na2EDTA.2H2O", 20, 372.24),
    ("MgSO4.7H2O", 2, 246.47),
    ("FeCl3.6H2O", 20, 270.30),
    ("NaNO3", 2000, 84.99),
    ("KH2PO4", 100, 136.09),
]


def medium_table(dilution=1000, stock_volume_ml=100):
    rows = []
    for comp, mM, mw in ALGAL:
        rows.append(dict(component=comp, stock_mM=mM,
                         g_per_L_stock=mM * mw / 1000,
                         g_for_stock_volume=mM * mw / 1000 * stock_volume_ml / 1000,
                         final_uM=mM * 1000 / dilution))
    return pd.DataFrame(rows)


def cmd_medium(argv=None):
    ap = argparse.ArgumentParser(description="ALGAL medium (Pereira et al. 2013, Section 3.1).")
    ap.add_argument("--dilution", type=float, default=1000)
    ap.add_argument("--stock-ml", type=float, default=100, help="volume of concentrate to make")
    a = ap.parse_args(argv)
    t = medium_table(a.dilution, a.stock_ml)
    print(t.round(4).to_string(index=False))
    n = t.loc[t.component == "NaNO3", "final_uM"].iloc[0]
    p = t.loc[t.component == "KH2PO4", "final_uM"].iloc[0]
    print(f"\nFinal N:P molar ratio = {n / p:.0f}:1 (Redfield 16:1); "
          f"nitrate {n:.0f} uM, phosphate {p:.0f} uM")
    print("""Notes:
  * Zinc is listed twice (ZnCl2 and ZnSO4.H2O): total Zn = 2 uM. Check whether that is intended.
  * The FeCl3 hydrate is not stated; masses above use FeCl3.6H2O (anhydrous: 162.20 g/mol).
  * EDTA:Fe is 1:1, which keeps iron in solution; add Fe-EDTA as a separate stock from the
    other metals and from phosphate to avoid precipitation.
  * No vitamins (B12, biotin, thiamine) are listed; many marine chlorophytes need B12.""")
    print(f"\nSaved {_save(t, 'algal_medium.csv')}")


# =============================================================================
# 2. FACS: gating of BODIPY-stained cells (Sections 2.2, 3.2; Figure 1)
# =============================================================================
DEFAULT_GATES = [  # name, x channel, y channel, xmin, xmax, ymin, ymax (log-scale boxes)
    ("Chl", "FL3", "SSC", 1e2, 3e5, 0, 2.6e5),
    ("P1", "FL1", "FL3", 2e3, 1e4, 1e4, 6e4),
    ("P2", "FL1", "FL3", 1e3, 3.5e3, 1e3, 6e3),
    ("P3", "FL1", "FL3", 80, 400, 1e3, 6e3),
]


def simulate_events(n=20000, seed=1):
    """Events resembling Fig. 1: debris/bacteria (no chlorophyll) + three algal populations."""
    r = np.random.default_rng(seed)
    def pop(k, fl1, fl3, ssc):
        return pd.DataFrame(dict(FL1=10 ** r.normal(np.log10(fl1), 0.15, k),
                                 FL3=10 ** r.normal(np.log10(fl3), 0.12, k),
                                 SSC=np.abs(r.normal(ssc, ssc * 0.4, k)),
                                 FSC=np.abs(r.normal(ssc * 1.2, ssc * 0.4, k))))
    parts = [pop(int(n * .30), 60, 25, 15000).assign(truth="debris/bacteria"),
             pop(int(n * .40), 4500, 2.4e4, 60000).assign(truth="P1"),
             pop(int(n * .15), 1900, 2.6e3, 30000).assign(truth="P2"),
             pop(int(n * .15), 180, 2.6e3, 20000).assign(truth="P3")]
    return pd.concat(parts, ignore_index=True)


def apply_gates(ev, gates=DEFAULT_GATES, parent="Chl"):
    rows, masks = [], {}
    for name, xc, yc, x0, x1, y0, y1 in gates:
        m = ev[xc].between(x0, x1) & ev[yc].between(y0, y1)
        if name != parent and parent in masks:
            m &= masks[parent]
        masks[name] = m
    total = len(ev)
    for name, m in masks.items():
        k = int(m.sum())
        rows.append(dict(gate=name, events=k, pct_of_total=100 * k / total,
                         pct_of_Chl=100 * k / max(int(masks[parent].sum()), 1),
                         median_FL1_BODIPY=float(ev.loc[m, "FL1"].median()) if k else np.nan,
                         median_FL3_Chl=float(ev.loc[m, "FL3"].median()) if k else np.nan))
    t = pd.DataFrame(rows)
    # BODIPY per unit chlorophyll: a size-independent lipid index for comparing populations
    t["lipid_index_FL1_per_FL3"] = t.median_FL1_BODIPY / t.median_FL3_Chl
    return t, masks


def read_events(path):
    path = Path(path)
    if path.suffix.lower() == ".fcs":
        try:
            import fcsparser
        except ImportError:
            sys.exit("Reading .fcs files needs: pip install fcsparser  (or export events to CSV)")
        _, ev = fcsparser.parse(str(path), reformat_meta=True)
        return ev
    return pd.read_csv(path)


def cmd_facs(argv=None):
    ap = argparse.ArgumentParser(description="Gate FACS events. CSV/FCS with columns FL1, FL3, SSC "
                                 "(rename yours with --map FL1=FITC-A,FL3=PerCP-A,SSC=SSC-A)")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--map", default="", help="our=theirs pairs, comma-separated")
    a = ap.parse_args(argv)
    ev = read_events(a.input) if a.input else simulate_events()
    if a.map:
        ev = ev.rename(columns={v: k for k, v in (p.split("=") for p in a.map.split(","))})
    t, masks = apply_gates(ev)
    print(f"{len(ev):,} events" + ("" if a.input else " (simulated, resembling Fig. 1)"))
    print(t.round(3).to_string(index=False))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].scatter(ev.FL3, ev.SSC, s=1, c="k", alpha=0.3)
    ax[0].set(xscale="log", xlabel="FL3 (chlorophyll)", ylabel="SSC", title="(A) Chl gate")
    chl = ev[masks["Chl"]]
    ax[1].hexbin(chl.FL1, chl.FL3, xscale="log", yscale="log", bins="log", gridsize=70, cmap="jet")
    ax[1].set(xlabel="FL1 (BODIPY 505/515)", ylabel="FL3 (chlorophyll)", title="(B) populations")
    for name, xc, yc, x0, x1, y0, y1 in DEFAULT_GATES:
        tgt = ax[0] if name == "Chl" else ax[1]
        tgt.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k" if name != "Chl"
                                    else "r", lw=1.5))
        if name == "Chl":
            tgt.text(x0 * 1.3, ev.SSC.max() * 0.95, name, color="r", fontsize=10)
        else:
            tgt.text(x1 * 1.05, y1, name, color="k", fontsize=10, va="top")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "facs_gating.png", dpi=250)
    plt.close(fig)
    print(f"\nSaved {_save(t, 'facs_gates.csv')} and {OUT / 'facs_gating.png'}")
    print("Tip: set gates on an unstained control (no BODIPY) of the same culture, and "
          "compensate chlorophyll spill-over into FL1; see `methods`.")


# =============================================================================
# 3. Primers (Section 3.3, Table 3)
# =============================================================================
PRIMERS = {
    "18SUnivFor": "ACCTGGTTGATCCTGCCAGT",
    "18SUnivRev": "TCAGCCTTGCGACCATAC",
    "5.8SFor": "AAGAACGCAGCGAAATGC",
    "28SRev": "GACTCCTTGGTCCGTGTTTC",
}
PAIRS = [("18SUnivFor", "18SUnivRev"), ("5.8SFor", "28SRev")]

# SantaLucia (1998) unified nearest-neighbour parameters: dH kcal/mol, dS cal/(K mol)
NN = {"AA": (-7.9, -22.2), "TT": (-7.9, -22.2), "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
      "CA": (-8.5, -22.7), "TG": (-8.5, -22.7), "GT": (-8.4, -22.4), "AC": (-8.4, -22.4),
      "CT": (-7.8, -21.0), "AG": (-7.8, -21.0), "GA": (-8.2, -22.2), "TC": (-8.2, -22.2),
      "CG": (-10.6, -27.2), "GC": (-9.8, -24.4), "GG": (-8.0, -19.9), "CC": (-8.0, -19.9)}
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.upper().translate(COMP)[::-1]


def gc_pct(s):
    s = s.upper()
    return 100 * (s.count("G") + s.count("C")) / len(s)


def tm_nn(seq, na_mM=50.0, primer_nM=250.0):
    """Nearest-neighbour Tm (C), SantaLucia 1998, Na+ correction; non-self-complementary."""
    s = seq.upper()
    dh = sum(NN[s[i:i + 2]][0] for i in range(len(s) - 1))
    ds = sum(NN[s[i:i + 2]][1] for i in range(len(s) - 1))
    for end in (s[0], s[-1]):                         # terminal initiation
        dh += 0.1 if end in "GC" else 2.3
        ds += -2.8 if end in "GC" else 4.1
    ds += 0.368 * (len(s) - 1) * math.log(na_mM / 1000)
    return dh * 1000 / (ds + 1.987 * math.log(primer_nM * 1e-9 / 4)) - 273.15


def three_prime_dimer(p1, p2, min_len=4):
    """Longest run at the 3' end of p1 that can base-pair with p2 (primer-dimer risk)."""
    best = 0
    for k in range(min_len, len(p1) + 1):
        if revcomp(p1[-k:]) in p2.upper():
            best = k
    return best


def find_site(template, primer, max_mm=2):
    """Best binding position of a primer on a template (either strand), allowing mismatches."""
    t = template.upper()
    best = None
    for strand, p in (("+", primer.upper()), ("-", revcomp(primer))):
        for i in range(len(t) - len(p) + 1):
            mm = sum(1 for a, b in zip(t[i:i + len(p)], p) if a != b)
            if mm <= max_mm and (best is None or mm < best[2]):
                best = (i, strand, mm)
    return best


def read_fasta(path):
    seqs, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line.strip()
    return seqs


def cmd_primers(argv=None):
    ap = argparse.ArgumentParser(description="Primer properties and optional in-silico PCR.")
    ap.add_argument("--template", type=Path, help="FASTA of rDNA sequences for in-silico PCR")
    ap.add_argument("--na", type=float, default=50, help="mM monovalent salt")
    a = ap.parse_args(argv)
    rows = [dict(primer=n, sequence=s, length=len(s), GC_pct=gc_pct(s), Tm_NN_C=tm_nn(s, a.na),
                 self_3prime_dimer=three_prime_dimer(s, s)) for n, s in PRIMERS.items()]
    t = pd.DataFrame(rows)
    print(t.round(1).to_string(index=False))
    print()
    for f, r in PAIRS:
        tf, tr = tm_nn(PRIMERS[f], a.na), tm_nn(PRIMERS[r], a.na)
        print(f"{f} / {r}: dTm = {abs(tf - tr):.1f} C; suggested annealing ~{min(tf, tr) - 3:.0f} C "
              f"(gradient PCR to confirm); cross 3'-dimer = "
              f"{max(three_prime_dimer(PRIMERS[f], PRIMERS[r]), three_prime_dimer(PRIMERS[r], PRIMERS[f]))} nt")
    if a.template:
        res = []
        for name, seq in read_fasta(a.template).items():
            for f, r in PAIRS:
                sf, sr = find_site(seq, PRIMERS[f]), find_site(seq, PRIMERS[r])
                size = (sr[0] + len(PRIMERS[r]) - sf[0]) if sf and sr and sf[1] == "+" and sr[1] == "-" else None
                res.append(dict(template=name, pair=f"{f}/{r}", fwd_mismatches=sf[2] if sf else None,
                                rev_mismatches=sr[2] if sr else None, amplicon_bp=size))
        print("\nIn-silico PCR (<= 2 mismatches):")
        print(pd.DataFrame(res).to_string(index=False))
    print("\nTm is for 50 mM Na+, 250 nM primer; Mg2+ in PCR buffer raises it by a few C.")
    print(f"Saved {_save(t, 'primers.csv')}")


# =============================================================================
# 4. Lipid droplets in BODIPY images (Section 2.4, Figure 2)
# =============================================================================
def otsu(x):
    hist, edges = np.histogram(x.ravel(), bins=256)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * edges[:-1])
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return edges[np.nanargmax(between)]


def detect_droplets(img, um_per_px=0.065, min_px=4, sigma=1.0, min_sep_px=5):
    """Gaussian smooth -> Otsu mask -> one marker per local maximum -> watershed split.

    Splitting at local maxima keeps touching droplets apart (a plain threshold merges them).
    """
    g = ndimage.gaussian_filter(img.astype(float), sigma)
    thr = otsu(g)
    mask = g > thr
    peaks = (g == ndimage.maximum_filter(g, size=2 * min_sep_px + 1)) & mask
    markers, n = ndimage.label(peaks)
    if n == 0:
        return pd.DataFrame(columns=["droplet", "area_um2", "diameter_um", "mean_intensity"])
    inv = ((g.max() - g) / (np.ptp(g) + 1e-12) * 65000).astype(np.uint16)
    markers = markers.astype(np.int32)
    markers[~mask] = -1                                    # background marker
    lab = ndimage.watershed_ift(inv, markers)
    lab[lab < 0] = 0
    idx = np.arange(1, n + 1)
    area = ndimage.sum(lab > 0, lab, idx)
    inten = ndimage.mean(img, lab, idx)
    keep = area >= min_px
    a_um2 = area[keep] * um_per_px ** 2
    return pd.DataFrame(dict(droplet=np.arange(1, keep.sum() + 1), area_um2=a_um2,
                             diameter_um=2 * np.sqrt(a_um2 / np.pi), mean_intensity=inten[keep],
                             _label=idx[keep]))


def synthetic_bodipy_image(seed=3, n_cells=12, size=512):
    """Test image: cells with 2-5 droplets each; droplets may touch but do not overlap."""
    r = np.random.default_rng(seed)
    img = r.normal(10, 3, (size, size))
    yy, xx = np.mgrid[:size, :size]
    placed = []
    for _ in range(n_cells):
        cx, cy = r.uniform(40, size - 40, 2)
        for _ in range(r.integers(2, 6)):
            for _try in range(50):
                x, y = cx + r.normal(0, 12), cy + r.normal(0, 12)
                rad = r.uniform(3, 6)
                if all(math.hypot(x - px, y - py) > 2 * (rad + pr) for px, py, pr in placed):
                    placed.append((x, y, rad))
                    img += 200 * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * rad ** 2)))
                    break
    return np.clip(img, 0, None), len(placed)


def cmd_droplets(argv=None):
    ap = argparse.ArgumentParser(description="Count BODIPY lipid droplets in a fluorescence image "
                                 "(PNG/TIFF; the green channel is used for RGB).")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--um-per-px", type=float, default=0.065, help="100x objective: ~0.065")
    ap.add_argument("--cells", type=int, help="number of cells in the image (from DIC)")
    a = ap.parse_args(argv)
    if a.image:
        import matplotlib.image as mpimg
        img = mpimg.imread(a.image).astype(float)
        img = img[..., 1] if img.ndim == 3 else img
        truth = None
    else:
        img, truth = synthetic_bodipy_image()
        a.cells = a.cells or 12
    d = detect_droplets(img, a.um_per_px)
    print(f"Droplets detected: {len(d)}" + (f" (synthetic image, {truth} drawn)" if truth else ""))
    if len(d):
        print(f"  median diameter {d.diameter_um.median():.2f} um, total area {d.area_um2.sum():.1f} um2")
        if a.cells:
            print(f"  droplets per cell {len(d) / a.cells:.2f}; droplet area per cell "
                  f"{d.area_um2.sum() / a.cells:.2f} um2")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="Greens_r")
    g = ndimage.gaussian_filter(img.astype(float), 1.0)
    pk = (g == ndimage.maximum_filter(g, size=11)) & (g > otsu(g))
    yy, xx = np.nonzero(pk)
    ax.plot(xx, yy, "r+", ms=6)
    ax.set_axis_off()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "droplets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_save(d.drop(columns='_label', errors='ignore'), 'droplets.csv')} and {OUT / 'droplets.png'}")


# =============================================================================
# 5. Gravimetric lipid (modified Bligh & Dyer, Section 3.5; Figure 3)
# =============================================================================
PAPER_LIPID = pd.DataFrame(dict(  # Figure 3 / Section 2.5: mean, SD, n = 4
    strain=["Nannochloris sp. SBL1", "Picochlorum sp. SBL2", "Desmochloris sp. SBL3",
            "Nannochloris sp. SBL4"],
    mean=[19.69, 25.28, 23.16, 22.65], sd=[2.19, 2.38, 3.00, 2.21], n=[4, 4, 4, 4]))


def lipid_content(d):
    """Lipid % of dry weight from tube weights.

    lipid % DW = (residue - blank) x (V_organic_total / V_aliquot) / dry biomass x 100
    dry biomass = biomass_mg x (1 - moisture_pct/100)   (moisture_pct = 0 for freeze-dried)
    """
    d = d.copy()
    for c, v in (("moisture_pct", 0.0), ("blank_mg", 0.0)):
        if c not in d:
            d[c] = v
        d[c] = d[c].fillna(v)
    d["dry_biomass_mg"] = d.biomass_mg * (1 - d.moisture_pct / 100)
    d["residue_mg"] = d.tube_after_mg - d.tube_empty_mg - d.blank_mg
    d["lipid_mg"] = d.residue_mg * d.organic_total_ml / d.aliquot_ml
    d["lipid_pct_dw"] = 100 * d.lipid_mg / d.dry_biomass_mg
    return d


def cmd_lipid(argv=None):
    ap = argparse.ArgumentParser(description="Lipid %% DW. CSV columns: strain, replicate, biomass_mg, "
                                 "tube_empty_mg, tube_after_mg, organic_total_ml, aliquot_ml "
                                 "[, moisture_pct, blank_mg]")
    ap.add_argument("--input", type=Path, default=Path("lipid_example.csv"))
    a = ap.parse_args(argv)
    if not a.input.exists():
        write_examples(a.input.parent)
    d = lipid_content(pd.read_csv(a.input))
    s = d.groupby("strain", sort=False).lipid_pct_dw.agg(["mean", "std", "count"]).reset_index()
    print(s.round(2).to_string(index=False))
    res = anova_raw(d, "lipid_pct_dw", "strain")
    print_anova(res)
    _save(d, "lipid_processed.csv")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(len(s)), s["mean"], yerr=s["std"], color="#7F7F7F", capsize=3)
    for i, (m, sd, st) in enumerate(zip(s["mean"], s["std"], s.strain)):
        ax.text(i, m + sd + 0.5, res["letters"].get(st, ""), ha="center")
    ax.set_xticks(range(len(s)), s.strain, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Total lipid content (% DW)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lipid_content.png", dpi=250)
    plt.close(fig)
    print(f"Saved {OUT / 'lipid_processed.csv'} and {OUT / 'lipid_content.png'}")


# =============================================================================
# 6. FAME profile -> biodiesel quality (Section 2.5, Table 2)
# =============================================================================
# methyl-ester molar masses (g/mol), number of double bonds, carbon number
FAME = {
    "C14:0": (242.40, 0, 14), "C15:0": (256.42, 0, 15), "C16:0": (270.45, 0, 16),
    "C16:1": (268.43, 1, 16), "C16:2": (266.42, 2, 16), "C16:3": (264.40, 3, 16),
    "C16:4": (262.39, 4, 16), "C17:0": (284.48, 0, 17), "C18:0": (298.50, 0, 18),
    "C18:1": (296.49, 1, 18), "C18:2": (294.47, 2, 18), "C18:3": (292.46, 3, 18),
    "C18:4": (290.44, 4, 18), "C20:0": (326.56, 0, 20), "C20:1": (324.54, 1, 20),
    "C20:4": (318.49, 4, 20), "C20:5": (316.48, 5, 20), "C22:0": (354.61, 0, 22),
    "C22:6": (342.51, 6, 22), "C24:0": (382.66, 0, 24),
}

PAPER_FAME = pd.DataFrame({  # Table 2 of Pereira et al. (2013), % of total FAME (means)
    "fame": ["C15:0", "C16:0", "C18:0", "C16:1", "C18:1", "C16:2", "C16:3", "C18:2", "C18:3"],
    "Nannochloris sp. SBL1": [0, 32.48, 2.28, 12.19, 33.82, 0, 3.80, 15.44, 0],
    "Picochlorum sp. SBL2": [0, 26.16, 3.22, 12.05, 21.96, 7.18, 5.68, 23.76, 0],
    "Desmochloris sp. SBL3": [4.19, 33.63, 1.94, 0, 28.17, 0, 1.32, 28.55, 2.21],
    "Nannochloris sp. SBL4": [0, 28.52, 0, 17.10, 40.41, 0, 4.46, 9.50, 0],
})
PAPER_IODINE = {"Nannochloris sp. SBL1": 86, "Picochlorum sp. SBL2": 99,
                "Desmochloris sp. SBL3": 84, "Nannochloris sp. SBL4": 88}

LIMITS = [  # property, EN 14214 (min, max), ASTM D6751 (min, max)
    ("iodine_value", (None, 120), (None, None)),
    ("C18:3_pct", (None, 12), (None, None)),
    ("PUFA_ge4_pct", (None, 1), (None, None)),
    ("cetane_number", (51, None), (47, None)),
    ("oxidation_stability_h", (8, None), (3, None)),
    ("viscosity_mm2_s", (3.5, 5.0), (1.9, 6.0)),
    ("density_kg_m3", (860, 900), (None, None)),
]


def biodiesel_properties(profile):
    """profile: dict {'C16:0': %, ...} (wt % of total FAME). Returns properties.

    SV, IV           : sum(560 A/M), sum(254 D A/M)              (Kalayasiri et al. 1996)
    CN               : 46.3 + 5458/SV - 0.225 IV                  (Krisnangkura 1986)
    DU, LCSF, CFPP   : MUFA + 2 PUFA; 0.1 C16:0 + 0.5 C18:0 + C20:0 + 1.5 C22:0 + 2 C24:0;
                       CFPP = 3.1417 LCSF - 16.477                (Ramos et al. 2009)
    OS (Rancimat, h) : 117.9295 / (C18:2 + C18:3) + 2.5905        (Park et al. 2008)
    viscosity, density, HHV : Ramirez-Verduzco et al. (2012), mass-fraction mixing
    """
    unknown = [k for k, v in profile.items() if v and k not in FAME]
    if unknown:
        raise ValueError(f"Unknown FAME {unknown}; add them to FAME with molar mass and double bonds")
    A = {k: float(v) for k, v in profile.items() if v}
    tot = sum(A.values())
    sfa = sum(v for k, v in A.items() if FAME[k][1] == 0)
    mufa = sum(v for k, v in A.items() if FAME[k][1] == 1)
    pufa = sum(v for k, v in A.items() if FAME[k][1] >= 2)
    pufa4 = sum(v for k, v in A.items() if FAME[k][1] >= 4)
    sv = sum(560 * v / FAME[k][0] for k, v in A.items())
    iv = sum(254 * FAME[k][1] * v / FAME[k][0] for k, v in A.items())
    cn = 46.3 + 5458 / sv - 0.225 * iv
    lcsf = (0.1 * A.get("C16:0", 0) + 0.5 * A.get("C18:0", 0) + 1.0 * A.get("C20:0", 0)
            + 1.5 * A.get("C22:0", 0) + 2.0 * A.get("C24:0", 0))
    cfpp = 3.1417 * lcsf - 16.477
    x = A.get("C18:2", 0) + A.get("C18:3", 0)
    os_h = 117.9295 / x + 2.5905 if x > 0 else np.inf
    z = {k: v / tot for k, v in A.items()}
    ln_nu = sum(z[k] * (-12.503 + 2.496 * math.log(FAME[k][0]) - 0.178 * FAME[k][1]) for k in z)
    rho = sum(z[k] * (0.8463 + 4.9 / FAME[k][0] + 0.0118 * FAME[k][1]) for k in z)
    hhv = sum(z[k] * (46.19 - 1794 / FAME[k][0] - 0.21 * FAME[k][1]) for k in z)
    c16_c18 = sum(v for k, v in A.items() if FAME[k][2] in (16, 18))
    return dict(total_pct=tot, SFA=sfa, MUFA=mufa, PUFA=pufa, PUFA_ge4_pct=pufa4,
                **{"C18:3_pct": A.get("C18:3", 0)},
                C16_0_plus_C18_1=A.get("C16:0", 0) + A.get("C18:1", 0),
                C16_C18_share=c16_c18, DU=mufa + 2 * pufa,
                saponification_value=sv, iodine_value=iv, cetane_number=cn, LCSF=lcsf,
                CFPP_C=cfpp, oxidation_stability_h=os_h, viscosity_mm2_s=math.exp(ln_nu),
                density_kg_m3=rho * 1000, HHV_MJ_kg=hhv)


def check_limits(p):
    out = {}
    for prop, en, astm in LIMITS:
        v = p[prop]
        for std, (lo, hi) in (("EN14214", en), ("ASTM_D6751", astm)):
            if lo is None and hi is None:
                continue
            ok = (lo is None or v >= lo) and (hi is None or v <= hi)
            out[f"{std}:{prop}"] = "pass" if ok else "FAIL"
    return out


def cmd_fame(argv=None):
    ap = argparse.ArgumentParser(description="Biodiesel properties from FAME composition. CSV: column "
                                 "'fame' (e.g. C16:0) and one column per strain (wt %% of total FAME).")
    ap.add_argument("--input", type=Path, help="default: Table 2 of Pereira et al. 2013")
    a = ap.parse_args(argv)
    t = pd.read_csv(a.input) if a.input else PAPER_FAME
    rows, checks = [], []
    for strain in [c for c in t.columns if c != "fame"]:
        prof = dict(zip(t.fame, t[strain].fillna(0)))
        p = biodiesel_properties(prof)
        rows.append(dict(strain=strain, **p))
        checks.append(dict(strain=strain, **check_limits(p)))
    props, lim = pd.DataFrame(rows), pd.DataFrame(checks)
    pd.set_option("display.width", 200)
    print(props.set_index("strain").T.round(2).to_string())
    print("\nStandards check:")
    print(lim.set_index("strain").T.to_string())
    if a.input is None:
        print("\nReproducibility check against the paper (Table 2 and Section 2.5):")
        for _, r in props.iterrows():
            print(f"  {r.strain:24s} FAME sum {r.total_pct:6.2f} %   iodine value: paper "
                  f"{PAPER_IODINE[r.strain]}, recalculated {r.iodine_value:.0f}   "
                  f"C16:0 + C18:1 = {r.C16_0_plus_C18_1:.1f} %")
        diff = props.iodine_value.values - np.array([PAPER_IODINE[s] for s in props.strain])
        low = props.loc[props.C16_0_plus_C18_1 < 50, "strain"].tolist()
        hi_pufa = props.loc[props.PUFA > 25, "strain"].tolist()
        print(f"  -> Recalculated iodine values (standard equation, methyl-ester masses) differ "
              f"from the reported ones by {diff.min():+.0f} to {diff.max():+.0f};\n"
              f"     the paper does not state its equation. All are below the EN limit of 120 "
              f"either way.")
        if low:
            print(f"  -> C16:0 + C18:1 is below 50 % for {', '.join(low)} "
                  f"(the text says > 50 % for all isolates).")
        if hi_pufa:
            print(f"  -> PUFA is above 25 % in {', '.join(hi_pufa)}, so 'low unsaturation' does "
                  f"not hold for every strain; their EN 14214 oxidation stability is predicted to fail.")
    _save(props, "biodiesel_properties.csv")
    _save(lim, "standards_check.csv")
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(props))
    b = np.zeros(len(props))
    for col, c in (("SFA", "#7F7F7F"), ("MUFA", "#548235"), ("PUFA", "#C00000")):
        ax[0].bar(x, props[col], bottom=b, color=c, label=col)
        b += props[col].values
    ax[0].set_xticks(x, props.strain, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("% of total FAME")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax[1].scatter(props.iodine_value, props.cetane_number, color="#2E75B6")
    for _, r in props.iterrows():
        ax[1].annotate(r.strain.split()[0][:5] + " " + r.strain.split()[-1],
                       (r.iodine_value, r.cetane_number), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
    ax[1].axhline(51, ls="--", color="grey", lw=0.8)
    ax[1].axvline(120, ls="--", color="grey", lw=0.8)
    ax[1].set(xlabel="Iodine value (g I2/100 g; EN max 120)", ylabel="Cetane number (EN min 51)")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fame_biodiesel.png", dpi=250)
    plt.close(fig)
    print(f"\nSaved {OUT / 'biodiesel_properties.csv'}, standards_check.csv, fame_biodiesel.png")


# =============================================================================
# 7. GC oven programme (Section 3.5)
# =============================================================================
PAPER_GC = [(60, None, 1.0), (120, 30, 0), (250, 4, 0), (270, 20, 0), (300, 2.5, 0)]
# (target C, ramp C/min or None for initial, hold min)


def gc_programme(prog):
    t, temp, pts = 0.0, prog[0][0], [(0.0, prog[0][0])]
    t += prog[0][2]
    pts.append((t, temp))
    for target, rate, hold in prog[1:]:
        t += abs(target - temp) / rate
        temp = target
        pts.append((t, temp))
        if hold:
            t += hold
            pts.append((t, temp))
    return t, pts


def cmd_gc(argv=None):
    ap = argparse.ArgumentParser(description="GC oven programme run time (default: the paper's).")
    ap.add_argument("--final-hold", type=float, default=0.0, help="min at the final temperature")
    ap.add_argument("--cool-down", type=float, default=5.0, help="min to cool and re-equilibrate")
    a = ap.parse_args(argv)
    prog = [list(s) for s in PAPER_GC]
    prog[-1][2] = a.final_hold
    total, pts = gc_programme([tuple(s) for s in prog])
    for (t0, T0), (t1, T1) in zip(pts, pts[1:]):
        print(f"  {t0:6.2f} -> {t1:6.2f} min : {T0:5.0f} -> {T1:5.0f} C")
    print(f"Run time {total:.1f} min; with {a.cool_down:g} min cool-down "
          f"{total + a.cool_down:.1f} min -> {60 * 24 / (total + a.cool_down):.0f} injections per day")
    print("Note: the programme ends at 300 C with no final hold; a 3-5 min hold elutes "
          "heavier compounds and cleans the column between runs.")


# =============================================================================
# 8. Statistics (the paper reports mean +/- SD without tests)
# =============================================================================
def compact_letters(groups, pmat, alpha=0.05):
    sets = [set(groups)]
    for a, b in combinations(groups, 2):
        if pmat.get((a, b), pmat.get((b, a), 1)) >= alpha:
            continue
        new = []
        for s in sets:
            new += [s - {a}, s - {b}] if (a in s and b in s) else [s]
        sets = [s for i, s in enumerate(new)
                if not any((s < t) or (s == t and j < i) for j, t in enumerate(new) if j != i)]
    sets.sort(key=lambda s: min(groups.index(g) for g in s))
    out = {g: "" for g in groups}
    for i, s in enumerate(sets):
        for g in s:
            out[g] += "abcdefghij"[i]
    return out


def anova_summary(names, means, sds, ns, alpha=0.05):
    """One-way ANOVA + Tukey-Kramer from group means, SDs and sizes."""
    m, s, n = map(lambda v: np.asarray(v, float), (means, sds, ns))
    k, N = len(m), n.sum()
    grand = (n * m).sum() / N
    ssb = (n * (m - grand) ** 2).sum()
    ssw = ((n - 1) * s ** 2).sum()
    dfb, dfw = k - 1, N - k
    msw = ssw / dfw
    F = (ssb / dfb) / msw
    p = stats.f.sf(F, dfb, dfw)
    pairs, pm = [], {}
    for i, j in combinations(range(k), 2):
        se = math.sqrt(msw / 2 * (1 / n[i] + 1 / n[j]))
        q = abs(m[i] - m[j]) / se
        pq = stats.studentized_range.sf(q, k, dfw)
        pm[(names[i], names[j])] = pq
        pairs.append(dict(group1=names[i], group2=names[j], diff=m[i] - m[j], q=q, p_adj=pq))
    return dict(F=F, df=(dfb, dfw), p=p, pairwise=pd.DataFrame(pairs),
                letters=compact_letters(list(names), pm, alpha))


def anova_raw(d, value, group):
    g = d.groupby(group, sort=False)[value]
    return anova_summary(list(g.groups), g.mean().values, g.std().values, g.count().values)


def print_anova(res):
    sig = "significant" if res["p"] < 0.05 else "not significant"
    print(f"\nOne-way ANOVA: F({res['df'][0]:.0f}, {res['df'][1]:.0f}) = {res['F']:.2f}, "
          f"P = {res['p']:.5f} ({sig})")
    print(res["pairwise"].round(4).to_string(index=False))
    print("Letters (Tukey-Kramer):", ", ".join(f"{k}: {v}" for k, v in res["letters"].items()))


def cmd_stats(argv=None):
    ap = argparse.ArgumentParser(description="ANOVA + Tukey from mean/SD/n (CSV: strain, mean, sd, n). "
                                 "Default: the paper's lipid contents (Figure 3).")
    ap.add_argument("--input", type=Path)
    a = ap.parse_args(argv)
    t = pd.read_csv(a.input) if a.input else PAPER_LIPID
    print(t.to_string(index=False))
    print_anova(anova_summary(list(t.strain), t["mean"], t["sd"], t["n"]))
    if a.input is None:
        print("\nThe paper reports SBL2 as highest and SBL1 as lowest; this test tells whether "
              "that ranking is statistically supported with n = 4.")


# =============================================================================
# 9. Improved methods document
# =============================================================================
METHODS_MD = """# Improved methods: screening microalgae for biofuel

Based on Pereira et al. (2013), *Energies* 6:2773-2783 (doi:10.3390/en6062773).
Each point gives the change, the reason, and the `lipid_biofuel.py` command that supports it.

## 1. Culture medium (`medium`)
- Give the hydrate of FeCl3 and check the duplicate zinc source (ZnCl2 and ZnSO4 both at 1 mM stock).
- Prepare Fe-EDTA, trace metals, nitrate and phosphate as separate stocks; add vitamin B12, biotin
  and thiamine (f/2 levels), which many marine chlorophytes need.
- Record salinity, temperature, light (umol photons m-2 s-1, not just 'ambient') and photoperiod
  during the 2-week pre-enrichment, so strain selection can be repeated.

## 2. FACS isolation with BODIPY 505/515 (`facs`)
- Set gates on an **unstained control** of the same sample (and a single-stain control), so P1-P3
  are defined by BODIPY rather than autofluorescence.
- Compensate chlorophyll spill-over into FL1 (530/30) - chlorophyll emission is weak there but
  bright cells still leak.
- Dissolve BODIPY in DMSO and keep DMSO constant (e.g. 0.1-0.5 %); optimise dye concentration and
  staining time per strain, as uptake differs between species.
- Report the BODIPY signal per unit chlorophyll (FL1/FL3), which corrects for cell size; the
  `facs` command gives this as a lipid index per gate.
- Report events acquired, sort mode (single-cell / purity), and sorting efficiency (wells with
  growth / wells sorted).

## 3. Identification (`primers`)
- The 18S primer pair yields a **partial** 18S; Nannochloris SBL1 and SBL4 could not be separated
  this way. Sequence ITS2 (and/or rbcL or tufA) for **all** isolates, not only two.
- Check primers in silico against reference sequences (`primers --template refs.fasta`), and
  choose annealing temperature by gradient PCR around the calculated Tm.
- Deposit sequences in GenBank and give accession numbers; build a phylogenetic tree with
  reference strains rather than relying on top BLAST hits.

## 4. Microscopy (`droplets`)
- Quantify instead of only showing images: droplets per cell, droplet area per cell and mean
  intensity from >= 50 cells per strain, with identical exposure settings.
- State pixel size and exposure; include an unstained control image.

## 5. Total lipid (Bligh & Dyer) (`lipid`)
- State whether 100 mg biomass is **dry (freeze-dried)** or wet; if wet, measure moisture and
  report lipid per dry weight.
- Follow the classical two-step ratios and count the water in the biomass: first a single phase
  (chloroform:methanol:water 1:2:0.8), then add chloroform and water to 2:2:1.8 to split the
  phases. The paper adds all three at once (2:2:1), which can extract less from wet cells.
- Correct for the aliquot: lipid = residue x (total organic volume / aliquot volume); subtract a
  solvent blank; dry to constant weight under N2.
- Gravimetric totals include pigments and polar lipids that are **not** converted to biodiesel;
  report FAME content (next point) alongside.

## 6. FAME profile (`fame`)
- Add an **internal standard** (e.g. C17:0 or C19:0 methyl ester / triheptadecanoin) before
  transesterification and calibrate with a FAME standard mix (e.g. Supelco 37). This gives
  **absolute** FAME content (mg/g DW) - the actual biodiesel yield - instead of only relative %.
- Use dry biomass (or record water content) for the 0.1 g sample.
- Report how the iodine value was calculated; with the standard equation the values are lower
  than reported (see `fame`).
- Predict the main biodiesel properties (cetane number, CFPP, oxidation stability, viscosity,
  density) from the profile and compare with EN 14214 and ASTM D6751; the paper checks only
  iodine value, C18:3 and >= 4 double bonds.

## 7. GC-MS (`gc`)
- Add a final hold (3-5 min at 300 C); identify FAME by retention time of standards **and** mass
  spectra; report the column and split ratio.

## 8. Growth and productivity
- Lipid **productivity** (mg L-1 d-1) = biomass productivity x lipid content; it ranks strains
  for biofuel better than lipid % alone. Measure dry weight over the growth curve.
- Test lipid induction (N or P starvation, high light) - the paper notes contents were measured
  only under optimal growth.

## 9. Statistics (`stats`)
- Test differences between strains (one-way ANOVA + Tukey), not only mean +/- SD; use n >= 3
  **biological** replicates (separate cultures).
"""


def cmd_methods(argv=None):
    ap = argparse.ArgumentParser(description="Write the improved-methods document.")
    ap.add_argument("--out", type=Path, default=OUT / "IMPROVED_METHODS.md")
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(METHODS_MD)
    print(f"Written {a.out}")


# =============================================================================
# 10. Example files, tests, dispatcher
# =============================================================================
def write_examples(folder=Path(".")):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    r = np.random.default_rng(2013)
    rows = []
    for _, s in PAPER_LIPID.iterrows():
        for rep in range(1, 5):
            pct = r.normal(s["mean"], s["sd"])
            dry = 100.0
            lipid = pct / 100 * dry
            rows.append(dict(strain=s.strain, replicate=rep, biomass_mg=dry, moisture_pct=0,
                             tube_empty_mg=round(r.uniform(5200, 5400), 2),
                             organic_total_ml=4.0, aliquot_ml=2.0, blank_mg=0.15))
            rows[-1]["tube_after_mg"] = round(rows[-1]["tube_empty_mg"] + lipid * 2 / 4 + 0.15, 2)
    pd.DataFrame(rows).to_csv(folder / "lipid_example.csv", index=False)
    PAPER_FAME.to_csv(folder / "fame_example.csv", index=False)
    PAPER_LIPID.to_csv(folder / "lipid_summary_example.csv", index=False)
    simulate_events(5000).drop(columns="truth").round(2).to_csv(folder / "facs_example.csv", index=False)
    print(f"Example files written to {folder.resolve()}: lipid_example.csv (simulated weights), "
          "fame_example.csv and lipid_summary_example.csv (published values), facs_example.csv "
          "(simulated events)")


def cmd_example(argv=None):
    write_examples(Path("."))


def cmd_test(argv=None):
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    sums = [biodiesel_properties(dict(zip(PAPER_FAME.fame, PAPER_FAME[c])))["total_pct"]
            for c in PAPER_FAME.columns[1:]]
    check("Table 2 profiles sum to ~100 %", all(abs(s - 100) < 0.1 for s in sums))
    p = biodiesel_properties({"C18:1": 100})
    check("methyl oleate iodine value ~85.7", abs(p["iodine_value"] - 85.7) < 0.2)
    check("methyl oleate saponification value ~188.9", abs(p["saponification_value"] - 188.9) < 0.3)
    check("methyl palmitate: IV 0, cetane > 70",
          biodiesel_properties({"C16:0": 100})["iodine_value"] == 0
          and biodiesel_properties({"C16:0": 100})["cetane_number"] > 70)
    check("SBL2 PUFA = 36.62 % (Table 2)",
          abs(biodiesel_properties(dict(zip(PAPER_FAME.fame, PAPER_FAME.iloc[:, 2])))["PUFA"] - 36.62) < 0.01)
    check("Tm of 20-mer in 50-65 C range", 50 < tm_nn(PRIMERS["18SUnivFor"]) < 65)
    check("GC % 18SUnivFor = 55 %", abs(gc_pct(PRIMERS["18SUnivFor"]) - 55) < 1e-9)
    check("revcomp", revcomp("ACGTT") == "AACGT")
    tmpl = "GGGG" + PRIMERS["5.8SFor"] + "A" * 300 + revcomp(PRIMERS["28SRev"]) + "CCCC"
    s1, s2 = find_site(tmpl, PRIMERS["5.8SFor"]), find_site(tmpl, PRIMERS["28SRev"])
    check("in-silico PCR amplicon = 338 bp", s2[0] + len(PRIMERS["28SRev"]) - s1[0] == 338)
    tot, _ = gc_programme(PAPER_GC)
    check("paper GC programme = 48.5 min", abs(tot - 48.5) < 1e-9)
    img, truth = synthetic_bodipy_image()
    n = len(detect_droplets(img))
    check(f"droplet detection within 10 % of truth ({n} vs {truth})", abs(n - truth) <= 0.10 * truth)
    t, _ = apply_gates(simulate_events())
    check("FACS gates recover ~40 % in P1", 35 < t.set_index("gate").loc["P1", "pct_of_total"] < 42)
    d = lipid_content(pd.DataFrame(dict(biomass_mg=[100], tube_empty_mg=[1000], tube_after_mg=[1010],
                                        organic_total_ml=[4], aliquot_ml=[2])))
    check("Bligh & Dyer aliquot correction: 10 mg x 2 / 100 mg = 20 %", abs(d.lipid_pct_dw.iloc[0] - 20) < 1e-9)
    res = anova_summary(["a", "b"], [10, 10], [1, 1], [4, 4])
    check("ANOVA with equal means: F = 0", abs(res["F"]) < 1e-12)
    t = medium_table()
    check("ALGAL nitrate 2000 uM final", abs(t.loc[t.component == "NaNO3", "final_uM"].iloc[0] - 2000) < 1e-9)
    print(f"\n{sum(ok)}/{len(ok)} tests passed")
    if not all(ok):
        sys.exit(1)


def cmd_all(argv=None):
    write_examples(Path("."))
    for title, fn, args in [
        ("CULTURE MEDIUM", cmd_medium, []), ("FACS GATING", cmd_facs, []),
        ("PRIMERS", cmd_primers, []), ("LIPID DROPLETS", cmd_droplets, []),
        ("TOTAL LIPID (BLIGH & DYER)", cmd_lipid, []), ("STATISTICS ON PUBLISHED LIPIDS", cmd_stats, []),
        ("FAME -> BIODIESEL QUALITY", cmd_fame, []), ("GC PROGRAMME", cmd_gc, []),
        ("IMPROVED METHODS", cmd_methods, []), ("TESTS", cmd_test, [])]:
        print(f"\n{'#' * 72}\n# {title}\n{'#' * 72}")
        fn(args)


COMMANDS = {"medium": cmd_medium, "facs": cmd_facs, "primers": cmd_primers,
            "droplets": cmd_droplets, "lipid": cmd_lipid, "fame": cmd_fame, "gc": cmd_gc,
            "stats": cmd_stats, "methods": cmd_methods, "example": cmd_example,
            "test": cmd_test, "all": cmd_all}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    COMMANDS[sys.argv[1]](sys.argv[2:])
