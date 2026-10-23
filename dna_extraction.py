#!/usr/bin/env python3
"""
dna_extraction.py - tools for genomic DNA isolation from microalgae and the
C. vulgaris / B. subtilis living biofilters.

Based on: Thajuddin F, Meenakshisundaram J, Nooruddin T, Dharumadurai D (2025).
Isolation of Genomic DNA from Microalgae. In: Plant Microbiome Engineering,
Methods and Protocols in Food Science. Springer. doi:10.1007/978-1-0716-4180-4_24
(a modification of Smoker & Barnum 1988, FEMS Microbiol Lett 56:119-122).

COMMANDS
  protocol    print the original or the improved protocol as a bench checklist
              (--markdown FILE writes it as a document)
  plan        reagent volumes for N samples (with overage) for either protocol
  buffers     buffer recipes from standard stocks, and a check of the
              published STE/TE recipes
  rcf         convert rpm <-> x g for your rotor (the protocol only gives rpm)
  precip      final salt / alcohol concentrations of the precipitation step
  qc          DNA concentration, purity ratios, yield and PCR dilutions from
              NanoDrop / spectrophotometer readings (CSV)
  gel         fragment size of bands from migration distance against a ladder
  example     write example input files (qc_example.csv, gel_example.csv)
  test        self-tests
  all         run every command on the example files

    python dna_extraction.py protocol --improved
    python dna_extraction.py plan --samples 12
    python dna_extraction.py rcf --rpm 8000 --radius 7.3
    python dna_extraction.py qc --input my_nanodrop.csv
    python dna_extraction.py <command> --help

Requires: numpy, pandas, matplotlib.   License: MIT.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.cwd() / "results" / "dna"


# =============================================================================
# 1. Unit helpers
# =============================================================================
def rcf_from_rpm(rpm, radius_cm):
    """Relative centrifugal force (x g) = 1.118e-5 * r(cm) * rpm^2."""
    return 1.118e-5 * radius_cm * np.asarray(rpm, dtype=float) ** 2


def rpm_from_rcf(rcf, radius_cm):
    return np.sqrt(np.asarray(rcf, dtype=float) / (1.118e-5 * radius_cm))


def c1v1(c_stock, c_final, v_final):
    """Volume of stock needed: V1 = C2 * V2 / C1 (any consistent units)."""
    return c_final * v_final / c_stock


def final_conc(c_stock, v_added, v_total):
    return c_stock * v_added / v_total


# =============================================================================
# 2. Protocols (original as published, and improved)
# =============================================================================
@dataclass
class Step:
    text: str
    minutes: float = 0.0          # hands-off/incubation time
    temp_c: str = ""
    why: str = ""                 # reason for the change (improved protocol)


ORIGINAL = [
    Step("Centrifuge 2 mL of culture, 8000 rpm, 8 min; discard supernatant.", 8),
    Step("Add 500 uL STE buffer, 30 uL lysozyme, 30 uL proteinase K, 20 uL SDS to the pellet."),
    Step("Vortex 5 min.", 5),
    Step("Mix thoroughly; water bath 1.5 h.", 90, "60-65"),
    Step("Cool to room temperature.", 5),
    Step("Add an equal volume of phenol:chloroform:isoamyl alcohol (25:24:1); mix gently."),
    Step("Centrifuge 8000 rpm, 8 min.", 8),
    Step("Transfer the upper (aqueous) phase to a fresh tube."),
    Step("Add 400 uL 4 M ammonium acetate and 600 uL isopropanol; centrifuge 8000 rpm, 8 min.", 8),
    Step("Remove the supernatant without disturbing the pellet."),
    Step("Wash with 70 % ethanol; centrifuge 8000 rpm, 8 min.", 8),
    Step("Discard supernatant; air-dry the pellet.", 10),
    Step("Add 20-50 uL TE; store at -20 C."),
    Step("Load 4 uL on an agarose gel; image in a gel documentation system.", 45),
]

IMPROVED = [
    Step("Standardise the input: harvest culture equal to OD750 x mL = 2 "
         "(e.g. 2.5 mL at OD750 0.8), or 10-20 mg wet biomass scraped from a biofilter.",
         why="'2 mL' gives different amounts of cells for different cultures; yields "
             "cannot be compared without a fixed input."),
    Step("BIOFILTER SAMPLES ONLY: scrape cells from the cotton, wash twice in 1 mL "
         "0.9 % NaCl (or STE), 5,000 x g 5 min, to remove guar gum/carrageenan binder.", 15,
         why="The GG/CG binder is polysaccharide; it co-precipitates with DNA, lowers "
             "A260/230 and inhibits PCR."),
    Step("Pellet cells 5,000 x g, 5 min; discard supernatant; wash once in 500 uL STE.", 10,
         why="Speeds are given in x g so they transfer between centrifuges (8000 rpm is "
             "~4,000-6,000 x g depending on the rotor). The wash removes medium and EPS."),
    Step("Resuspend in 500 uL STE. Add ~200 mg 0.5 mm glass or zirconia beads; bead-beat "
         "3 x 30 s (on ice between), OR do 3 freeze-thaw cycles (-80 C / liquid N2 <-> 65 C).", 10,
         why="Chlorella walls are resistant to lysozyme; mechanical or freeze-thaw "
             "disruption is the step that most improves yield from green algae."),
    Step("Add 30 uL lysozyme (50 mg/mL; ~2.8 mg/mL final). Incubate 30 min.", 30, "37",
         why="Lysozyme lyses Bacillus subtilis (Gram-positive) walls. It works best at 37 C "
             "and BEFORE SDS, which inhibits it; in the original all reagents go in together "
             "at 60-65 C."),
    Step("Add 30 uL 20 % SDS (~1 % final) and 10 uL proteinase K (20 mg/mL). "
         "Incubate 1 h, inverting every 15 min.", 60, "55",
         why="SDS at ~1 % final (the original 20 uL may give only 0.3-0.7 %); proteinase K is "
             "most active at 50-56 C."),
    Step("OPTIONAL for pigment/polysaccharide-rich or biofilter samples: add 100 uL 5 M NaCl and "
         "80 uL CTAB/NaCl (10 % CTAB in 0.7 M NaCl); 65 C 10 min; extract once with an equal "
         "volume of chloroform:isoamyl alcohol (24:1).", 15, "65",
         why="CTAB removes polysaccharides and pigments that the original method leaves in."),
    Step("Mix by INVERSION (not vortex) with an equal volume of PCI 25:24:1 "
         "(Tris-saturated phenol, pH 8.0) for 5 min. Work in a fume hood.", 5,
         why="Vortexing after lysis shears genomic DNA (smears on the gel). Acidic phenol "
             "pulls DNA into the organic phase, so the phenol must be at pH ~8."),
    Step("Centrifuge 12,000 x g, 10 min, 4 C. Transfer the aqueous phase with a "
         "wide-bore tip; leave the interphase.", 10, "4",
         why="A higher force gives a sharper interphase and less protein carry-over."),
    Step("Back-extract the aqueous phase with an equal volume of chloroform:isoamyl alcohol "
         "(24:1); 12,000 x g, 5 min. Transfer the aqueous phase.", 7,
         why="Removes residual phenol, which absorbs at 270 nm and inhibits PCR."),
    Step("RNase A: 2 uL of 10 mg/mL, 37 C, 30 min.", 30, "37",
         why="Algae contain much RNA; the bright low band in lane 3 of the original Fig. 1 "
             "is typical of RNA, and RNA inflates A260 readings."),
    Step("Add an equal volume of 4 M ammonium acetate (2 M final), then 0.7 volume "
         "isopropanol at room temperature. Invert; stand 10 min (or -20 C 30 min for low yields; "
         "add 1 uL 20 mg/mL glycogen as carrier if needed).", 10,
         why="Brings the salt to the usual 2-2.5 M range (the original gives ~1.8 M). "
             "The `precip` command checks the numbers for your volumes."),
    Step("Centrifuge 15,000 x g, 20 min, 4 C. Discard supernatant.", 20, "4",
         why="8 min at ~5,000 x g can leave much DNA unpelleted."),
    Step("Wash twice with 500 uL ice-cold 70 % ethanol; 15,000 x g, 5 min each.", 12,
         why="A second wash removes more salt (improves A260/230)."),
    Step("Air-dry 5-10 min (do not over-dry). Dissolve in 30-50 uL TE (or 10 mM Tris-HCl "
         "pH 8.5 for sequencing), 55 C 10 min. Store at 4 C (days) or -20 C.", 20, "55",
         why="The original lists drying after adding TE. Over-dried pellets are hard to "
             "redissolve; EDTA in TE can inhibit some downstream enzymes."),
    Step("QC: NanoDrop (A260/280 1.8-2.0, A260/230 >= 1.8), Qubit dsDNA for accurate "
         "concentration, and 0.8 % agarose gel with a 1 kb ladder (expect one band > 10 kb). "
         "Run an extraction blank (no cells) with every batch. Check with PCR, e.g. 16S rRNA "
         "(27F/1492R) for B. subtilis and 18S/rbcL for C. vulgaris.", 60,
         why="The original reports only a gel without a ladder or controls; purity "
             "and yield numbers make extractions comparable (see the `qc` command)."),
]

# reagent use per sample: (name, uL original, uL improved, note)
REAGENTS = [
    ("STE buffer", 500, 1000, "improved: 500 wash + 500 lysis"),
    ("Lysozyme (50 mg/mL)", 30, 30, "make fresh or store aliquots at -20 C"),
    ("Proteinase K (20 mg/mL)", 30, 10, ""),
    ("SDS (original: stock not stated; improved: 20 %)", 20, 30, ""),
    ("0.9 % NaCl wash (biofilter samples)", 0, 2000, "biofilter samples only"),
    ("5 M NaCl (CTAB option)", 0, 100, "optional"),
    ("CTAB/NaCl 10 % / 0.7 M (CTAB option)", 0, 80, "optional"),
    ("Chloroform:isoamyl alcohol 24:1", 0, 1200, "CTAB option + back-extraction"),
    ("PCI 25:24:1, pH 8.0", 580, 600, "equal volume of lysate"),
    ("RNase A (10 mg/mL)", 0, 2, ""),
    ("4 M ammonium acetate", 400, 500, "improved: equal volume of aqueous phase"),
    ("Isopropanol", 600, 700, "improved: 0.7 vol of aqueous + salt"),
    ("70 % ethanol (ice-cold)", 500, 1000, "improved: two washes"),
    ("TE buffer", 50, 50, ""),
    ("0.5 mm glass/zirconia beads (mg)", 0, 200, "mg, not uL"),
]


def protocol_text(steps, title, markdown=False):
    lines = [f"# {title}" if markdown else title.upper(), ""]
    total = 0
    for i, s in enumerate(steps, 1):
        total += s.minutes
        t = f" [{s.temp_c} C]" if s.temp_c else ""
        d = f" (~{s.minutes:g} min)" if s.minutes else ""
        box = "- [ ]" if markdown else "[ ]"
        lines.append(f"{box} {i}. {s.text}{t}{d}")
        if s.why:
            lines.append(f"    {'> ' if markdown else ''}Why: {s.why}")
    lines += ["", f"Approximate total time: {total / 60:.1f} h (hands-on time not included)"]
    return "\n".join(lines)


def cmd_protocol(argv=None):
    ap = argparse.ArgumentParser(description="Print the protocol as a checklist.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--original", action="store_true")
    g.add_argument("--improved", action="store_true", help="default")
    ap.add_argument("--markdown", type=Path, help="write the checklist to this .md file")
    a = ap.parse_args(argv)
    if a.original:
        txt = protocol_text(ORIGINAL, "Genomic DNA isolation - original (Thajuddin et al. 2025)",
                            bool(a.markdown))
    else:
        txt = protocol_text(IMPROVED, "Genomic DNA isolation - improved for C. vulgaris, "
                            "B. subtilis and living-biofilter samples", bool(a.markdown))
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(txt + "\n" + SAFETY_MD)
        print(f"Written {a.markdown}")
    else:
        print(txt)


SAFETY_MD = """
## Safety
- Phenol and chloroform are toxic and volatile: fume hood, nitrile gloves, goggles;
  collect organic waste separately.
- SDS and CTAB powders are irritants: weigh with a mask or in a hood.
- Liquid nitrogen: cryogenic gloves and face shield.
- DNA stains (ethidium bromide, SYBR, GelRed): follow your institution's rules.
"""


# =============================================================================
# 3. Reagent planning for a batch
# =============================================================================
def reagent_plan(n_samples, improved=True, overage=0.10, biofilter=True, ctab=True):
    rows = []
    n_eff = n_samples * (1 + overage) + 1          # +10 % and one extra sample
    for name, v_orig, v_imp, note in REAGENTS:
        v = v_imp if improved else v_orig
        if v == 0:
            continue
        if improved and not biofilter and "biofilter" in note:
            continue
        if improved and not ctab and "CTAB option" in name:
            continue
        unit = "mg" if "(mg)" in name else "uL"
        total = v * n_eff
        rows.append(dict(reagent=name, per_sample=v, unit=unit,
                         total=round(total, 1),
                         total_mL=round(total / 1000, 3) if unit == "uL" else None,
                         note=note))
    return pd.DataFrame(rows)


def cmd_plan(argv=None):
    ap = argparse.ArgumentParser(description="Reagents needed for a batch of extractions.")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--original", action="store_true")
    ap.add_argument("--overage", type=float, default=0.10)
    ap.add_argument("--no-biofilter", action="store_true", help="liquid cultures only")
    ap.add_argument("--no-ctab", action="store_true")
    a = ap.parse_args(argv)
    df = reagent_plan(a.samples, not a.original, a.overage, not a.no_biofilter, not a.no_ctab)
    n_eff = a.samples * (1 + a.overage) + 1
    print(f"Reagents for {a.samples} samples ({'original' if a.original else 'improved'} "
          f"protocol, +{a.overage:.0%} overage +1 extra = {n_eff:.1f} sample-equivalents)\n")
    print(df.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"reagent_plan_{a.samples}_samples.csv"
    df.to_csv(f, index=False)
    print(f"\nSaved {f}")


# =============================================================================
# 4. Buffers
# =============================================================================
def check_published_buffers():
    """Back-calculate what the published STE/TE recipes contain."""
    out = []
    # STE: volumes given for 50 mM Tris, 5 mM EDTA, 50 mM NaCl, total 100 mL
    ste_total = 25 + 20 + 20 + 35
    for comp, final_mM, v in [("Tris", 50, 25), ("EDTA", 5, 20), ("NaCl", 50, 20)]:
        out.append(dict(buffer="STE", component=comp, stated_final_mM=final_mM,
                        volume_mL=v, total_mL=ste_total,
                        implied_stock_mM=final_mM * ste_total / v))
    # TE: 1 mL 1 M Tris-HCl + 0.2 mL 0.5 M EDTA + 100 mL water
    te_total = 1 + 0.2 + 100
    out.append(dict(buffer="TE", component="Tris-HCl", stated_final_mM=10, volume_mL=1,
                    total_mL=te_total, implied_stock_mM=1000,
                    actual_final_mM=1000 * 1 / te_total))
    out.append(dict(buffer="TE", component="EDTA", stated_final_mM=1, volume_mL=0.2,
                    total_mL=te_total, implied_stock_mM=500,
                    actual_final_mM=500 * 0.2 / te_total))
    return pd.DataFrame(out)


RECIPES = {  # final concentrations; made from standard stocks
    "STE (pH 8.0)": [("1 M Tris-HCl pH 8.0", 1000, 50, "mM"), ("0.5 M EDTA pH 8.0", 500, 5, "mM"),
                     ("5 M NaCl", 5000, 50, "mM")],
    "TE (pH 8.0)": [("1 M Tris-HCl pH 8.0", 1000, 10, "mM"), ("0.5 M EDTA pH 8.0", 500, 1, "mM")],
    "CTAB extraction buffer": [("1 M Tris-HCl pH 8.0", 1000, 100, "mM"),
                               ("0.5 M EDTA pH 8.0", 500, 20, "mM"), ("5 M NaCl", 5000, 1400, "mM"),
                               ("10 % (w/v) CTAB", 10, 2, "%")],
    "TAE 1x": [("50x TAE", 50, 1, "x")],
}


def recipe(name, volume_ml):
    rows, used = [], 0.0
    for stock, c_stock, c_final, unit in RECIPES[name]:
        v = c1v1(c_stock, c_final, volume_ml)
        used += v
        rows.append(dict(component=stock, final=f"{c_final:g} {unit}", volume_mL=round(v, 3)))
    rows.append(dict(component="Water (to final volume)", final="",
                     volume_mL=round(volume_ml - used, 3)))
    return pd.DataFrame(rows)


def cmd_buffers(argv=None):
    ap = argparse.ArgumentParser(description="Buffer recipes and recipe check.")
    ap.add_argument("--volume", type=float, default=100.0, help="mL of each buffer")
    a = ap.parse_args(argv)
    print("CHECK OF THE PUBLISHED RECIPES")
    print(check_published_buffers().round(2).to_string(index=False))
    print("""
  * STE: the volumes only give 50/5/50 mM if the stocks are 200 mM Tris, 25 mM EDTA and
    250 mM NaCl, which the protocol does not state. Use the recipe below from standard stocks,
    and use Tris-HCl adjusted to pH 8.0 (Tris base alone gives pH ~10.5).
  * TE: '100 mL water' makes 101.2 mL; add water TO 100 mL for exactly 10 mM / 1 mM.
""")
    for name in RECIPES:
        print(f"{name}, {a.volume:g} mL")
        print(recipe(name, a.volume).to_string(index=False), "\n")
    print("Lysis additives in 580 uL (original protocol) for common stock strengths:")
    rows = []
    for reag, v, stocks in [("Lysozyme", 30, [10, 50]), ("Proteinase K", 30, [20]),
                            ("SDS", 20, [10, 20])]:
        for s in stocks:
            unit = "%" if reag == "SDS" else "mg/mL"
            rows.append(dict(reagent=reag, stock=f"{s} {unit}", added_uL=v,
                             final=f"{final_conc(s, v, 580):.2f} {unit}"))
    print(pd.DataFrame(rows).to_string(index=False))
    print("  -> SDS ends up at 0.3-0.7 %; ~1 % final is more usual for complete lysis.")


# =============================================================================
# 5. Centrifuge speed
# =============================================================================
def cmd_rcf(argv=None):
    ap = argparse.ArgumentParser(description="rpm <-> x g. Radius = rotor centre to tube bottom.")
    ap.add_argument("--rpm", type=float)
    ap.add_argument("--g", type=float)
    ap.add_argument("--radius", type=float, default=7.3, help="cm (typical microcentrifuge 6-9)")
    a = ap.parse_args(argv)
    if a.g:
        print(f"{a.g:,.0f} x g at r = {a.radius} cm  =  {rpm_from_rcf(a.g, a.radius):,.0f} rpm")
    rpm = a.rpm or 8000
    print(f"{rpm:,.0f} rpm at r = {a.radius} cm  =  {rcf_from_rpm(rpm, a.radius):,.0f} x g")
    print("\nThe protocol's 8000 rpm on common microcentrifuge rotors:")
    for r in (6.0, 7.3, 8.4, 9.5):
        print(f"  r = {r:4.1f} cm  ->  {rcf_from_rpm(8000, r):6,.0f} x g")


# =============================================================================
# 6. Precipitation step
# =============================================================================
def precipitation(aqueous_ul, salt_stock_m=4.0, salt_ul=400.0, isopropanol_ul=600.0):
    v_salted = aqueous_ul + salt_ul
    return dict(
        aqueous_uL=aqueous_ul,
        salt_before_alcohol_M=salt_stock_m * salt_ul / v_salted,
        salt_in_final_mix_M=salt_stock_m * salt_ul / (v_salted + isopropanol_ul),
        isopropanol_volumes=isopropanol_ul / v_salted,
        isopropanol_pct_v_v=100 * isopropanol_ul / (v_salted + isopropanol_ul),
        total_uL=v_salted + isopropanol_ul,
    )


def cmd_precip(argv=None):
    ap = argparse.ArgumentParser(description="Check salt and alcohol in the precipitation step.")
    ap.add_argument("--aqueous", type=float, default=500, help="uL aqueous phase recovered")
    ap.add_argument("--salt-stock", type=float, default=4.0, help="M ammonium acetate")
    ap.add_argument("--salt", type=float, default=400, help="uL added")
    ap.add_argument("--isopropanol", type=float, default=600, help="uL added")
    a = ap.parse_args(argv)
    r = precipitation(a.aqueous, a.salt_stock, a.salt, a.isopropanol)
    print("Your precipitation:")
    for k, v in r.items():
        print(f"  {k:24s} {v:8.2f}")
    ok_salt = 2.0 <= r["salt_before_alcohol_M"] <= 2.5
    ok_iso = 0.6 <= r["isopropanol_volumes"] <= 1.0
    print(f"\n  NH4OAc before alcohol: target 2.0-2.5 M  -> {'OK' if ok_salt else 'adjust'}")
    print(f"  Isopropanol: target 0.6-1.0 volumes      -> {'OK' if ok_iso else 'adjust'}")
    salt = a.aqueous * 2.0 / (a.salt_stock - 2.0)
    iso = 0.7 * (a.aqueous + salt)
    print(f"\n  Suggested for {a.aqueous:g} uL aqueous: {salt:.0f} uL {a.salt_stock:g} M NH4OAc "
          f"(2 M final) + {iso:.0f} uL isopropanol (0.7 vol); tube total {a.aqueous + salt + iso:.0f} uL"
          + ("  (> 1.5 mL: split into two tubes or use 2 mL tubes)"
             if a.aqueous + salt + iso > 1500 else ""))


# =============================================================================
# 7. Quality control from absorbance readings
# =============================================================================
def dna_qc(df, target_ng_ul=20.0, pcr_volume_ul=50.0, ext=50.0):
    """Concentration, purity, yield. ext = ug/mL per A260 unit (50 for dsDNA)."""
    d = df.copy()
    d["dilution"] = d.get("dilution", 1).fillna(1) if "dilution" in d else 1
    d["path_cm"] = d.get("path_cm", 1).fillna(1) if "path_cm" in d else 1
    if "conc_ng_ul" not in d or d["conc_ng_ul"].isna().any():
        calc = d.a260 * ext * d.dilution / d.path_cm
        d["conc_ng_ul"] = d["conc_ng_ul"].fillna(calc) if "conc_ng_ul" in d else calc
    d["A260_280"] = d.a260 / d.a280
    d["A260_230"] = d.a260 / d.a230
    d["yield_ug"] = d.conc_ng_ul * d.elution_ul / 1000
    if "input_amount" in d:
        d["yield_per_input"] = d.yield_ug / d.input_amount.where(d.input_amount > 0)
    flags = []
    for _, r in d.iterrows():
        f = []
        if r.conc_ng_ul < 2:
            flags.append("below ~2 ng/uL: ratios not meaningful (expected for a blank)")
            continue
        if r.A260_280 < 1.7:
            f.append("protein/phenol (260/280 < 1.7)")
        elif r.A260_280 > 2.05:
            f.append("RNA likely (260/280 > 2.05): RNase A")
        if r.A260_230 < 1.8:
            f.append("salt/polysaccharide/phenol (260/230 < 1.8): extra 70% EtOH wash or CTAB")
        if r.conc_ng_ul < 5:
            f.append("low concentration: Qubit recommended")
        if r.conc_ng_ul <= target_ng_ul:
            f.append(f"at or below {target_ng_ul:g} ng/uL: use undiluted")
        flags.append("; ".join(f) if f else "pass")
    d["qc_result"] = flags
    # dilution for PCR template: uL DNA + uL water to reach target in pcr_volume_ul
    d["uL_DNA_for_target"] = np.where(d.conc_ng_ul > target_ng_ul,
                                      c1v1(d.conc_ng_ul, target_ng_ul, pcr_volume_ul), np.nan)
    d["uL_water_for_target"] = pcr_volume_ul - d.uL_DNA_for_target
    return d


def cmd_qc(argv=None):
    ap = argparse.ArgumentParser(
        description="QC from absorbance. CSV columns: sample, a260, a280, a230, elution_ul "
                    "[, dilution, path_cm, conc_ng_ul, input_amount, input_unit]")
    ap.add_argument("--input", type=Path, default=Path("qc_example.csv"))
    ap.add_argument("--target", type=float, default=20.0, help="ng/uL working stock for PCR")
    ap.add_argument("--working-volume", type=float, default=50.0, help="uL of working stock")
    a = ap.parse_args(argv)
    if not a.input.exists():
        write_examples(a.input.parent)
    d = dna_qc(pd.read_csv(a.input), a.target, a.working_volume)
    cols = ["sample", "conc_ng_ul", "A260_280", "A260_230", "yield_ug"] + \
           (["yield_per_input"] if "yield_per_input" in d else []) + \
           ["uL_DNA_for_target", "uL_water_for_target", "qc_result"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(d[cols].round(2).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "dna_qc_results.csv", index=False)
    plot_qc(d, OUT / "dna_qc.png")
    print(f"\nSaved {OUT / 'dna_qc_results.csv'} and {OUT / 'dna_qc.png'}")


def plot_qc(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(d))
    ax[0].bar(x, d.yield_ug, color="#548235")
    ax[0].set_xticks(x, d["sample"], rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("DNA yield (ug)")
    bad = d["qc_result"].str.contains("260/|below ~2")            # purity problems only
    ax[1].scatter(d.A260_280, d.A260_230, c=np.where(bad, "#C00000", "#548235"))
    for _, r in d.iterrows():
        ax[1].annotate(r["sample"], (r.A260_280, r.A260_230), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].axvspan(1.8, 2.0, color="grey", alpha=0.15)
    ax[1].axhline(1.8, ls="--", color="grey", lw=0.8)
    ax[1].set(xlabel="A260/280 (target 1.8-2.0)", ylabel="A260/230 (target >= 1.8)")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# =============================================================================
# 8. Gel: fragment size from a ladder
# =============================================================================
GENERULER_1KB = [10000, 8000, 6000, 5000, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 750, 500, 250]


def fit_ladder(sizes_bp, distances):
    """Semi-log calibration: log10(bp) vs distance, fitted with a 2nd-order polynomial."""
    d = np.asarray(distances, float)
    y = np.log10(np.asarray(sizes_bp, float))
    deg = 2 if len(d) >= 5 else 1
    return np.poly1d(np.polyfit(d, y, deg))


def band_size(model, distance):
    return 10 ** model(np.asarray(distance, float))


def cmd_gel(argv=None):
    ap = argparse.ArgumentParser(
        description="Band sizes from migration distance. CSV columns: lane, kind (ladder/sample), "
                    "size_bp (ladder only), distance_mm")
    ap.add_argument("--input", type=Path, default=Path("gel_example.csv"))
    a = ap.parse_args(argv)
    if not a.input.exists():
        write_examples(a.input.parent)
    g = pd.read_csv(a.input)
    lad = g[g.kind == "ladder"].dropna(subset=["size_bp"])
    model = fit_ladder(lad.size_bp, lad.distance_mm)
    pred = band_size(model, lad.distance_mm)
    err = np.abs(pred - lad.size_bp) / lad.size_bp * 100
    print(f"Ladder fit: mean error {err.mean():.1f} % (max {err.max():.1f} %) over {len(lad)} bands")
    s = g[g.kind != "ladder"].copy()
    lo, hi = lad.distance_mm.min(), lad.distance_mm.max()
    est = band_size(model, s.distance_mm)
    big, small = lad.size_bp.max(), lad.size_bp.min()
    # never extrapolate beyond the ladder: report a bound instead
    s["estimated_size"] = np.where(s.distance_mm < lo, f"> {big:,.0f} bp",
                          np.where(s.distance_mm > hi, f"< {small:,.0f} bp",
                                   [f"~{v:,.0f} bp" for v in np.round(est, -1)]))
    s["note"] = np.where(s.distance_mm < lo, "high-molecular-weight genomic DNA (good)",
                         np.where(s.distance_mm > hi, "likely RNA or degraded DNA: add RNase A, "
                                  "check for shearing", ""))
    s = s.drop(columns=["size_bp"])
    print(s.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "gel_band_sizes.csv", index=False)


# =============================================================================
# 9. Example files and tests
# =============================================================================
def write_examples(folder=Path(".")):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    qc = pd.DataFrame({
        "sample": ["CV_culture_1", "CV_culture_2", "BS_culture_1", "Biofilter_1to1_A",
                   "Biofilter_1to1_B", "Biofilter_3to1_A", "Extraction_blank"],
        "a260": [1.10, 0.95, 1.60, 0.42, 0.38, 0.55, 0.004],
        "a280": [0.59, 0.52, 0.85, 0.24, 0.20, 0.30, 0.003],
        "a230": [0.52, 0.60, 0.75, 0.35, 0.19, 0.28, 0.006],
        "elution_ul": [50, 50, 50, 30, 30, 30, 30],
        "input_amount": [2.0, 2.0, 2.0, 15, 15, 15, 0],
        "input_unit": ["OD750*mL"] * 3 + ["mg wet biomass"] * 3 + ["none"],
    })
    qc.to_csv(folder / "qc_example.csv", index=False)
    ladder = [(s, d) for s, d in zip(GENERULER_1KB,
              [12.0, 14.0, 17.0, 19.0, 21.5, 23.0, 25.0, 27.5, 30.5, 35.0, 41.5, 46.5, 53.0, 62.0])]
    rows = [dict(lane="M", kind="ladder", size_bp=s, distance_mm=d) for s, d in ladder]
    rows += [dict(lane="1 C. vulgaris", kind="sample", size_bp=None, distance_mm=10.5),
             dict(lane="2 B. subtilis", kind="sample", size_bp=None, distance_mm=10.0),
             dict(lane="3 Biofilter", kind="sample", size_bp=None, distance_mm=11.0),
             dict(lane="3 Biofilter (low band)", kind="sample", size_bp=None, distance_mm=66.0),
             dict(lane="4 16S PCR product", kind="sample", size_bp=None, distance_mm=36.2)]
    pd.DataFrame(rows).to_csv(folder / "gel_example.csv", index=False)
    print(f"Example files written: {folder / 'qc_example.csv'}, {folder / 'gel_example.csv'}")


def cmd_example(argv=None):
    write_examples(Path("."))


def cmd_test(argv=None):
    res = []

    def check(name, ok):
        res.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name)

    check("RCF: 8000 rpm, 7.3 cm ~ 5,223 x g", abs(rcf_from_rpm(8000, 7.3) - 5223) < 2)
    check("rpm <-> x g round trip", abs(rpm_from_rcf(rcf_from_rpm(13000, 8), 8) - 13000) < 1e-6)
    check("C1V1: 1 M -> 50 mM in 100 mL = 5 mL", abs(c1v1(1000, 50, 100) - 5) < 1e-9)
    b = check_published_buffers()
    check("STE implies 200 mM Tris stock",
          abs(b.query("buffer=='STE' and component=='Tris'").implied_stock_mM.iloc[0] - 200) < 1e-9)
    check("TE as written gives 9.88 mM Tris",
          abs(b.query("buffer=='TE' and component=='Tris-HCl'").actual_final_mM.iloc[0] - 9.881) < 1e-3)
    p = precipitation(500)
    check("original precipitation: 1.78 M NH4OAc, 0.67 vol isopropanol",
          abs(p["salt_before_alcohol_M"] - 1.778) < 1e-3 and abs(p["isopropanol_volumes"] - 0.667) < 1e-3)
    q = dna_qc(pd.DataFrame(dict(sample=["x"], a260=[1.0], a280=[0.5], a230=[0.5], elution_ul=[50])))
    check("A260 1.0 -> 50 ng/uL, 2.5 ug in 50 uL",
          abs(q.conc_ng_ul.iloc[0] - 50) < 1e-9 and abs(q.yield_ug.iloc[0] - 2.5) < 1e-9)
    check("A260/280 = 2.0 passes", q["qc_result"].iloc[0] == "pass")
    m = fit_ladder(GENERULER_1KB, np.linspace(10, 60, len(GENERULER_1KB)))
    check("ladder fit is monotonic", band_size(m, 20) > band_size(m, 40))
    df = reagent_plan(10)
    check("plan: STE for 10 samples (+10 %, +1) = 12 mL",
          abs(df.query("reagent=='STE buffer'").total.iloc[0] - 12000) < 1e-6)
    print(f"\n{sum(res)}/{len(res)} tests passed")
    if not all(res):
        sys.exit(1)


def cmd_all(argv=None):
    write_examples(Path("."))
    for title, fn, args in [
        ("BUFFERS", cmd_buffers, []), ("CENTRIFUGE", cmd_rcf, []),
        ("PRECIPITATION (original volumes)", cmd_precip, []),
        ("REAGENT PLAN", cmd_plan, ["--samples", "12"]),
        ("QC", cmd_qc, []), ("GEL", cmd_gel, []),
        ("IMPROVED PROTOCOL", cmd_protocol, ["--markdown", str(OUT / "IMPROVED_PROTOCOL.md")]),
        ("TESTS", cmd_test, [])]:
        print(f"\n{'#' * 70}\n# {title}\n{'#' * 70}")
        fn(args)


COMMANDS = {"protocol": cmd_protocol, "plan": cmd_plan, "buffers": cmd_buffers,
            "rcf": cmd_rcf, "precip": cmd_precip, "qc": cmd_qc, "gel": cmd_gel,
            "example": cmd_example, "test": cmd_test, "all": cmd_all}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    COMMANDS[sys.argv[1]](sys.argv[2:])
