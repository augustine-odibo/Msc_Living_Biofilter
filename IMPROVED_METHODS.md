# Improved methods: screening microalgae for biofuel

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
