# Genomic DNA isolation - improved for C. vulgaris, B. subtilis and living-biofilter samples

- [ ] 1. Standardise the input: harvest culture equal to OD750 x mL = 2 (e.g. 2.5 mL at OD750 0.8), or 10-20 mg wet biomass scraped from a biofilter.
    > Why: '2 mL' gives different amounts of cells for different cultures; yields cannot be compared without a fixed input.
- [ ] 2. BIOFILTER SAMPLES ONLY: scrape cells from the cotton, wash twice in 1 mL 0.9 % NaCl (or STE), 5,000 x g 5 min, to remove guar gum/carrageenan binder. (~15 min)
    > Why: The GG/CG binder is polysaccharide; it co-precipitates with DNA, lowers A260/230 and inhibits PCR.
- [ ] 3. Pellet cells 5,000 x g, 5 min; discard supernatant; wash once in 500 uL STE. (~10 min)
    > Why: Speeds are given in x g so they transfer between centrifuges (8000 rpm is ~4,000-6,000 x g depending on the rotor). The wash removes medium and EPS.
- [ ] 4. Resuspend in 500 uL STE. Add ~200 mg 0.5 mm glass or zirconia beads; bead-beat 3 x 30 s (on ice between), OR do 3 freeze-thaw cycles (-80 C / liquid N2 <-> 65 C). (~10 min)
    > Why: Chlorella walls are resistant to lysozyme; mechanical or freeze-thaw disruption is the step that most improves yield from green algae.
- [ ] 5. Add 30 uL lysozyme (50 mg/mL; ~2.8 mg/mL final). Incubate 30 min. [37 C] (~30 min)
    > Why: Lysozyme lyses Bacillus subtilis (Gram-positive) walls. It works best at 37 C and BEFORE SDS, which inhibits it; in the original all reagents go in together at 60-65 C.
- [ ] 6. Add 30 uL 20 % SDS (~1 % final) and 10 uL proteinase K (20 mg/mL). Incubate 1 h, inverting every 15 min. [55 C] (~60 min)
    > Why: SDS at ~1 % final (the original 20 uL may give only 0.3-0.7 %); proteinase K is most active at 50-56 C.
- [ ] 7. OPTIONAL for pigment/polysaccharide-rich or biofilter samples: add 100 uL 5 M NaCl and 80 uL CTAB/NaCl (10 % CTAB in 0.7 M NaCl); 65 C 10 min; extract once with an equal volume of chloroform:isoamyl alcohol (24:1). [65 C] (~15 min)
    > Why: CTAB removes polysaccharides and pigments that the original method leaves in.
- [ ] 8. Mix by INVERSION (not vortex) with an equal volume of PCI 25:24:1 (Tris-saturated phenol, pH 8.0) for 5 min. Work in a fume hood. (~5 min)
    > Why: Vortexing after lysis shears genomic DNA (smears on the gel). Acidic phenol pulls DNA into the organic phase, so the phenol must be at pH ~8.
- [ ] 9. Centrifuge 12,000 x g, 10 min, 4 C. Transfer the aqueous phase with a wide-bore tip; leave the interphase. [4 C] (~10 min)
    > Why: A higher force gives a sharper interphase and less protein carry-over.
- [ ] 10. Back-extract the aqueous phase with an equal volume of chloroform:isoamyl alcohol (24:1); 12,000 x g, 5 min. Transfer the aqueous phase. (~7 min)
    > Why: Removes residual phenol, which absorbs at 270 nm and inhibits PCR.
- [ ] 11. RNase A: 2 uL of 10 mg/mL, 37 C, 30 min. [37 C] (~30 min)
    > Why: Algae contain much RNA; the bright low band in lane 3 of the original Fig. 1 is typical of RNA, and RNA inflates A260 readings.
- [ ] 12. Add an equal volume of 4 M ammonium acetate (2 M final), then 0.7 volume isopropanol at room temperature. Invert; stand 10 min (or -20 C 30 min for low yields; add 1 uL 20 mg/mL glycogen as carrier if needed). (~10 min)
    > Why: Brings the salt to the usual 2-2.5 M range (the original gives ~1.8 M). The `precip` command checks the numbers for your volumes.
- [ ] 13. Centrifuge 15,000 x g, 20 min, 4 C. Discard supernatant. [4 C] (~20 min)
    > Why: 8 min at ~5,000 x g can leave much DNA unpelleted.
- [ ] 14. Wash twice with 500 uL ice-cold 70 % ethanol; 15,000 x g, 5 min each. (~12 min)
    > Why: A second wash removes more salt (improves A260/230).
- [ ] 15. Air-dry 5-10 min (do not over-dry). Dissolve in 30-50 uL TE (or 10 mM Tris-HCl pH 8.5 for sequencing), 55 C 10 min. Store at 4 C (days) or -20 C. [55 C] (~20 min)
    > Why: The original lists drying after adding TE. Over-dried pellets are hard to redissolve; EDTA in TE can inhibit some downstream enzymes.
- [ ] 16. QC: NanoDrop (A260/280 1.8-2.0, A260/230 >= 1.8), Qubit dsDNA for accurate concentration, and 0.8 % agarose gel with a 1 kb ladder (expect one band > 10 kb). Run an extraction blank (no cells) with every batch. Check with PCR, e.g. 16S rRNA (27F/1492R) for B. subtilis and 18S/rbcL for C. vulgaris. (~60 min)
    > Why: The original reports only a gel without a ladder or controls; purity and yield numbers make extractions comparable (see the `qc` command).

Approximate total time: 5.2 h (hands-on time not included)

## Safety
- Phenol and chloroform are toxic and volatile: fume hood, nitrile gloves, goggles;
  collect organic waste separately.
- SDS and CTAB powders are irritants: weigh with a mask or in a hood.
- Liquid nitrogen: cryogenic gloves and face shield.
- DNA stains (ethidium bromide, SYBR, GelRed): follow your institution's rules.
