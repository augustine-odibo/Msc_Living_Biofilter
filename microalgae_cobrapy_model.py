"""
Microalgae core metabolic model in COBRApy
------------------------------------------
A compact, teaching-scale model of a green microalga (Chlamydomonas-like).
It lumps the main pathways so that flux balance analysis (FBA) can show
how the cell grows autotrophically, mixotrophically and heterotrophically,
and how it partitions carbon into starch and lipid.

Sign convention: exchange flux < 0 = uptake, > 0 = secretion.
Units: fluxes in mmol/gDW/h, light in mmol photons/gDW/h, growth in 1/h.
This is a simplified model; for research use a curated genome-scale
reconstruction such as iRC1080 (BiGG) or iCre1355.

Run:  python microalgae_cobrapy_model.py
"""

from cobra import Model, Reaction, Metabolite


def build_model():
    model = Model("microalgae_core")

    # ------------------------------------------------------------ metabolites
    def met(mid, name, formula=""):
        return Metabolite(mid, name=name, formula=formula, compartment="c")

    m = {
        "photon": met("photon_c", "Photon"),
        "co2": met("co2_c", "CO2", "CO2"),
        "o2": met("o2_c", "O2", "O2"),
        "h2o": met("h2o_c", "Water", "H2O"),
        "nh4": met("nh4_c", "Ammonium", "NH4"),
        "ac": met("ac_c", "Acetate", "C2H3O2"),
        "atp": met("atp_c", "ATP"),
        "adp": met("adp_c", "ADP"),
        "nadph": met("nadph_c", "NADPH"),
        "nadp": met("nadp_c", "NADP+"),
        "nadh": met("nadh_c", "NADH"),
        "nad": met("nad_c", "NAD+"),
        "g3p": met("g3p_c", "Triose phosphate (C3)"),
        "pyr": met("pyr_c", "Pyruvate (C3)"),
        "accoa": met("accoa_c", "Acetyl-CoA (C2)"),
        "oaa": met("oaa_c", "Oxaloacetate (C4)"),
        "starch": met("starch_c", "Starch unit (C6)"),
        "lipid": met("lipid_c", "Fatty acid, C16 (lipid)"),
        "aa": met("aa_c", "Amino acid unit (C3N)"),
        "biomass": met("biomass_c", "Biomass"),
    }

    def rxn(rid, name, stoich, lb=0.0, ub=1000.0, subsystem=""):
        r = Reaction(rid, name=name, lower_bound=lb, upper_bound=ub, subsystem=subsystem)
        r.add_metabolites({m[k]: v for k, v in stoich.items()})
        return r

    reactions = [
        # ---------------------------------------------------- exchanges
        rxn("EX_photon", "Light supply", {"photon": -1}, -100, 0, "Exchange"),
        rxn("EX_co2", "CO2 exchange", {"co2": -1}, -1000, 1000, "Exchange"),
        rxn("EX_o2", "O2 exchange", {"o2": -1}, -1000, 1000, "Exchange"),
        rxn("EX_h2o", "Water exchange", {"h2o": -1}, -1000, 1000, "Exchange"),
        rxn("EX_nh4", "Ammonium uptake", {"nh4": -1}, -10, 0, "Exchange"),
        rxn("EX_ac", "Acetate uptake", {"ac": -1}, 0, 0, "Exchange"),

        # ---------------------------------------------------- photosynthesis
        # Linear electron flow: 8 photons -> 1 O2, 2 NADPH, 3 ATP
        rxn("LEF", "Linear electron flow (PSII + PSI)",
            {"photon": -8, "h2o": -2, "nadp": -2, "adp": -3,
             "o2": 1, "nadph": 2, "atp": 3}, 0.0, 9.0, "Photosynthesis"),
        # (upper bound 9 = photosystem capacity, so growth saturates in high light)
        # Cyclic electron flow around PSI: extra ATP, no NADPH
        rxn("CEF", "Cyclic electron flow (PSI)",
            {"photon": -2, "adp": -1, "atp": 1}, subsystem="Photosynthesis"),
        # Calvin-Benson cycle (lumped): 3 CO2 -> 1 triose phosphate
        rxn("CBB", "Calvin-Benson cycle (RuBisCO)",
            {"co2": -3, "atp": -9, "nadph": -6,
             "g3p": 1, "adp": 9, "nadp": 6}, subsystem="Carbon fixation"),

        # ---------------------------------------------------- central carbon
        rxn("GLYC", "Glycolysis (G3P to pyruvate)",
            {"g3p": -1, "adp": -1, "nad": -1, "pyr": 1, "atp": 1, "nadh": 1},
            subsystem="Glycolysis"),
        rxn("GNG", "Gluconeogenesis (pyruvate to G3P)",
            {"pyr": -1, "atp": -2, "nadh": -1, "g3p": 1, "adp": 2, "nad": 1},
            subsystem="Gluconeogenesis"),
        rxn("PDH", "Pyruvate dehydrogenase",
            {"pyr": -1, "nad": -1, "accoa": 1, "co2": 1, "nadh": 1},
            subsystem="Central carbon"),
        rxn("ACS", "Acetate activation (acetyl-CoA synthetase)",
            {"ac": -1, "atp": -2, "accoa": 1, "adp": 2}, subsystem="Acetate assimilation"),
        rxn("TCA", "TCA cycle (lumped)",
            {"accoa": -1, "adp": -1, "nad": -4, "co2": 2, "atp": 1, "nadh": 4},
            subsystem="TCA cycle"),
        rxn("GLX", "Glyoxylate cycle (2 acetyl-CoA to OAA)",
            {"accoa": -2, "nad": -1, "oaa": 1, "nadh": 1}, subsystem="Acetate assimilation"),
        rxn("PEPCK", "PEP carboxykinase route (OAA to pyruvate)",
            {"oaa": -1, "atp": -1, "pyr": 1, "co2": 1, "adp": 1}, subsystem="Gluconeogenesis"),
        rxn("TH", "NADH to NADPH transhydrogenation",
            {"nadh": -1, "nadp": -1, "nad": 1, "nadph": 1}, subsystem="Redox"),

        # ---------------------------------------------------- respiration
        rxn("RESP", "Mitochondrial respiration (oxidative phosphorylation)",
            {"nadh": -1, "o2": -0.5, "adp": -2.5, "nad": 1, "atp": 2.5, "h2o": 1},
            subsystem="Respiration"),
        rxn("ATPM", "Non-growth ATP maintenance",
            {"atp": -1, "adp": 1}, 1.0, 1000, "Maintenance"),

        # ---------------------------------------------------- storage
        rxn("STARCH_SYN", "Starch synthesis",
            {"g3p": -2, "atp": -1, "starch": 1, "adp": 1}, subsystem="Storage"),
        rxn("STARCH_DEG", "Starch degradation",
            {"starch": -1, "g3p": 2}, subsystem="Storage"),
        rxn("FAS", "Fatty acid synthesis (C16)",
            {"accoa": -8, "atp": -7, "nadph": -14,
             "lipid": 1, "adp": 7, "nadp": 14}, subsystem="Lipid synthesis"),
        rxn("DM_starch", "Starch accumulation (storage sink)",
            {"starch": -1}, subsystem="Storage"),
        rxn("DM_lipid", "Lipid accumulation (storage sink)",
            {"lipid": -1}, subsystem="Storage"),

        # ---------------------------------------------------- nitrogen
        rxn("AAS", "Amino acid synthesis (lumped)",
            {"pyr": -1, "nh4": -1, "nadph": -1, "atp": -1,
             "aa": 1, "nadp": 1, "adp": 1}, subsystem="Nitrogen assimilation"),

        # ---------------------------------------------------- biomass
        # Per gDW: 6 mmol amino acid units, 4 mmol triose (carbohydrate),
        # 0.5 mmol C16 fatty acid, 40 mmol ATP for polymerization.
        rxn("BIOMASS", "Biomass formation",
            {"aa": -6, "g3p": -4, "lipid": -0.5, "atp": -40,
             "adp": 40, "biomass": 1}, subsystem="Biomass"),
        rxn("EX_biomass", "Biomass out", {"biomass": -1}, subsystem="Exchange"),
    ]
    model.add_reactions(reactions)
    model.objective = "BIOMASS"
    return model


def set_mode(model, mode, light=60.0, acetate=10.0):
    """Set the growth mode: 'autotrophic', 'mixotrophic' or 'heterotrophic'."""
    if mode == "autotrophic":
        model.reactions.EX_photon.lower_bound = -light
        model.reactions.EX_ac.lower_bound = 0
    elif mode == "mixotrophic":
        model.reactions.EX_photon.lower_bound = -light
        model.reactions.EX_ac.lower_bound = -acetate
    elif mode == "heterotrophic":
        model.reactions.EX_photon.lower_bound = 0
        model.reactions.EX_ac.lower_bound = -acetate
        model.reactions.CBB.upper_bound = 0   # Calvin cycle is light-activated
    else:
        raise ValueError(mode)
    if mode != "heterotrophic" and light == 0:
        model.reactions.CBB.upper_bound = 0   # no light, no Calvin cycle


if __name__ == "__main__":
    model = build_model()
    print(f"{model.id}: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites")
    for mode in ["autotrophic", "mixotrophic", "heterotrophic"]:
        with model:
            set_mode(model, mode)
            sol = model.optimize()
            print(f"{mode:<14} growth = {sol.objective_value:.4f} 1/h  "
                  f"O2 = {sol.fluxes['EX_o2']:+.2f}  CO2 = {sol.fluxes['EX_co2']:+.2f}")
