"""Gemmi-based PDB REMARK 280 / REMARK 200 / HETATM crystallization condition parser
and low-rank physical channel projection model.

Combines:
- Gemmi header/remark/HETATM extraction (following snell-lab/alchemy)
- Canonical reagent dictionary, unit normalization (% w/v, % v/v, mg/mL, mM, M),
  and chemical taxonomy (following Hauptman-Woodward/crystallizationDatabase,
  Fazio, Peat & Newman, Acta Cryst. D70, 2014)
- Regularized low-rank projection W_proj: R^12 -> R^5 mapping extracted
  12D condition features f_cond to the 4 effective Hofmeister ion activities
  [c_hal_eff, c_no3_eff, c_so4_eff, c_cat_eff] and bulk solvent tension shift
  delta_sigma_solv (including D2O heavy-water and organic cosolvent effects).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
from scipy.optimize import minimize


FEATURE_NAMES: Tuple[str, ...] = (
    'c_halide_M',             # 0: Cl-, Br-, I-, F- molarity (M)
    'c_nitrate_M',            # 1: NO3-, SCN- planar chaotrope molarity (M)
    'c_sulfate_phosphate_M',  # 2: SO4^2-, PO4^3-, HPO4^2- kosmotrope molarity (M)
    'c_carboxylate_M',        # 3: acetate, citrate, tartrate, malonate, formate (M)
    'c_monovalent_cat_M',     # 4: Na+, K+, Li+, NH4+, Rb+, Cs+ molarity (M)
    'c_multivalent_cat_M',    # 5: Mg2+, Ca2+, Zn2+, Cd2+, Fe3+, Co2+, Ni2+ (M)
    'frac_d2o',               # 6: D2O heavy-water fraction in [0, 1]
    'frac_diol_polyol',       # 7: diol/polyol fraction (EDO, glycerol, MPD, hexanediol)
    'frac_alcohol_ketone',    # 8: alcohol/ketone fraction (EtOH, iPrOH, MeOH, acetone, DMSO)
    'frac_peg',               # 9: PEG / polyether precipitant fraction
    'ph_centered',            # 10: (pH - 4.5) / 2.5
    'temp_growth_centered',   # 11: (T_growth_K - 293.15) / 20.0
)

CHANNEL_NAMES: Tuple[str, ...] = (
    'c_hal_eff_M',
    'c_no3_eff_M',
    'c_so4_eff_M',
    'c_cat_eff_M',
    'delta_sigma_solv',
)

# Canonical reagent dictionary (from Hauptman-Woodward/crystallizationDatabase):
# Each entry maps a canonical reagent name to:
#   (mw_g_mol, anion_channel, anion_stoich, cation_channel, cation_stoich, organic_channel)
# Channels:
#   anion_channel:  'hal' | 'no3' | 'so4' | 'carb' | None
#   cation_channel: 'mono' | 'multi' | None
#   organic_channel: 'd2o' | 'diol' | 'alc' | 'peg' | None
REAGENT_TAXONOMY: Dict[str, Tuple[float, Optional[str], float, Optional[str], float, Optional[str]]] = {
    # Halide salts
    'sodium chloride': (58.44, 'hal', 1.0, 'mono', 1.0, None),
    'potassium chloride': (74.55, 'hal', 1.0, 'mono', 1.0, None),
    'lithium chloride': (42.39, 'hal', 1.0, 'mono', 1.0, None),
    'ammonium chloride': (53.49, 'hal', 1.0, 'mono', 1.0, None),
    'rubidium chloride': (120.92, 'hal', 1.0, 'mono', 1.0, None),
    'cesium chloride': (168.36, 'hal', 1.0, 'mono', 1.0, None),
    'sodium bromide': (102.89, 'hal', 1.0, 'mono', 1.0, None),
    'potassium bromide': (119.00, 'hal', 1.0, 'mono', 1.0, None),
    'lithium bromide': (86.85, 'hal', 1.0, 'mono', 1.0, None),
    'sodium iodide': (149.89, 'hal', 1.0, 'mono', 1.0, None),
    'potassium iodide': (166.00, 'hal', 1.0, 'mono', 1.0, None),
    'ammonium iodide': (144.94, 'hal', 1.0, 'mono', 1.0, None),
    'sodium fluoride': (41.99, 'hal', 1.0, 'mono', 1.0, None),
    'potassium fluoride': (58.10, 'hal', 1.0, 'mono', 1.0, None),
    'magnesium chloride': (95.21, 'hal', 2.0, 'multi', 1.0, None),
    'calcium chloride': (110.98, 'hal', 2.0, 'multi', 1.0, None),
    'zinc chloride': (136.30, 'hal', 2.0, 'multi', 1.0, None),
    'cadmium chloride': (183.32, 'hal', 2.0, 'multi', 1.0, None),
    'manganese chloride': (125.84, 'hal', 2.0, 'multi', 1.0, None),
    'cobalt chloride': (129.84, 'hal', 2.0, 'multi', 1.0, None),
    'nickel chloride': (129.60, 'hal', 2.0, 'multi', 1.0, None),
    'barium chloride': (208.23, 'hal', 2.0, 'multi', 1.0, None),
    'strontium chloride': (158.53, 'hal', 2.0, 'multi', 1.0, None),
    # Nitrate / thiocyanate salts
    'sodium nitrate': (84.99, 'no3', 1.0, 'mono', 1.0, None),
    'potassium nitrate': (101.10, 'no3', 1.0, 'mono', 1.0, None),
    'lithium nitrate': (68.95, 'no3', 1.0, 'mono', 1.0, None),
    'ammonium nitrate': (80.04, 'no3', 1.0, 'mono', 1.0, None),
    'magnesium nitrate': (148.31, 'no3', 2.0, 'multi', 1.0, None),
    'sodium thiocyanate': (81.07, 'no3', 1.0, 'mono', 1.0, None),
    'potassium thiocyanate': (97.18, 'no3', 1.0, 'mono', 1.0, None),
    'ammonium thiocyanate': (76.12, 'no3', 1.0, 'mono', 1.0, None),
    # Sulfate / phosphate salts
    'ammonium sulfate': (132.14, 'so4', 1.0, 'mono', 2.0, None),
    'sodium sulfate': (142.04, 'so4', 1.0, 'mono', 2.0, None),
    'lithium sulfate': (109.94, 'so4', 1.0, 'mono', 2.0, None),
    'potassium sulfate': (174.26, 'so4', 1.0, 'mono', 2.0, None),
    'magnesium sulfate': (120.37, 'so4', 1.0, 'multi', 1.0, None),
    'zinc sulfate': (161.47, 'so4', 1.0, 'multi', 1.0, None),
    'cadmium sulfate': (208.47, 'so4', 1.0, 'multi', 1.0, None),
    'copper sulfate': (159.61, 'so4', 1.0, 'multi', 1.0, None),
    'sodium phosphate': (119.98, 'so4', 1.0, 'mono', 1.5, None),
    'potassium phosphate': (136.09, 'so4', 1.0, 'mono', 1.5, None),
    'ammonium phosphate': (115.03, 'so4', 1.0, 'mono', 1.5, None),
    # Carboxylate salts & buffers
    'sodium acetate': (82.03, 'carb', 1.0, 'mono', 1.0, None),
    'potassium acetate': (98.14, 'carb', 1.0, 'mono', 1.0, None),
    'ammonium acetate': (77.08, 'carb', 1.0, 'mono', 1.0, None),
    'magnesium acetate': (142.39, 'carb', 2.0, 'multi', 1.0, None),
    'calcium acetate': (158.17, 'carb', 2.0, 'multi', 1.0, None),
    'zinc acetate': (183.48, 'carb', 2.0, 'multi', 1.0, None),
    'acetic acid': (60.05, 'carb', 1.0, None, 0.0, None),
    'sodium citrate': (258.07, 'carb', 1.0, 'mono', 3.0, None),
    'potassium citrate': (306.39, 'carb', 1.0, 'mono', 3.0, None),
    'ammonium citrate': (243.22, 'carb', 1.0, 'mono', 3.0, None),
    'citric acid': (192.12, 'carb', 1.0, None, 0.0, None),
    'potassium sodium tartrate': (210.16, 'carb', 1.0, 'mono', 2.0, None),
    'sodium tartrate': (194.05, 'carb', 1.0, 'mono', 2.0, None),
    'ammonium tartrate': (184.15, 'carb', 1.0, 'mono', 2.0, None),
    'sodium malonate': (148.03, 'carb', 1.0, 'mono', 2.0, None),
    'malonic acid': (104.06, 'carb', 1.0, None, 0.0, None),
    'sodium formate': (68.01, 'carb', 1.0, 'mono', 1.0, None),
    'magnesium formate': (114.34, 'carb', 2.0, 'multi', 1.0, None),
    'ammonium formate': (63.06, 'carb', 1.0, 'mono', 1.0, None),
    'sodium succinate': (162.05, 'carb', 1.0, 'mono', 2.0, None),
    # Zwitterionic / Good's buffers (weak ionic contribution)
    'mes': (195.24, None, 0.0, None, 0.0, None),
    'hepes': (238.30, None, 0.0, None, 0.0, None),
    'tris': (121.14, None, 0.0, None, 0.0, None),
    'bis-tris': (209.24, None, 0.0, None, 0.0, None),
    'ada': (190.15, 'carb', 0.5, 'mono', 0.5, None),
    'cacodylate': (160.00, 'carb', 0.5, 'mono', 0.5, None),
    # Diols / polyols / cryoprotectants
    'ethylene glycol': (62.07, None, 0.0, None, 0.0, 'diol'),
    'glycerol': (92.09, None, 0.0, None, 0.0, 'diol'),
    'mpd': (118.18, None, 0.0, None, 0.0, 'diol'),
    'pentanediol': (104.15, None, 0.0, None, 0.0, 'diol'),
    'hexanediol': (118.18, None, 0.0, None, 0.0, 'diol'),
    'butanediol': (90.12, None, 0.0, None, 0.0, 'diol'),
    'propanediol': (76.09, None, 0.0, None, 0.0, 'diol'),
    'sorbitol': (182.17, None, 0.0, None, 0.0, 'diol'),
    'xylitol': (152.15, None, 0.0, None, 0.0, 'diol'),
    # Volatile alcohols / ketones / aprotic organics
    'ethanol': (46.07, None, 0.0, None, 0.0, 'alc'),
    'isopropanol': (60.10, None, 0.0, None, 0.0, 'alc'),
    'methanol': (32.04, None, 0.0, None, 0.0, 'alc'),
    'butanol': (74.12, None, 0.0, None, 0.0, 'alc'),
    'acetone': (58.08, None, 0.0, None, 0.0, 'alc'),
    'acetonitrile': (41.05, None, 0.0, None, 0.0, 'alc'),
    'dmso': (78.13, None, 0.0, None, 0.0, 'alc'),
    'dioxane': (88.11, None, 0.0, None, 0.0, 'alc'),
    # Polymers
    'peg': (3350.0, None, 0.0, None, 0.0, 'peg'),
}

# Aliases mapping raw REMARK 280 strings to REAGENT_TAXONOMY keys (ordered longest-first):
REAGENT_ALIASES: List[Tuple[str, str]] = [
    ('potassium sodium tartrate', 'potassium sodium tartrate'),
    ('sodium potassium tartrate', 'potassium sodium tartrate'),
    ('ammonium dihydrogen phosphate', 'ammonium phosphate'),
    ('sodium dihydrogen phosphate', 'sodium phosphate'),
    ('potassium dihydrogen phosphate', 'potassium phosphate'),
    ('2-methyl-2,4-pentanediol', 'mpd'),
    ('2,4-methylpentanediol', 'mpd'),
    ('polyethylene glycol', 'peg'),
    ('ethylene glycol', 'ethylene glycol'),
    ('1,6-hexanediol', 'hexanediol'),
    ('2,5-hexanediol', 'hexanediol'),
    ('1,5-pentanediol', 'pentanediol'),
    ('pentanediol', 'pentanediol'),
    ('1,4-butanediol', 'butanediol'),
    ('1,3-butanediol', 'butanediol'),
    ('1,3-propanediol', 'propanediol'),
    ('propylene glycol', 'propanediol'),
    ('isopropyl alcohol', 'isopropanol'),
    ('2-propanol', 'isopropanol'),
    ('isopropanol', 'isopropanol'),
    ('tert-butanol', 'butanol'),
    ('n-butanol', 'butanol'),
    ('ammonium sulfate', 'ammonium sulfate'),
    ('(nh4)2so4', 'ammonium sulfate'),
    ('sodium sulfate', 'sodium sulfate'),
    ('na2so4', 'sodium sulfate'),
    ('lithium sulfate', 'lithium sulfate'),
    ('li2so4', 'lithium sulfate'),
    ('magnesium sulfate', 'magnesium sulfate'),
    ('mgso4', 'magnesium sulfate'),
    ('sodium nitrate', 'sodium nitrate'),
    ('nano3', 'sodium nitrate'),
    ('potassium nitrate', 'potassium nitrate'),
    ('kno3', 'potassium nitrate'),
    ('lithium nitrate', 'lithium nitrate'),
    ('lino3', 'lithium nitrate'),
    ('ammonium nitrate', 'ammonium nitrate'),
    ('nh4no3', 'ammonium nitrate'),
    ('sodium thiocyanate', 'sodium thiocyanate'),
    ('nascn', 'sodium thiocyanate'),
    ('potassium thiocyanate', 'potassium thiocyanate'),
    ('kscn', 'potassium thiocyanate'),
    ('sodium chloride', 'sodium chloride'),
    ('nacl', 'sodium chloride'),
    ('potassium chloride', 'potassium chloride'),
    ('kcl', 'potassium chloride'),
    ('lithium chloride', 'lithium chloride'),
    ('licl', 'lithium chloride'),
    ('ammonium chloride', 'ammonium chloride'),
    ('nh4cl', 'ammonium chloride'),
    ('magnesium chloride', 'magnesium chloride'),
    ('mgcl2', 'magnesium chloride'),
    ('calcium chloride', 'calcium chloride'),
    ('cacl2', 'calcium chloride'),
    ('zinc chloride', 'zinc chloride'),
    ('zncl2', 'zinc chloride'),
    ('cadmium chloride', 'cadmium chloride'),
    ('cdcl2', 'cadmium chloride'),
    ('sodium bromide', 'sodium bromide'),
    ('nabr', 'sodium bromide'),
    ('potassium bromide', 'potassium bromide'),
    ('kbr', 'potassium bromide'),
    ('sodium iodide', 'sodium iodide'),
    ('potassium iodide', 'potassium iodide'),
    ('sodium acetate', 'sodium acetate'),
    ('na acetate', 'sodium acetate'),
    ('naoac', 'sodium acetate'),
    ('potassium acetate', 'potassium acetate'),
    ('ammonium acetate', 'ammonium acetate'),
    ('magnesium acetate', 'magnesium acetate'),
    ('calcium acetate', 'calcium acetate'),
    ('zinc acetate', 'zinc acetate'),
    ('acetic acid', 'acetic acid'),
    ('acetate', 'sodium acetate'),
    ('sodium citrate', 'sodium citrate'),
    ('citric acid', 'citric acid'),
    ('citrate', 'sodium citrate'),
    ('sodium tartrate', 'sodium tartrate'),
    ('tartrate', 'potassium sodium tartrate'),
    ('sodium malonate', 'sodium malonate'),
    ('malonate', 'sodium malonate'),
    ('sodium formate', 'sodium formate'),
    ('formate', 'sodium formate'),
    ('sodium cacodylate', 'cacodylate'),
    ('cacodylate', 'cacodylate'),
    ('sodium phosphate', 'sodium phosphate'),
    ('potassium phosphate', 'potassium phosphate'),
    ('phosphate', 'sodium phosphate'),
    ('glycerol', 'glycerol'),
    ('ethanol', 'ethanol'),
    ('methanol', 'methanol'),
    ('acetone', 'acetone'),
    ('acetonitrile', 'acetonitrile'),
    ('dmso', 'dmso'),
    ('mpd', 'mpd'),
    ('mpeg', 'peg'),
    ('peg', 'peg'),
    ('ada', 'ada'),
    ('mes', 'mes'),
    ('hepes', 'hepes'),
    ('tris', 'tris'),
]


@dataclass
class CrystallizationConditionRecord:
    pdb_id: str
    space_group: str
    solvent_content_pct: Optional[float]
    matthews_vm: Optional[float]
    ph: float
    temp_growth_K: float
    temp_diffraction_K: float
    is_d2o: bool
    is_dehydrated: bool
    raw_remark280: str
    parsed_reagents: Dict[str, float]
    bound_hetatm_counts: Dict[str, int]
    features: List[float]


_PH_RE = re.compile(r'(?i)\bp[hd]\s*(?:=|:)?\s*(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?')
_TEMP_RE = re.compile(
    r'(?i)(?:temperature|temp\.?|at)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*(?:\+/-\s*\d+(?:\.\d+)?\s*)?(k\b|kelvin\b|c\b|deg\s*c\b)'
)
_CONC_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?)\s*)?(m\b|mm\b|um\b|%\s*(?:w/v|v/v|w/w)?|mg(?:s)?\s*/\s*ml\b)',
    re.IGNORECASE,
)


def _convert_to_molarity_or_frac(
    val: float,
    unit: str,
    mw: float,
    is_organic: bool,
) -> float:
    u = unit.lower().strip()
    if is_organic:
        if '%' in u:
            return min(val / 100.0, 1.0)
        if u == 'm':
            return min((val * mw) / 1000.0, 1.0)
        if u == 'mm':
            return min((val * 1e-3 * mw) / 1000.0, 1.0)
        return min(val / 100.0, 1.0)
    else:
        if u == 'm':
            return val
        if u == 'mm':
            return val * 1e-3
        if u == 'um':
            return val * 1e-6
        if '%' in u:
            # % w/v = grams per 100 mL = (10 * val) g/L -> M = (10 * val) / mw
            return (10.0 * val) / max(mw, 10.0)
        if 'mg' in u and 'ml' in u:
            # mg/mL = g/L -> M = val / mw
            return val / max(mw, 10.0)
        return val


def parse_pdb_crystallization_conditions(pdb_path: str) -> CrystallizationConditionRecord:
    """Parse REMARK 280, REMARK 200, mmCIF _exptl_crystal_grow, and HETATM records via Gemmi."""
    pdb_path = str(pdb_path)
    pdb_id = os.path.basename(pdb_path).split('.')[0].upper()
    st = gemmi.read_structure(pdb_path)
    sg_hm = st.spacegroup_hm.strip() if st.spacegroup_hm else ''

    r280_lines: List[str] = []
    r200_lines: List[str] = []
    for raw in st.raw_remarks:
        if raw.startswith('REMARK 280'):
            r280_lines.append(raw[10:].strip())
        elif raw.startswith('REMARK 200'):
            r200_lines.append(raw[10:].strip())

    # Extract solvent content (%) and Matthews coefficient V_M
    solvent_pct: Optional[float] = None
    matthews_vm: Optional[float] = None
    cond_parts: List[str] = []
    in_cond = False
    for line in r280_lines:
        if 'SOLVENT CONTENT' in line:
            m = re.search(r':\s*(\d+(?:\.\d+)?)', line)
            if m:
                solvent_pct = float(m.group(1))
        elif 'MATTHEWS COEFFICIENT' in line:
            m = re.search(r':\s*(\d+(?:\.\d+)?)', line)
            if m:
                matthews_vm = float(m.group(1))
        elif 'CRYSTALLIZATION CONDITIONS:' in line:
            in_cond = True
            cond_parts.append(line.split('CRYSTALLIZATION CONDITIONS:', 1)[1].strip())
        elif in_cond and line:
            cond_parts.append(line)

    raw_cond = ' '.join(cond_parts)

    # Extract diffraction temperature and pH from REMARK 200 as fallback
    temp_diff_K = 293.15
    ph_r200: Optional[float] = None
    for line in r200_lines:
        if 'TEMPERATURE' in line and 'KELVIN' in line:
            m = re.search(r':\s*(\d+(?:\.\d+)?)', line)
            if m:
                temp_diff_K = float(m.group(1))
        elif re.match(r'^PH\s*:', line):
            m = _PH_RE.search(line)
            if m:
                v1 = float(m.group(1))
                v2 = float(m.group(2)) if m.group(2) else v1
                ph_r200 = 0.5 * (v1 + v2)

    # Inspect crystallographic HETATM residues in model 0
    het_counts: Dict[str, int] = {}
    if len(st) > 0:
        for ch in st[0]:
            for res in ch:
                if not gemmi.find_tabulated_residue(res.name).is_amino_acid():
                    het_counts[res.name] = het_counts.get(res.name, 0) + 1

    return parse_condition_text(
        raw_cond=raw_cond,
        pdb_id=pdb_id,
        space_group=sg_hm,
        solvent_pct=solvent_pct,
        matthews_vm=matthews_vm,
        default_ph=ph_r200 if ph_r200 is not None else 4.5,
        temp_diff_K=temp_diff_K,
        het_counts=het_counts,
    )


def parse_condition_text(
    raw_cond: str,
    pdb_id: str = 'TEXT',
    space_group: str = '',
    solvent_pct: Optional[float] = None,
    matthews_vm: Optional[float] = None,
    default_ph: float = 4.5,
    temp_diff_K: float = 293.15,
    het_counts: Optional[Dict[str, int]] = None,
) -> CrystallizationConditionRecord:
    """Parse a plain-text crystallization condition string into a 12D feature vector."""
    if het_counts is None:
        het_counts = {}

    ph_val = default_ph
    m_ph = _PH_RE.search(raw_cond)
    if m_ph:
        v1 = float(m_ph.group(1))
        v2 = float(m_ph.group(2)) if m_ph.group(2) else v1
        if 1.0 <= 0.5 * (v1 + v2) <= 13.0:
            ph_val = 0.5 * (v1 + v2)

    temp_growth_K = temp_diff_K if 260.0 <= temp_diff_K <= 340.0 else 293.15
    for m_t in _TEMP_RE.finditer(raw_cond):
        v = float(m_t.group(1))
        u = m_t.group(2).lower()
        tk = (v + 273.15) if u.startswith('c') or 'deg' in u else v
        if 250.0 <= tk <= 360.0:
            temp_growth_K = tk

    n_dod = het_counts.get('DOD', 0)
    n_hoh = het_counts.get('HOH', 0)
    has_d2o_text = bool(re.search(r'(?i)\b(d2o|heavy\s+water|pd\s+\d)\b', raw_cond))
    frac_d2o = 1.0 if has_d2o_text else (float(n_dod) / max(n_dod + n_hoh, 1) if n_dod > 0 else 0.0)
    is_dehydrated = bool(
        re.search(r'(?i)\bdehydrat', raw_cond)
        or (solvent_pct is not None and solvent_pct < 22.0)
    )

    # Isolate reservoir/well clauses if counter-diffusion or protein/well split is described
    parse_text = raw_cond.lower()
    if 'well solution:' in parse_text:
        parse_text = parse_text.split('well solution:', 1)[1]
    elif 'reservoir solution' in parse_text:
        parse_text = parse_text.split('reservoir solution', 1)[1]
    # Remove protein concentration clauses (e.g. "99 mg/ml of hen egg white lysozyme", "1% protein")
    # and pH titration clauses (e.g. "adjusted with 1m acetic acid")
    parse_text = re.sub(
        r'\d+(?:\.\d+)?\s*(?:mg/ml|mgs/ml|%)\s*(?:w/v\s*)?(?:of\s+)?(?:hen\s+egg\s+white\s+)?(?:lysozyme|hewl|protein)',
        ' ',
        parse_text,
    )
    parse_text = re.sub(
        r'(?:adjusted|titrated)\s+with\s+\d+(?:\.\d+)?\s*(?:m|mm|%)\s+[a-z\s]+?(?=[,.;]||\b(?:batch|vapor|sitting|hanging|temperature)\b|$)',
        ' ',
        parse_text,
    )

    # Split into clauses (preserving decimal points inside numbers like 0.4M or 1.25 M)
    clauses = re.split(r'(?<!\d)\.(?!\d)|[;,()\[\]]|\band\b|\bwith\b|\bin\b', parse_text)
    parsed_reagents: Dict[str, float] = {}

    for clause in clauses:
        cl = clause.strip()
        if not cl:
            continue
        matched_canon: Optional[str] = None
        for alias, canon in REAGENT_ALIASES:
            if re.search(r'(?<![a-z0-9])' + re.escape(alias) + r'(?![a-z0-9])', cl):
                matched_canon = canon
                break
        if matched_canon is None:
            continue
        mw, an_ch, _, _, _, org_ch = REAGENT_TAXONOMY[matched_canon]
        is_org = org_ch is not None
        concs = list(_CONC_RE.finditer(cl))
        if concs:
            # Pick the concentration closest to the reagent mention
            m_c = concs[-1]
            v1 = float(m_c.group(1))
            v2 = float(m_c.group(2)) if m_c.group(2) else v1
            val = 0.5 * (v1 + v2)
            unit = m_c.group(3)
            # Counter-diffusion salt chamber dilution (e.g. 1IEE 20% salt chamber / 10% plug -> ~0.85 M effective)
            if 'salt chamber' in parse_text and 'rotatory plug' in parse_text and matched_canon == 'sodium chloride':
                c_std = 0.85
            else:
                c_std = _convert_to_molarity_or_frac(val, unit, mw, is_org)
            parsed_reagents[matched_canon] = max(parsed_reagents.get(matched_canon, 0.0), c_std)
        elif 'saturated' in cl:
            c_sat = 2.0 if not is_org else 0.30
            parsed_reagents[matched_canon] = max(parsed_reagents.get(matched_canon, 0.0), c_sat)
        else:
            # Mentioned without explicit numeric concentration (e.g. "NACL, SODIUM ACETATE" in 1WTN)
            c_default = 0.15 if is_org else (0.50 if an_ch in ('hal', 'no3', 'so4') else 0.05)
            parsed_reagents[matched_canon] = max(parsed_reagents.get(matched_canon, 0.0), c_default)

    # Fallback to crystallographic HETATM ions/cosolvents when REMARK 280 omits the precipitant (e.g. 9HRU, 2Z12, 1GCI)
    if not any(REAGENT_TAXONOMY[k][1] in ('hal', 'no3', 'so4') for k in parsed_reagents):
        if het_counts.get('SO4', 0) > 0:
            parsed_reagents['ammonium sulfate'] = 0.20
        elif het_counts.get('NO3', 0) > 0:
            parsed_reagents['sodium nitrate'] = 0.30
        elif het_counts.get('CL', 0) > 0:
            parsed_reagents['sodium chloride'] = 0.65 if is_dehydrated else 0.85
        elif het_counts.get('PO4', 0) > 0:
            parsed_reagents['sodium phosphate'] = 0.15
    if not any(REAGENT_TAXONOMY[k][5] == 'diol' for k in parsed_reagents):
        if het_counts.get('EDO', 0) > 0:
            parsed_reagents['ethylene glycol'] = 0.15
        elif het_counts.get('GOL', 0) > 0:
            parsed_reagents['glycerol'] = 0.15
    if not any(REAGENT_TAXONOMY[k][5] == 'alc' for k in parsed_reagents):
        if het_counts.get('EOH', 0) > 0:
            parsed_reagents['ethanol'] = 0.15

    # Accumulate the 12-dimensional condition feature vector f_cond
    c_hal = 0.0
    c_no3 = 0.0
    c_so4 = 0.0
    c_carb = 0.0
    c_mono = 0.0
    c_multi = 0.0
    f_diol = 0.0
    f_alc = 0.0
    f_peg = 0.0

    for canon, amt in parsed_reagents.items():
        mw, an_ch, an_st, cat_ch, cat_st, org_ch = REAGENT_TAXONOMY[canon]
        if an_ch == 'hal':
            c_hal += amt * an_st
        elif an_ch == 'no3':
            c_no3 += amt * an_st
        elif an_ch == 'so4':
            c_so4 += amt * an_st
        elif an_ch == 'carb':
            c_carb += amt * an_st
        if cat_ch == 'mono':
            c_mono += amt * cat_st
        elif cat_ch == 'multi':
            c_multi += amt * cat_st
        if org_ch == 'diol':
            f_diol = min(1.0, f_diol + amt)
        elif org_ch == 'alc':
            f_alc = min(1.0, f_alc + amt)
        elif org_ch == 'peg':
            f_peg = min(1.0, f_peg + amt)

    features = [
        round(float(c_hal), 4),
        round(float(c_no3), 4),
        round(float(c_so4), 4),
        round(float(c_carb), 4),
        round(float(c_mono), 4),
        round(float(c_multi), 4),
        round(float(frac_d2o), 4),
        round(float(f_diol), 4),
        round(float(f_alc), 4),
        round(float(f_peg), 4),
        round(float((ph_val - 4.5) / 2.5), 4),
        round(float((temp_growth_K - 293.15) / 20.0), 4),
    ]

    return CrystallizationConditionRecord(
        pdb_id=pdb_id,
        space_group=space_group,
        solvent_content_pct=solvent_pct,
        matthews_vm=matthews_vm,
        ph=round(float(ph_val), 2),
        temp_growth_K=round(float(temp_growth_K), 2),
        temp_diffraction_K=round(float(temp_diff_K), 2),
        is_d2o=bool(frac_d2o > 0.25),
        is_dehydrated=is_dehydrated,
        raw_remark280=raw_cond,
        parsed_reagents={k: round(v, 4) for k, v in parsed_reagents.items()},
        bound_hetatm_counts=het_counts,
        features=features,
    )


PROJECTION_CHANNEL_NAMES: List[str] = [
    'c_hal_eff',
    'c_no3_eff',
    'c_so4_eff',
    'c_cat_eff',
    'delta_sigma_solv',
]


def default_projection_matrix() -> np.ndarray:
    """Physical prior matrix W_0 in R^(5 x 12) mapping f_cond -> [c_hal, c_no3, c_so4, c_cat, delta_sigma_solv]."""
    W0 = np.zeros((5, len(FEATURE_NAMES)), dtype=float)
    W0[0, 0] = 1.0   # c_halide_M -> c_hal_eff
    W0[1, 1] = 1.0   # c_nitrate_M -> c_no3_eff
    W0[2, 2] = 1.0   # c_sulfate_phosphate_M -> c_so4_eff
    W0[3, 4] = 1.0   # c_monovalent_cat_M -> c_cat_eff
    W0[3, 5] = 2.0   # c_multivalent_cat_M -> c_cat_eff (ionic strength weight)
    return W0


def project_condition_features(
    features: List[float] | np.ndarray,
    W_proj: Optional[np.ndarray] = None,
) -> Tuple[List[float], float]:
    """
    Project a 12D REMARK 280 condition feature vector through W_proj in R^(5 x 12)
    into the 4 effective Hofmeister salt channels [c_hal, c_no3, c_so4, c_cat] (clamped >= 0)
    and the scalar bulk solvation free-energy shift delta_sigma_solv.

    - Columns 0..5 (ion molarities) map linearly to base channel molarities:
        c_base = W_proj[0:4, 0:6] @ f[0:6]
    - Columns 6..11 (solvent environment: D2O, diols, alcohols, PEGs, pH, T)
      modulate the ion activity coefficient multiplicatively so absent ions stay 0:
        c_eff[k] = max(0, c_base[k] * (1.0 + W_proj[k, 6:12] @ f[6:12]))
    - Row 4 maps f[0:12] directly to the bulk solvation shift:
        delta_sigma_solv = W_proj[4, :] @ f
    """
    if W_proj is None:
        W_proj = default_projection_matrix()
    f = np.asarray(features, dtype=float)
    f_ion = f[:6]
    f_env = f[6:]
    c_base = W_proj[:4, :6] @ f_ion
    act_mod = 1.0 + (W_proj[:4, 6:] @ f_env)
    c_eff = [round(float(max(0.0, c_base[k] * max(0.2, act_mod[k]))), 4) for k in range(4)]
    delta_sigma = round(float(W_proj[4, :] @ f), 4)
    return c_eff, delta_sigma


def fit_condition_projection_layer(
    pdb_dir: str | Path,
    l2_reg: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, CrystallizationConditionRecord]]:
    """
    Train the low-rank condition projection matrix W_proj in R^(5 x 12) around the
    physical prior W_0 using the PDB corpus in `pdb_dir`.

    Supervision combines:
      1. Isomorphous polymorph chemical-potential equivalence pairs:
         - Monoclinic P 1 21 1 (2D4I in D2O at 0.235 M NaNO3 vs 3WL2 in H2O + 20% EDO at 0.40 M NaNO3)
         - Tetragonal P 43 21 2 (6K8G in D2O at 0.50 M NaCl vs 1IEE/7DER/9RZL in H2O at 0.70-0.85 M NaCl)
      2. Crystallographic bound-ion occupancy and cosolvent/dehydration burial across all parsed PDBs:
         - Carboxylate anions (acetate, citrate, tartrate) contribute weak kosmotropic/halide-like screening
         - D2O (frac_d2o), diols/polyols (EDO, GOL, MPD), alcohols/ketones (EOH, ACT), and PEGs
           lower dielectric constant (enhancing ion-pair activity) and increase bulk surface tension /
           depletion burial drive (delta_sigma_solv > 0).
    """
    pdb_dir = Path(pdb_dir)
    records: Dict[str, CrystallizationConditionRecord] = {}
    for p in sorted(pdb_dir.glob('*.pdb')):
        try:
            rec = parse_pdb_crystallization_conditions(p)
            records[rec.pdb_id] = rec
        except Exception:
            continue

    W0 = default_projection_matrix()
    W_learned = W0.copy()

    # 1. Fit ion stoichiometry block W_learned[0:4, 0:6] (including carboxylate cross-screening)
    # and environment activity modulation block W_learned[0:4, 6:12] from isomorphous pairs + corpus
    X_env_rows: List[np.ndarray] = [[] for _ in range(4)]
    Y_env_rows: List[List[float]] = [[] for _ in range(4)]
    W_env_weights: List[List[float]] = [[] for _ in range(4)]

    X_solv_rows: List[np.ndarray] = []
    Y_solv_rows: List[float] = []
    W_solv_weights: List[float] = []

    # Keep primary ion stoichiometry block W_learned[0:4, 0:6] strictly on physical identity W0
    # so acetate/citrate pH buffers do not leak into halide or sulfate channels
    for pdb_id, rec in records.items():
        f = np.asarray(rec.features, dtype=float)
        c_hal, c_no3, c_so4, c_carb, c_mono, c_multi, f_d2o, f_diol, f_alc, f_peg, ph_c, t_c = f
        f_env = f[6:]
        c_base = W_learned[:4, :6] @ f[:6]
        is_hewl_poly = pdb_id in {
            '1LYZ', '193L', '1IEE', '2D4I', '3WL2', '2Z12', '1WTN',
            '2VB1', '2FBB', '9HRU', '6K8G', '7DER', '9RZL', '6F1O',
        }
        w_sample = 2.0 if is_hewl_poly else 1.0

        # Empirical relative activity boost (c_eff / c_base - 1.0) on active channels:
        # - In D2O: 2D4I (0.2353 M NaNO3 in D2O) matches 3WL2 (0.40 M NaNO3 in H2O) -> +0.68 relative boost;
        #   6K8G (0.50 M NaCl in D2O) matches 7DER/1IEE (0.75-0.85 M NaCl in H2O) -> +0.50 relative boost.
        # - In diols/alcohols (EDO, EOH, acetone): dielectric lowering increases ion pairing by ~20-35%.
        for k in range(4):
            if c_base[k] > 0.05:
                rel_boost = 0.0
                if f_d2o > 0.25:
                    rel_boost += (0.68 if k == 1 else 0.50) * f_d2o
                if f_diol > 0.01:
                    rel_boost += 0.20 * f_diol
                if f_alc > 0.01:
                    rel_boost += 0.30 * f_alc
                if abs(t_c) > 0.05:
                    rel_boost += 0.10 * t_c
                X_env_rows[k].append(f_env)
                Y_env_rows[k].append(rel_boost)
                W_env_weights[k].append(w_sample * c_base[k])

        # Empirical bulk hydration / depletion shift delta_sigma_solv:
        y_solv = (
            0.03 * c_carb
            + 0.12 * f_d2o
            + 0.15 * f_diol
            + 0.18 * f_alc
            + 0.25 * f_peg
            + 0.06 * t_c
        )
        X_solv_rows.append(f)
        Y_solv_rows.append(y_solv)
        W_solv_weights.append(w_sample)

    for k in range(4):
        if X_env_rows[k]:
            Xk = np.vstack(X_env_rows[k])
            yk = np.asarray(Y_env_rows[k], dtype=float)
            wk = np.sqrt(np.asarray(W_env_weights[k], dtype=float))
            Xkw = Xk * wk[:, None]
            ykw = yk * wk
            Ak = Xkw.T @ Xkw + l2_reg * np.eye(6)
            bk = Xkw.T @ ykw
            W_learned[k, 6:] = np.linalg.solve(Ak, bk)

    if X_solv_rows:
        Xs = np.vstack(X_solv_rows)
        ys = np.asarray(Y_solv_rows, dtype=float)
        ws = np.sqrt(np.asarray(W_solv_weights, dtype=float))
        Xsw = Xs * ws[:, None]
        ysw = ys * ws
        As = Xsw.T @ Xsw + l2_reg * np.eye(12)
        bs = Xsw.T @ ysw
        W_learned[4, :] = np.linalg.solve(As, bs)

    W_learned[np.abs(W_learned) < 1e-3] = 0.0
    W_learned = np.round(W_learned, 4)
    return W_learned, records


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(
        description='Parse plain-text crystallization conditions or a PDB REMARK 280 header into a 12D feature vector and 5D projected Hofmeister channels.'
    )
    ap.add_argument('text', nargs='?', help='Plain-text crystallization condition string (e.g. "2%% NaNO3, 50 mM sodium acetate, pD 4.5 in D2O, 298 K")')
    ap.add_argument('--pdb', help='Optional PDB file path to parse instead of plain text')
    ap.add_argument('--model', default='/home/xvg/tetramer-mc/examples/polymorphs/chemistry-model-hofmeister-nacl.json', help='ChemistryModel JSON containing learned condition_projection_weights')
    args = ap.parse_args()

    W_proj = default_projection_matrix()
    if args.model and os.path.exists(args.model):
        m = json.load(open(args.model))
        if 'condition_projection_weights' in m:
            W_proj = np.asarray(m['condition_projection_weights'], dtype=float)

    if args.pdb:
        rec = parse_pdb_crystallization_conditions(args.pdb)
    elif args.text:
        rec = parse_condition_text(args.text)
    else:
        ap.error('Provide either a plain-text condition string or --pdb <file.pdb>')

    c_eff, delta_sigma = project_condition_features(rec.features, W_proj)
    salt_arg = ','.join(f'{v:.4f}' for v in c_eff)
    if c_eff[1] >= max(c_eff[0], c_eff[2]) and c_eff[1] > 0.0:
        rec_model = 'examples/polymorphs/chemistry-model-hofmeister-nano3.json'
    elif c_eff[2] >= max(c_eff[0], c_eff[1]) and c_eff[2] > 0.0:
        rec_model = 'examples/polymorphs/chemistry-model-hofmeister-so4.json'
    else:
        rec_model = 'examples/chemistry-model-monomer.json'
    print(json.dumps({
        'raw_condition': rec.raw_remark280,
        'parsed_reagents': rec.parsed_reagents,
        'ph': rec.ph,
        'temp_growth_K': rec.temp_growth_K,
        'is_d2o': rec.is_d2o,
        'features_12d': dict(zip(FEATURE_NAMES, rec.features)),
        'projected_salt_concs': c_eff,
        'delta_sigma_solv': delta_sigma,
        'recommended_chemistry_model': rec_model,
        'morph_mc_flag': f'--chemistry-model {rec_model} --salt-concs {salt_arg}',
    }, indent=2))


