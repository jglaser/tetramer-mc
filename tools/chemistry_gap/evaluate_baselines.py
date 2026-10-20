"""Evaluate and summarize the 3 progressive physics baselines with and without the learned
SE(3)-invariant interface chemistry residual model (`ChemistryModel`):
  1. Hard shape + 6 A square well (`--no-solvation`)
  2. Ideal Asakura-Oosawa depletion (`--depletion-z 0.056`) + 6 A square well
  3. Exact White Bear II morphometric solvation + 6 A square well + like-charge penalty

Usage:
  /vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/gudhienv/bin/python \
      tools/chemistry_gap/evaluate_baselines.py
"""
import json

D = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/pair-binding'
B = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/b22'


def main():
    files = [
        ('B1_raw', f'{D}/binding_key12_b1_raw.json'),
        ('B1_chem', f'{D}/binding_key12_b1_chem.json'),
        ('B2_raw', f'{D}/binding_key12_b2_raw.json'),
        ('B2_chem', f'{D}/binding_key12_b2_chem.json'),
        ('B3_raw', f'{D}/binding_key12_b3_raw.json'),
        ('B3_chem', f'{D}/binding_key12_b3_chem.json'),
    ]
    data = {
        name: {
            r['label']: {round(x['epsilon'], 6): x['lnK'] for x in r['K']}[0.00325]
            for r in json.load(open(path))
        }
        for name, path in files
    }
    labels = list(data['B3_raw'].keys())
    print('=== 1. Key Contact Basin Association Constants ln(K / A^3) at eps = 0.00325 kT, w = 6.0 A ===')
    print(f"{'Contact':15s} | {'B1_raw':>7s} {'B1_chem':>8s} | {'B2_raw':>7s} {'B2_chem':>8s} | {'B3_raw':>7s} {'B3_chem':>8s}")
    print('-' * 72)
    for l in labels:
        print(f"{l:15s} | {data['B1_raw'][l]:+7.2f} {data['B1_chem'][l]:+8.2f} | "
              f"{data['B2_raw'][l]:+7.2f} {data['B2_chem'][l]:+8.2f} | "
              f"{data['B3_raw'][l]:+7.2f} {data['B3_chem'][l]:+8.2f}")

    print('\n=== 2. Greedy Nucleation Path Free-Energy Steps dG_step = -ln(rho K_site) at 2.8 mM ===')
    sp_raw = json.load(open(f'{B}/site_path_w6.json'))
    sp_chem = json.load(open(f'{D}/site_path_b3_chem.json'))
    sp_b1c = json.load(open(f'{D}/site_path_b1_chem.json'))
    ln_rho = -13.2906
    print(f"{'Step':25s} | {'lnK_raw(325)':>12s} {'dG_raw':>8s} | {'B3_chem(325)':>12s} {'dG_B3c':>8s} | {'B1_chem(325)':>12s} {'dG_B1c':>8s}")
    print('-' * 94)
    for r0, r1, r2 in zip(sp_raw, sp_chem, sp_b1c):
        k0 = {round(x['epsilon'], 6): x['lnK'] for x in r0['K']}
        k1 = {round(x['epsilon'], 6): x['lnK'] for x in r1['K']}
        k2 = {round(x['epsilon'], 6): x['lnK'] for x in r2['K']}
        dg0 = -(k0[0.00325] + ln_rho)
        dg1 = -(k1[0.00325] + ln_rho)
        dg2 = -(k2[0.00325] + ln_rho)
        print(f"{r0['label']:25s} | {k0[0.00325]:+12.2f} {dg0:+8.2f} | {k1[0.00325]:+12.2f} {dg1:+8.2f} | {k2[0.00325]:+12.2f} {dg2:+8.2f}")

    print('\n=== 3. Reduced Second Virial Coefficient B22 / B22^HS (V_HS = 24750 A^3) ===')
    v_raw = json.load(open(f'{D}/virial_b3_raw.json'))
    v_chem = json.load(open(f'{D}/virial_b3_chem.json'))
    v_b1c = json.load(open(f'{D}/virial_b1_chem.json'))
    for x0, x1, x2 in zip(v_raw['B22'], v_chem['B22'], v_b1c['B22']):
        e = x0['epsilon']
        b0, se0 = x0['B22_A3'] / 99000.0, x0['B22_se_A3'] / 99000.0
        b1, se1 = x1['B22_A3'] / 99000.0, x1['B22_se_A3'] / 99000.0
        b2, se2 = x2['B22_A3'] / 99000.0, x2['B22_se_A3'] / 99000.0
        print(f"eps={e:.5f}: B3_raw = {b0:+6.2f} +- {se0:.2f} | B3_chem = {b1:+6.2f} +- {se1:.2f} | B1_chem = {b2:+6.2f} +- {se2:.2f}")


if __name__ == '__main__':
    main()
