#!/usr/bin/env python3
"""Present completed, authenticated region-mass analysis; no sampling/classification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/hard-free-protein-smc-recovered-analysis-preparation-20261002/analysis.json'
EXPECTED = '12552e9a3168a58f9af9bce3e514669796fa32d22262ac8b56f2eeaa214c4dda'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    data = json.loads(raw)
    assert data['complete'] and not data['full_vessel_gate_open'] and not data['assembly_gate_open']
    args.out.mkdir(parents=True, exist_ok=False)
    regions = ['registered_native_entry', 'old_R5_intersection_native', 'remaining_R4_native']
    rows = []
    for region in regions:
        new = data['comparisons']['larger_iid_baseline']['left'][region]
        baseline = data['comparisons']['larger_iid_baseline']['right'][region]
        conditioned = data['comparisons']['larger_iid_conditioned']['right'][region]
        historical = data['comparisons']['historical_broad']['right'][region]
        rows.append(dict(region=region, estimates=dict(new_smc=new, iid=baseline,
            conditioned_iid=conditioned, historical_smc=historical),
            new_vs_iid=data['comparisons']['larger_iid_baseline']['comparisons'][region],
            new_vs_conditioned=data['comparisons']['larger_iid_conditioned']['comparisons'][region]))
    report = dict(schema='hard-free-smc-reconciliation-presentation-v1', source=str(SOURCE),
        source_sha256=EXPECTED, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_physical_draws=0, new_classifier_calls=0, rows=rows,
        failed_material_strata=data['failed_material_strata'],
        no_entry_observed=False, full_vessel_gate_open=False, assembly_gate_open=False,
        scope='Same frozen R4 domain, shape, scaffold and physical measure. Whole-population linear masses. Aggregate native agreement does not remove stratum failures, bound unseen contacts or decide finite-system assembly.')
    (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    names = ['Earlier SMC', 'New SMC', 'Uniform IID', 'Conditioned IID']
    keys = ['historical_smc', 'new_smc', 'iid', 'conditioned_iid']
    colors = ['#b06b40', '#26709b', '#585f6b', '#54946d']
    titles = ['Complete native region', 'Native: old R5 intersection', 'Native: remaining R4']
    for ax, row, title in zip(axes, rows, titles):
        reference = row['estimates']['iid']['log_Q']
        for i, (key, color) in enumerate(zip(keys, colors)):
            item = row['estimates'][key]
            mass = math.exp(item['log_Q'] - reference)
            ax.errorbar(mass, i, xerr=mass*item['population_relative_SE'], fmt='o',
                        color=color, capsize=4)
        ax.axvline(1, color='#888888', linewidth=.8, linestyle='--')
        ax.set_yticks(range(4), names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Region mass / uniform-IID mean')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Native-region mass agreement improves; convergence gates remain closed', fontsize=13)
    fig.text(.5, .025, 'Four independent populations per estimator; bars show one population-based SE. '
             'Fixed scaffold, 1.5 Å / 0.035 Å⁻³.\nCompeting terminal mass is unobserved in SMC; '
             'material stratum discrepancies and full-domain coverage remain unresolved.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.92))
    fig.savefig(args.out/'reconciliation.png', dpi=180)
    fig.savefig(args.out/'reconciliation.svg')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.out), physical_draws=0, classification_calls=0,
                          regional_primary_comparisons_pass=True, full_vessel_gate_open=False)))


if __name__ == '__main__':
    main()
