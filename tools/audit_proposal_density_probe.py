#!/usr/bin/env python3
"""Independent coefficient-aware comparison of deterministic Rust density probes.

No geometry predicates, physical weights, random draws or proposal fitting.
Numerical agreement is checked only at the explicitly supplied probe panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from dimer_destination_density import exported_covariance, map_cholesky, ordered_sum
from normalizer_proposal_density import NormalizerProposalDensity, scalar_cholesky
from prepare_smc_normalizer_atlas import Density, unwrap_proposal_model

# Three arithmetic excerpts are shared with the earlier independent audit.
# Constructor/current chart implementations are bound to the reviewed source
# used by proposal-density-probe validation attempt01, not an old scorer.
SOURCE_EXCERPTS = (
    ('src/docking.rs', 'pub fn new(\n        model: FrozenRelativePoseProposal,',
     '    pub fn with_anchor_index', '0dbfe5090d6d0ef6207d4d666f180a9a56bd307596c1379cb3112c108064c1fc'),
    ('src/proposal.rs', '    pub fn component_parameters(',
     '    /// Replace Gaussian parameters', '48c6c1cb0146e16f3a4f859cfa9d3334072ebe6ad49f8d7aa883887824d5a948'),
    ('src/proposal.rs', 'fn prepare_cholesky(', '/// The finite Cayley coordinate',
     '09ddf40beec7203b00ef5e837bfcb359bd8bd3a265034644ba9d03875d8887ea'),
    ('src/basin_involution.rs', 'fn cholesky(', 'fn chart_parameters_valid(',
     'e06c2385578d028b4a68cb912f64011c63b62ea8f441910dc13945731bb6c238'),
    ('src/basin_involution.rs', '    pub fn encode(', '\n}\n\n/// The correlated latent map',
     'e6354e442aa659e280a96d591992bbfbb5300c730e63b2a0be67197c35c9bc05'),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_source(path):
    bundle=json.loads(path.read_bytes()); checked=[]
    for name,start,end,expected in SOURCE_EXCERPTS:
        entry=bundle['files'][name]; text=entry['text']
        require(hashlib.sha256(text.encode()).hexdigest()==entry['sha256'], 'Corrupt source entry: '+name)
        first=text.index(start); actual=hashlib.sha256(text[first:text.index(end,first)].encode()).hexdigest()
        require(actual==expected, 'Unknown constructor/factor arithmetic: '+name)
        checked.append(dict(file=name,sha256=actual))
    return dict(source_bundle_sha256=sha(path),excerpts=checked)


def prepare_references(model, source_bundle):
    """Keep original direct factors and reconstructed map factors distinct."""
    direct=NormalizerProposalDensity(model,source_bundle=source_bundle)
    base,flags=unwrap_proposal_model(model)
    # Explicit scalar products/sums: do not replace these by BLAS LL.T or
    # LAPACK Cholesky. Both change the law for ill-conditioned covariances.
    factors=np.asarray([map_cholesky(exported_covariance(scalar_cholesky(c)))
                        for c in base['covariances']])
    mapped=Density(model)
    total=ordered_sum(base['weights'])
    mapped.weights=np.asarray([(float(base['weights'][k])/total)/(2 if flags[k] else 1)
                               for k in mapped.base_indices])
    mapped.lower=factors[mapped.base_indices]
    mapped.logdet=np.asarray([ordered_sum(math.log(float(l[i,i])) for i in range(6)) for l in mapped.lower])
    return direct,mapped


def compare_row(rust, expected, tolerance):
    status=rust.get('status'); value=rust.get('log_density')
    if status=='finite' and value is not None and math.isfinite(float(value)) and math.isfinite(float(expected)):
        difference=float(value)-float(expected)
        return dict(passed=abs(difference)<=tolerance,rust_status=status,reference_status='finite',
                    rust_log_density=float(value),reference_log_density=float(expected),difference=difference)
    if status=='negative_infinity' and value is None and float(expected)==-math.inf:
        return dict(passed=True,rust_status=status,reference_status='negative_infinity',difference=None)
    reference_status='finite' if math.isfinite(float(expected)) else ('negative_infinity' if float(expected)==-math.inf else 'invalid')
    return dict(passed=False,rust_status=status,reference_status=reference_status,difference=None,error=rust.get('error'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline-model','exported-model','baseline-report','exported-report','probes','source-bundle','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-probes',type=int,required=True)
    parser.add_argument('--absolute-tolerance',type=float,default=2e-6)
    args=parser.parse_args()
    require(not args.out.exists(), 'Fresh output required')
    require(args.expected_probes>0 and math.isfinite(args.absolute_tolerance) and args.absolute_tolerance>0, 'Invalid audit allocation')
    files={name:getattr(args,name) for name in ('baseline_model','exported_model','baseline_report','exported_report','probes','source_bundle')}
    hashes={name:sha(path) for name,path in files.items()}
    binding=bind_source(args.source_bundle)
    panel=json.loads(args.probes.read_bytes())
    require(isinstance(panel,list) and len(panel)==args.expected_probes,'Wrong complete probe allocation')
    require(len({p['label'] for p in panel})==len(panel),'Duplicate labels')
    poses=[p['pose'] for p in panel]; arms={}; passed=True
    for arm in ('baseline','exported'):
        model=json.loads(files[arm+'_model'].read_bytes())
        rust=json.loads(files[arm+'_report'].read_bytes())
        base,flags=unwrap_proposal_model(model)
        require(rust['schema']=='proposal-density-probe-v1' and rust['complete'] is True, 'Incomplete Rust report')
        require(rust['model_sha256']==hashes[arm+'_model'] and rust['probes_sha256']==hashes['probes'],'Mismatched Rust inputs')
        require(rust['shape_sha256']==base['shape_sha256'] and rust['base_components']==len(base['weights']),'Mismatched model inventory')
        require(rust['reciprocal_components']==flags and len(rust['virtual_branches'])==len(flags)+sum(flags),'Mismatched virtual branches')
        require(rust['transport_factor_check'] is True and rust['transport_pair_entries']==1,'Require the declared one-pair factor check')
        require(all(rust[k]==0 for k in ('physical_updates','geometry_queries','random_draws')),'Unexpected physical execution')
        require(len(rust['probes'])==len(panel) and all(a['probe']==b for a,b in zip(rust['probes'],panel)),'Changed probe identities or poses')
        direct,mapped=prepare_references(model,args.source_bundle)
        expected={'direct':direct.evaluate(poses)[0],'transport':mapped.evaluate(poses)[0]}
        rows=[]
        for i,row in enumerate(rust['probes']):
            values={kind:compare_row(row[kind],expected[kind][i],args.absolute_tolerance) for kind in expected}
            rows.append(dict(label=panel[i]['label'],**values))
        counts={kind:sum(not row[kind]['passed'] for row in rows) for kind in expected}
        maximum={kind:max((abs(row[kind]['difference']) for row in rows if row[kind]['difference'] is not None),default=None) for kind in expected}
        arm_passed=rust['passed'] is True and all(n==0 for n in counts.values())
        arms[arm]=dict(passed=arm_passed,base_components=len(base['weights']),virtual_branches=len(direct.weights),
            mismatches=counts,maximum_absolute_log_density_error=maximum,probes=rows,
            direct_factor_certificates=len(direct.factor_diagnostics),
            direct_maximum_covariance_certificate_bound_ratio=max(x['maximum_componentwise_bound_ratio'] for x in direct.factor_diagnostics))
        passed &= arm_passed
    require(all(sha(path)==hashes[name] for name,path in files.items()),'Input changed during audit')
    report=dict(schema='proposal-density-independent-audit-v1',complete=True,passed=bool(passed),
        input_sha256=hashes,source_binding=binding,absolute_tolerance=args.absolute_tolerance,expected_probes=args.expected_probes,
        arms=arms,geometry_queries=0,physical_updates=0,random_draws=0,
        scope='Finite deterministic coefficient-aware density agreement; no global density bound, equilibrium validation, core/depletion checks, or proposal-efficiency claim.')
    with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
