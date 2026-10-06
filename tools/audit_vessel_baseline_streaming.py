#!/usr/bin/env python3
"""Schema-4/8 full-wall baseline audit with a reporting-only R4 chart.

The chart never supplies a proposal density or target restriction. The baseline
must provide an attempted-draw journal and completion hashes. Native labels are
left to a later pass over the saved geometry records.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

import audit_full_vessel_baseline as baseline
from audit_hard_free_vessel_streaming import WallOracle, attempt_batches, row_counts
from analyze_mobile_native_pocket import local_sources
from analyze_r4_smc_control import Chart, Ledger, close, read, require, sha, write
from streaming_weight_moments import RegionMoments
from prepare_deep_far_normalizer_atlas import registration

SCHEMA = 'full-vessel-baseline-streaming-audit-v1'
CLASSES = ('total','inside_R4','outside_R4','exclusion_contact','unbound')


def validate_proposal_contract(manifest):
    """The envelope changes only the defensive vessel law, never the target."""
    require(type(manifest.get('schema')) is int and manifest['schema'] in (4,8),
            'Schema-4 cube or schema-8 wall-envelope baseline required')
    require(not any(k.startswith(('outer_', 'latent_')) for k in manifest),
            'Baseline cannot contain an outer or regional mixture')
    envelope_fields = ('pre_envelope_schema','vessel_uniform_schema','vessel_uniform_envelope')
    if manifest['schema'] == 8:
        require(type(manifest.get('pre_envelope_schema')) is int
                and manifest['pre_envelope_schema'] == 4
                and manifest.get('vessel_uniform_schema') == 'one-atom-wall-envelope-v1'
                and isinstance(manifest.get('vessel_uniform_envelope'),dict),
                'Schema 8 needs the pure-vessel wall-envelope contract')
    else:
        require(not any(k in manifest for k in envelope_fields),
                'Schema 4 cannot conceal a wall-envelope proposal')


def check_rows(config, manifest, rows, vessel, reporting):
    """Schema-aware original-vessel algebra; no relabeling of schema-8 inputs.

    The complete anchor-averaged density already includes the learned chart
    Jacobian and normalized Haar law. The envelope adds a unit-Jacobian shift
    at each orientation. No reporting-chart Jacobian enters these weights.
    This retains the schema-4 reference's row checks and adds strict exclusion
    of outer-mixture fields and checked envelope presence.
    """
    validate_proposal_contract(manifest)
    require((manifest['schema'] == 8) == (vessel.envelope is not None),
            'Declared vessel law and reconstructed envelope differ')
    require(rows and [r['draw'] for r in rows] == list(range(manifest['samples'])),
            'Missing attempted draw')
    require(all(r.get('pose') is not None for r in rows),'Censored numerical null')
    require(all(not any(k.startswith(('outer_', 'latent_', 'log_latent_'))
                        or k == 'log_vessel_proposal_density' for k in row) for row in rows),
            'Baseline has outer-mixture generation metadata')
    poses = [row['pose'] for row in rows]
    logs,geometry = vessel.evaluate(poses)
    require(np.isfinite(logs).all(),'Generated pose lacks vessel support')
    coordinates = reporting.evaluate_many(poses)
    metric = registration(poses,config['metadata']); errors = []
    for i,row in enumerate(rows):
        require(row.get('proposal') is not None,'Missing vessel generation metadata')
        errors.append(baseline.log_close(row['log_proposal_density'],logs[i],
                                        'Full original vessel density differs'))
        require(type(row['capture_valid']) is bool and row['capture_valid'] == bool(geometry['capture'][i]),
                'Capture predicate differs')
        require(type(row['hard_valid']) is bool and type(row['wall_valid']) is bool,
                'Non-Boolean hard/wall flag')
        if row['hard_valid']:
            require(row['capture_valid'] and row['wall_valid'],'Hard-valid row leaves physical domain')
            close(row['log_hard_weight'],-float(logs[i]),
                  'Baseline hard weight must use original vessel density, no mixture or extra J')
            close(row['q'],float(metric[i]),'Original physical registration metric differs')
            require(len(row['clouds']) == manifest['cloud_replicates'] == 2,'Two independent clouds required')
            for cloud in row['clouds']:
                require(type(cloud['overlap_points']) is int and cloud['overlap_points'] >= 0
                        and math.isfinite(cloud['lower_volume']) and cloud['lower_volume'] >= 0
                        and math.isfinite(cloud['uncertain_volume']) and cloud['uncertain_volume'] >= 0,
                        'Invalid cloud count/volume')
                z,lam = manifest['activity'],manifest['lambda']
                factor = z*cloud['lower_volume']+cloud['overlap_points']*math.log1p(z/lam) if z else 0.
                close(cloud['log_weight'],factor,'Poisson estimator count identity differs')
            expected = float(logsumexp([c['log_weight'] for c in row['clouds']])-math.log(2)-logs[i])
            close(row['log_importance_weight'],expected,'Arithmetic cloud-mean weight differs')
        else:
            require(all(row[k] is None for k in ('log_hard_weight','log_importance_weight','q','region','depletion_contact'))
                    and not row['clouds'],'Invalid attempted draw lost its explicit zero')
    return dict(checked_attempts=len(rows),maximum_log_density_error=max(errors),
        valid_outside_R4=sum(r['hard_valid'] and not c.in_reference_ball for r,c in zip(rows,coordinates)),
        scope='Original vessel density only on every attempted world pose; regional coordinates are reporting labels.'),coordinates


class ReportingChart:
    def __init__(self, region):
        require(region['mahalanobis_radius'] == 4., 'Frozen R4 reporting chart required')
        self.chart = Chart(region)

    def evaluate_many(self, poses):
        return [SimpleNamespace(in_reference_ball=self.chart.evaluate(p)['inside']) for p in poses]


def audit(directory, out, binary, region_path, batch_size=64):
    require(sys.flags.optimize == 0, 'Unoptimized Python required for independent assertions')
    require(type(batch_size) is int and 1 <= batch_size <= 1024, 'Batch size must be 1..1024')
    root, out, binary, region_path = (Path(p).resolve() for p in (directory,out,binary,region_path))
    require(not out.exists(), 'Fresh baseline audit destination required')
    started = time.process_time(); ledger = Ledger(); sources = local_sources(__file__)
    for path in sources.values(): ledger.bind(path)
    manifest = read(ledger.bind(root/'manifest.json')); summary = read(ledger.bind(root/'summary.json'))
    validate_proposal_contract(manifest)
    require(summary['complete'] is True and summary['manifest'] == manifest and summary['numerical_nulls'] == 0
            and not (root/'failure.json').exists(), 'Incomplete or failed population')
    require(type(manifest['samples']) is int and manifest['samples'] > 0 and manifest['cloud_replicates'] == 2,
            'Positive all-attempt allocation and two clouds required')
    require(manifest['bath_wall_permeable'] is True and 'atomic_wall' in manifest
            and manifest.get('attempt_journal') == 'attempts.jsonl; begin before each attempt; no retries'
            and manifest.get('resume_supported') is False, 'Full-wall baseline journal/domain absent')
    for name,key in [('input-config.json','config_sha256'),('model.json','model_sha256'),
                     ('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
        ledger.bind(root/'provenance'/name,manifest[key])
    ledger.bind(binary,manifest['executable_sha256']); bundle = root/'provenance/source-bundle.json'
    require(bundle.read_bytes() in binary.read_bytes(),'Pinned source bundle absent from executable')
    samples = ledger.bind(root/'samples.jsonl',summary['samples_sha256'])
    attempts = ledger.bind(root/'attempts.jsonl',summary['attempts_sha256'])
    config = read(ledger.bind(root/'config.json')); original = read(root/'provenance/input-config.json')
    for key in ('fixed_poses','capture_center','capture_radius','depletant_radius','metadata'):
        require(config[key] == original[key],'Physical config changed: '+key)
    require(config['reservoir_density'] == manifest['activity'] and config.get('target_region') is None,
            'Changed bath or hidden target restriction')
    require(math.isfinite(manifest['activity']) and manifest['activity'] >= 0
            and math.isfinite(manifest['lambda']) and manifest['lambda'] > 0,'Invalid bath/cloud intensity')
    close(manifest['lambda'],config['poisson_lambda_ratio']*manifest['activity']
          if manifest['activity'] > 0 else 1.,'Auxiliary intensity differs')
    region = read(ledger.bind(region_path))
    require(region['shape_sha256'] == manifest['shape_sha256'] and region['fixed_neighbor'] in config['fixed_poses']
            and region.get('physical_fixed_neighbors',[region['fixed_neighbor']]) == config['fixed_poses'],
            'Reporting shape/scaffold differs')
    if 'physical_metric' in region:
        require(region['physical_metric'] == config['metadata'],'Reporting registration metric differs')
    reporting = ReportingChart(region)
    shape = read(root/'provenance/shape.json')
    vessel = baseline.VesselDensity(config,manifest,read(root/'provenance/model.json'),bundle,shape=shape)
    require((manifest['schema'] == 8) == (vessel.envelope is not None),
            'Declared vessel law and reconstructed envelope differ')
    wall = WallOracle(config,manifest,shape,read(bundle))
    contact = baseline.PrunedExclusionContact(shape,config['fixed_poses'],config['depletant_radius'])
    reducers = {name:RegionMoments(manifest['samples']) for name in CLASSES}
    counters = Counter(); processed = near = generation_count = outside = batches = peak = 0
    max_density_error = 0.
    out.mkdir(parents=True)
    def status(phase,**extra):
        temporary = out/'status.json.tmp'
        write(temporary,dict(complete=phase == 'complete',phase=phase,processed_attempts=processed,
                             batch_size=batch_size,**extra)); temporary.replace(out/'status.json')
    status('audit')
    try:
        with (out/'geometry.jsonl').open('x') as output:
            for rows in attempt_batches(samples,attempts,manifest['samples'],batch_size):
                local = dict(manifest,samples=len(rows)); adapted = [dict(r,draw=i) for i,r in enumerate(rows)]
                density,coordinates = check_rows(config,local,adapted,vessel,reporting)
                max_density_error = max(max_density_error,density['maximum_log_density_error'])
                outside += density['valid_outside_R4']
                generation = baseline.check_generation_metadata(config,local,
                    [dict(r,outer_branch='vessel') for r in adapted],vessel)
                generation_count += generation['checked_vessel_generation_rows']
                counters.update(baseline.check_cloud_envelopes_and_counts(local,
                    dict(samples=len(rows),**row_counts(rows)),adapted))
                batches += 1; peak = max(peak,len(rows))
                for row,coordinate in zip(rows,coordinates):
                    wall.check(row)
                    geometry = contact.classify(row['pose'],capture_valid=row['capture_valid'],wall_valid=row['wall_valid'])
                    valid = row['hard_valid']; expected = row['capture_valid'] and row['wall_valid'] and geometry['core_disjoint']
                    require(valid == expected,'Independent atomic hard predicate differs')
                    if valid: require(row['depletion_contact'] == geometry['exclusion_contact'],'Independent exclusion contact differs')
                    inside = coordinate.in_reference_ball
                    classes = dict(total=valid,inside_R4=valid and inside,outside_R4=valid and not inside,
                        exclusion_contact=valid and geometry['exclusion_contact'],unbound=valid and not geometry['exclusion_contact'])
                    for name,selected in classes.items(): reducers[name].add(row,selected)
                    output.write(json.dumps(dict(draw=row['draw'],classes=classes,geometry=geometry),allow_nan=False)+'\n')
                    processed += 1; near += int(geometry['near_core_boundary'])
                output.flush(); status('audit')
        require(processed == generation_count == wall.checked == summary['samples'] == manifest['samples'],
                'Lost unconditional audit/generation/wall denominator')
        require(all(summary[k] == v for k,v in counters.items()) and wall.rejected == summary['wall_rejected'],
                'Global rejection/point/wall counters differ')
        estimates = {name:reducer.result() for name,reducer in reducers.items()}
        for key,kind in [('total','Qz'),('hard_total','Q0')]:
            baseline.log_close(summary['estimates'][key]['log_normalizer'],
                baseline.nullable_log(estimates['total'][kind]['logQ']),'Total all-attempt mass differs')
        ledger.recheck(); (out/'provenance').mkdir()
        for name,path in sources.items():
            shutil.copy2(path,out/'provenance'/name)
            require(sha(out/'provenance'/name) == ledger.files[str(Path(path).resolve())],'Audit source changed while copying')
        shutil.copy2(region_path,out/'provenance/reporting-region.json')
        require(sha(out/'provenance/reporting-region.json') == ledger.files[str(region_path)],'Reporting region changed')
        ledger.recheck()
        result = dict(schema=SCHEMA,complete=True,population=str(root),manifest=manifest,estimates=estimates,
            density_audit=dict(checked_attempts=processed,maximum_log_density_error=max_density_error,
                valid_outside_R4=outside,scope='Original full vessel density only; no regional density or Jacobian factor.'),
            vessel_generation_audit=dict(checked_vessel_generation_rows=generation_count),
            primitive_count_audit=dict(counters),wall_audit=wall.result(),geometry_audit=contact.report(),
            near_core_boundary_poses=near,source_sha256=ledger.files,samples_sha256=sha(samples),
            attempts_sha256=sha(attempts),geometry_sha256=sha(out/'geometry.jsonl'),
            reporting_region_binding=dict(path=str(region_path),sha256=sha(region_path),
                affects_proposal=False,restricts_target=False,scope='Chart ball only; capture and other historical pocket restrictions are not applied here.'),
            executable_binding=dict(path=str(binary),sha256=manifest['executable_sha256'],artifact_verified=True),
            batching=dict(requested_rows=batch_size,peak_rows=peak,batches=batches,
                scalar_top_weights_per_region=max(1,(manifest['samples']+99)//100)),
            analysis_CPU_seconds=time.process_time()-started,new_pose_draws=0,new_clouds=0,new_native_classifier_calls=0,
            scope='Independent full-wall baseline audit, preserving every attempted zero and original denominator. '
                  'Reporting chart labels do not change target or proposal. No convergence or assembly claim. '
                  'RNG, exact thinning/envelope certification and floating-point execution remain obligations.')
        if vessel.envelope is not None:
            result['vessel_uniform_envelope'] = vessel.envelope.witness
        write(out/'analysis.json',result); status('complete',analysis_sha256=sha(out/'analysis.json'))
        write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
        return result
    except BaseException as error:
        status('failed',error=str(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory','out','binary','region'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=64); args = parser.parse_args()
    result = audit(args.directory,args.out,args.binary,args.region,args.batch_size)
    print(json.dumps(dict(complete=result['complete'],samples=result['manifest']['samples'],batching=result['batching'])))
