#!/usr/bin/env python3
"""Audit complete native-cover draws and retain direct original-chart tail masks."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import heapq
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_latent_region import original_q_contains, shell_log_volume, validate_manifest_q_window
from analyze_native_region_reference import native_q
from analyze_peak_neighborhood import partition_check
from audit_shoulder_mis_independently import chart_values, near
from compare_intermediate_local_reference import read_batches
from compare_intermediate_reference import population_rse
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_native_tail_reference import MASKS, WINDOW, old_radius_bin, exact_cover_check
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments, quota_summary

KEYS = ('full', *MASKS)


def validated_cloud_log_weight(cloud, activity, intensity):
    """Check the recorded positive-Poisson law before reconstructing its weight."""
    require(math.isfinite(activity) and activity >= 0 and math.isfinite(intensity) and intensity > 0,
            'Invalid Poisson bath parameters')
    counters = ('raw_points','overlap_points','retained_cells','created_cells','certified_cells')
    for key in counters:
        require(type(cloud[key]) is int and 0 <= cloud[key] < 2**64, 'Invalid nonnegative integer cloud counter: '+key)
    require(cloud['overlap_points'] <= cloud['raw_points'], 'Accepted overlap count exceeds raw Poisson count')
    require(cloud['retained_cells']+cloud['certified_cells'] <= cloud['created_cells'], 'Envelope leaf counts exceed created cells')
    for key in ('lower_volume','upper_volume','uncertain_volume'):
        require(type(cloud[key]) in (int,float) and math.isfinite(cloud[key]) and cloud[key] >= 0,
                'Invalid finite nonnegative envelope volume: '+key)
    lower,upper,uncertain = (cloud[k] for k in ('lower_volume','upper_volume','uncertain_volume'))
    require(lower <= upper, 'Lower envelope volume exceeds upper volume')
    near(upper, lower+uncertain)
    if uncertain == 0:
        require(cloud['raw_points'] == cloud['overlap_points'] == 0, 'A zero-intensity envelope produced Poisson points')
    require(math.isfinite(cloud['log_weight']), 'Nonfinite cloud log weight')
    value = activity*lower+cloud['overlap_points']*math.log1p(activity/intensity)
    near(value,cloud['log_weight'])
    return value


def validate_terminal_status(status, job):
    require(status['id'] == job['id'] and type(status['returncode']) is int and status['returncode'] == 0,
            'Population launcher did not report terminal success')
    require(type(status['completed_unix']) in (int,float) and math.isfinite(status['completed_unix'])
            and status['completed_unix'] > 0, 'Invalid terminal completion timestamp')


class NativeTailMoments:
    def __init__(self):
        self.values = {kind:{key:LogMoments() for key in KEYS} for kind in ('physical','hard')}

    def add(self, radius=None, physical=None, hard=None, pair=None):
        require((physical is None) == (hard is None), 'Physical and hard zero masks disagree')
        selected = () if physical is None else ('full', old_radius_bin(radius))
        if selected:
            require(math.isfinite(physical) and math.isfinite(hard) and len(pair) == 2, 'Invalid original contribution')
            near(float(logsumexp(pair))-math.log(2), physical)
        for key in KEYS:
            for kind,value in (('physical',physical),('hard',hard)):
                self.values[kind][key].add(value if key in selected else -math.inf,
                                          pair if key in selected and kind == 'physical' else None)

    def merge(self, other):
        for kind in self.values:
            for key in KEYS: self.values[kind][key].merge(other.values[kind][key])

    def report(self):
        result = {}
        for kind, values in self.values.items():
            result[kind] = {}
            for key, moment in values.items():
                row = quota_summary([moment]); row['row_RSE'] = row.pop('stratified_RSE')
                row['variance_rule'] = 'IID row variance over full original N, including invalid and off-mask zeros.'
                result[kind][key] = row
        result['partition_checks'] = {kind:partition_check(values,MASKS) for kind,values in self.values.items()}
        return result


def analyze(preparation, campaign, out, amendment=None):
    preparation,campaign,out = map(lambda p:Path(p).resolve(), (preparation,campaign,out))
    require(not out.exists(), 'Use a fresh analysis output')
    start = time.monotonic(); protocol = read(preparation/'protocol.json')
    amendment_record = None
    if amendment is not None:
        amendment = Path(amendment).resolve(); amended = read(amendment)
        require(amended['schema'] == 'native-tail-analyzer-amendment-v1'
                and Path(amended['base_preparation']).resolve() == preparation
                and Path(amended['physical_campaign']).resolve() == campaign, 'Amendment targets different inputs')
        for name,digest in amended['base_preparation_sha256'].items():
            require(sha(preparation/name) == digest, 'Physical preparation changed after analyzer amendment')
        for name,digest in amended['archived_sha256'].items():
            require(sha(amendment.parent/'provenance'/name) == digest, 'Amended analyzer dependency changed')
        require(sha(Path(__file__)) == amended['archived_sha256'][Path(__file__).name], 'Executing a different analyzer than the amendment')
        amendment_record = dict(path=str(amendment), sha256=sha(amendment), reason=amended['reason'])
    require(protocol['schema'] == 'complete-native-tail-reference-preparation-v1', 'Wrong preparation')
    require(protocol['original_q_window'] == WINDOW and protocol['old_chart_reporting_edges'] == [4.,5.,8.,12.], 'Changed masks')
    for name,digest in read(preparation/'freeze.json').items(): require(sha(preparation/name) == digest, 'Changed frozen preparation')
    for name,digest in protocol['archived_sha256'].items(): require(sha(preparation/'provenance'/name) == digest, 'Changed archive')
    declared = protocol['proposed_campaign']; require(Path(declared['output']).resolve() == campaign, 'Undeclared campaign')
    master = read(campaign/'manifest.json'); cfg = read(campaign/'provenance/config.json')
    region = read(campaign/'provenance/region.json'); model = read(preparation/'model.json'); old = read(preparation/'old-model.json')
    for name,digest in master['archive_sha256'].items(): require(sha(campaign/'provenance'/name) == digest, 'Changed campaign archive')
    require(sha(campaign/'provenance/region.json') == master['region_sha256'] == protocol['region_sha256'], 'Changed region')
    require(region == read(preparation/'region.json') and region['gaussian_chart'] == model, 'Different source region')
    expected_cfg = read(preparation/'config.json')
    for key in ('metadata','fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density'):
        require(cfg[key] == expected_cfg[key], 'Changed physical target: '+key)
    require(sha(campaign/'provenance/shape.json') == region['shape_sha256'] == old['shape_sha256'], 'Different shape')
    exact = exact_cover_check((preparation/'provenance/input-config.json').read_text(), model)
    require([j['seed'] for j in master['jobs']] == declared['seeds'] and len(master['jobs']) == declared['populations'], 'Changed populations')
    require(len(set(declared['seeds'])) == declared['populations'], 'Repeated seed')
    require(master['lambda_ratio'] == declared['lambda_ratio'] == 64. and master['cloud_replicates'] == declared['cloud_replicates'] == 2,
            'Changed cloud law')
    logv = shell_log_volume(region); aggregate = NativeTailMoments(); populations=[]; input_hashes={}; cpu=0.; heap=[]
    errors=dict(cover_radius=0., jacobian=0., q=0.); max_native_old_radius=0.; checked_native=0
    for job in master['jobs']:
        path = Path(job['directory']); manifest=read(path/'manifest.json'); summary=read(path/'summary.json')
        status_path = campaign/f"{job['id']}-status.json"
        require(status_path.is_file(), 'Population terminal status is missing; do not audit a running job')
        validate_terminal_status(read(status_path),job)
        input_hashes[str(status_path)] = sha(status_path)
        require(summary['complete'] and summary['manifest'] == manifest, 'Incomplete population')
        require(manifest['schema'] == 'uniform-latent-region-normalizer-v2', 'Require all-pose extended output')
        require(job['samples'] == summary['samples'] == manifest['samples'] == declared['samples_per_population'], 'Changed full N')
        require(manifest['seed'] == job['seed'], 'Changed seed')
        for field,expected in dict(region_sha256=master['region_sha256'], shape_sha256=region['shape_sha256'],
            config_sha256=sha(campaign/'provenance/config.json'), executable_sha256=protocol['executable_sha256'],
            physical_fixed_neighbors=cfg['fixed_poses'], chart_anchor=region['fixed_neighbor'],
            latent_radius=2., minimum_latent_radius=0., activity=.035, lambda_ratio=64., cloud_replicates=2).items():
            require(manifest[field] == expected, 'Changed population field: '+field)
        require(manifest['lambda'] == manifest['activity']*64., 'Changed Poisson intensity')
        validate_manifest_q_window(manifest,region,WINDOW); near(manifest['log_latent_shell_volume'],logv)
        for name,field in (('input-config.json','config_sha256'),('shape.json','shape_sha256'),
                           ('region.json','region_sha256'),('source-bundle.json','source_bundle_sha256')):
            require(sha(path/'provenance'/name) == manifest[field], 'Changed population source')
        local=NativeTailMoments(); digest=hashlib.sha256(); count=0
        for lines,rows in read_batches(path/'samples.jsonl'):
            for line in lines: digest.update(line)
            positions=np.asarray([r['pose']['position'] for r in rows]); q=np.asarray([r['pose']['orientation'] for r in rows])
            near(np.sum(q*q,axis=1),np.ones(len(q)),1e-8)
            rotations=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
            _,cover_r,jac=chart_values(model,positions,rotations,region['fixed_neighbor'])
            old_r=chart_values(old,positions,rotations,region['fixed_neighbor'])[1][:,0]
            for i,row in enumerate(rows):
                require(row['draw'] == count, 'Missing or repeated row'); count+=1
                errors['cover_radius']=max(errors['cover_radius'],near(cover_r[i,0],row['latent_radius']))
                errors['jacobian']=max(errors['jacobian'],near(jac[i,0],row['log_physical_jacobian']))
                errors['q']=max(errors['q'],near(native_q(cfg['metadata'],row['pose']),row['q']))
                require(0<=row['latent_radius']<=2. and row['region_valid'] == original_q_contains(row['q'],WINDOW), 'Changed native support/predicate')
                require(row['capture_valid'] == (math.dist(row['pose']['position'],cfg['capture_center'])<=18.), 'Changed capture predicate')
                if row['region_valid']:
                    checked_native+=1; max_native_old_radius=max(max_native_old_radius,float(old_r[i]))
                    require(old_r[i]<52., 'Native row violates support certificate; do not censor it')
                valid=row['capture_valid'] and row['hard_valid'] and row['region_valid']
                if not valid:
                    require(row['log_importance_weight'] is None and row['log_hard_weight'] is None and not row['clouds'], 'Invalid row carries weight')
                    local.add(); continue
                require(len(row['clouds'])==2, 'Require two independent cloud estimates')
                hard=logv+jac[i,0]; near(hard,row['log_hard_weight']); pair=[]
                envelope_keys = ('lower_volume','upper_volume','uncertain_volume','retained_cells','created_cells','certified_cells')
                require(all(row['clouds'][0][key] == row['clouds'][1][key] for key in envelope_keys),
                        'The two clouds do not share their recorded fixed envelope')
                for cloud in row['clouds']:
                    validated_cloud_log_weight(cloud,.035,manifest['lambda'])
                    pair.append(cloud['log_weight']+row['log_hard_weight'])
                near(float(logsumexp(pair))-math.log(2),row['log_importance_weight'])
                local.add(float(old_r[i]),row['log_importance_weight'],row['log_hard_weight'],pair)
                point=dict(population=job['id'],seed=job['seed'],draw=row['draw'],pose=row['pose'],original_q=row['q'],
                           old_radius=float(old_r[i]),reporting_bin=old_radius_bin(float(old_r[i])),
                           log_importance_weight=row['log_importance_weight'])
                entry=(row['log_importance_weight'],job['id'],row['draw'],point)
                if len(heap)<8: heapq.heappush(heap,entry)
                elif entry[:3]>heap[0][:3]: heapq.heapreplace(heap,entry)
        require(count==job['samples'] and digest.hexdigest()==summary['samples_sha256'], 'Changed rows/count')
        local_report=local.report()
        for kind,name in (('physical','region'),('hard','hard_region')):
            current=local_report[kind]['full']['logQ']; expected=summary['estimates'][name]['logQ']
            if current is None: require(expected is None, 'Changed zero estimate')
            else: near(current,expected)
        populations.append(dict(id=job['id'],seed=job['seed'],samples=count,**local_report)); aggregate.merge(local)
        cpu+=summary['sampler_cpu_seconds']
        for name in ('samples.jsonl','manifest.json','summary.json'): input_hashes[str(path/name)]=sha(path/name)
    result=aggregate.report()
    for kind in ('physical','hard'):
        for key in KEYS: result[kind][key]['independent_population_RSE']=population_rse([p[kind][key]['logQ'] for p in populations])
    atom=AtomUnionAudit(read(campaign/'provenance/shape.json'),cfg['fixed_poses']); top=[]
    for *_,point in sorted(heap,reverse=True):
        gaps=atom.gaps(point['pose']); require(min(gaps)>=0, 'Independent top-pose atom check failed')
        point['minimum_atomic_gap_by_neighbor_A']=gaps; top.append(point)
    sources={'protocol.json':preparation/'protocol.json','freeze.json':preparation/'freeze.json','model.json':preparation/'model.json',
             'old-model.json':preparation/'old-model.json','region.json':campaign/'provenance/region.json',
             'config.json':campaign/'provenance/config.json','campaign-manifest.json':campaign/'manifest.json',
             **local_dependencies([Path(__file__)])}
    if amendment is not None: sources['analyzer-amendment.json'] = amendment
    archive=out/'provenance'; archive.mkdir(parents=True)
    for name,path in sources.items(): (archive/name).write_bytes(path.read_bytes())
    result.update(complete=True, campaign=str(campaign), original_q_window=WINDOW, reporting_edges=[4.,5.,8.,12.],
                  populations=populations, CPU_seconds=cpu, reconstruction_max_errors=errors,
                  native_rows_checked_against_support_certificate=checked_native, maximum_native_old_radius=max_native_old_radius,
                  cover_certificate=exact, top_poses=top, independent_atom_check_count=len(top),
                  analyzer_amendment=amendment_record,
                  input_sha256={**input_hashes,**{str(p):sha(p) for p in sources.values()}},
                  archived_sha256={name:sha(archive/name) for name in sources}, elapsed_seconds=time.monotonic()-start,
                  scope='Direct full-N native statistical weights under one independent complete geometric-cover law. Old r>12 has no finite cutoff. All masks preserve original q/capture/AB; no noisy subtraction, success conditioning, refit, or tail mass bound. Observed errors cannot exclude unseen weight.')
    write(out/'analysis.json',result); print(result['physical']); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--amendment',type=Path,help='Frozen analyzer amendment linked to the unchanged physical preparation')
    args=parser.parse_args(); analyze(args.preparation,args.campaign,args.out,args.amendment)
