"""Exact original-q boundary controls for independent latent-region audits."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_latent_region import analyze,original_q_contains,original_q_window,validate_manifest_q_window,shell_log_volume
from prepare_smc_normalizer_atlas import sha,write


def pose(x):return dict(position=[x,0.,0.],orientation=[1.,0.,0.,0.])


def fixture(root,flags=None):
    archive=root/'provenance';archive.mkdir()
    write(archive/'shape.json',dict(atoms=[dict(center=[0.]*3,radius=.01)]))
    shape_sha=sha(archive/'shape.json')
    metric=dict(native_poses=[pose(0)],rigid_members=[pose(0)],member_error_scale=1.,angle_error_scale_deg=180.)
    fixed=[pose(0),pose(1.5)]
    cfg=dict(shape=str(archive/'shape.json'),metadata=metric,fixed_poses=fixed,
        capture_center=[0.]*3,capture_radius=2.25,reservoir_density=0.,depletant_radius=.1)
    write(archive/'config.json',cfg)
    chart=dict(angular_length=1.,means=[[0.]*6],weights=[1.],
        covariances=[np.eye(6).tolist()],anchors=[dict(position=[0.]*3,rotation=np.eye(3).tolist())])
    region=dict(gaussian_chart=chart,mahalanobis_radius=3.,minimum_original_q=1.,maximum_original_q=2.,
        physical_fixed_neighbors=fixed,fixed_neighbor=fixed[0],physical_metric=metric,
        capture_center=cfg['capture_center'],capture_radius=cfg['capture_radius'],activity=0.,depletant_radius=.1,
        shape_sha256=shape_sha)
    if flags is not None:region.update(flags)
    write(archive/'region.json',region)
    (archive/'latent-region-normalizer').write_bytes(b'synthetic audit fixture; never executable\n')
    directory=root/'runs/r00';(directory/'provenance').mkdir(parents=True)
    for source,dest in [('config.json','input-config.json'),('shape.json','shape.json'),('region.json','region.json')]:
        (directory/'provenance'/dest).write_bytes((archive/source).read_bytes())
    (directory/'provenance/source-bundle.json').write_bytes(b'{}\n')
    logv=shell_log_volume(region);window=original_q_window(region)
    manifest=dict(schema='uniform-latent-region-normalizer-v2',samples=6,seed=727197,
        cloud_replicates=2,activity=0.,lambda_=1.,lambda_ratio=64.,config_sha256=sha(archive/'config.json'),
        region_sha256=sha(archive/'region.json'),shape_sha256=shape_sha,
        source_bundle_sha256=sha(directory/'provenance/source-bundle.json'),
        executable_sha256=sha(archive/'latent-region-normalizer'),latent_radius=3.,log_latent_ball_volume=logv,
        minimum_original_q=1.,maximum_original_q=2.,minimum_latent_radius=0.,physical_fixed_neighbors=fixed,
        chart_anchor=fixed[0],log_latent_shell_volume=logv)
    manifest['lambda']=manifest.pop('lambda_')
    if flags is not None:manifest.update(flags)
    rows=[];logs=[]
    for i,q in enumerate([.5,1.,1.25,1.5,2.,2.5]):
        capture=q<=2.25;hard=capture and q!=1.5
        selected=original_q_contains(q,window);valid=hard and selected
        logj=-2*math.log(math.pi);weight=logv+logj
        row=dict(draw=i,latent=[q,0.,0.,0.,0.,0.],latent_radius=q,pose=pose(q),q=q,
            log_physical_jacobian=logj,capture_valid=capture,hard_valid=hard,region_valid=selected,
            log_importance_weight=weight if valid else None,log_hard_weight=weight if valid else None,
            clouds=[dict(log_weight=0.,lower_volume=0.,overlap_points=0),dict(log_weight=0.,lower_volume=0.,overlap_points=0)] if valid else [])
        rows.append(row)
        if valid:logs.append(weight)
    (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    estimate=None if not logs else logs[0]+math.log(len(logs)/len(rows))
    summary=dict(complete=True,samples=6,manifest=manifest,samples_sha256=sha(directory/'samples.jsonl'),
        estimates=dict(region=dict(logQ=estimate)),sampler_cpu_seconds=0.,maximum_backmap_error=0.)
    write(directory/'manifest.json',manifest);write(directory/'summary.json',summary)
    campaign=dict(region_sha256=sha(archive/'region.json'),archive_sha256={p.name:sha(p) for p in archive.iterdir()},
        lambda_ratio=64.,cloud_replicates=2,jobs=[dict(id='r00',seed=727197,samples=6,directory=str(directory))])
    write(root/'manifest.json',campaign)
    return region,manifest,summary,rows,directory


class LatentRegionQWindowControls(unittest.TestCase):
    def test_missing_flags_are_exact_legacy_predicate(self):
        for region in [dict(minimum_original_q=1.),dict(minimum_original_q=1.,maximum_original_q=2.)]:
            window=original_q_window(region)
            for q in [.5,1.,math.nextafter(1.,2.),1.5,math.nextafter(2.,1.),2.,2.5]:
                self.assertEqual(original_q_contains(q,window),region['minimum_original_q']<=q<=region.get('maximum_original_q',math.inf))

    def test_open_endpoints_without_epsilon_shell(self):
        region=dict(minimum_original_q=1.,maximum_original_q=2.,minimum_original_q_inclusive=False,maximum_original_q_inclusive=False)
        window=original_q_window(region)
        for q,expected in [(1.,False),(2.,False),(math.nextafter(1.,2.),True),(math.nextafter(2.,1.),True),(.999,False),(2.001,False)]:
            self.assertEqual(original_q_contains(q,window),expected)
        for name in ('minimum_original_q_inclusive','maximum_original_q_inclusive'):
            for bad in (None,0,1,'false'):
                with self.subTest(name=name,bad=bad),self.assertRaises(AssertionError):original_q_window(dict(region,**{name:bad}))

    def test_manifest_omissions_preserve_legacy_and_cannot_erase_open_flags(self):
        region=dict(minimum_original_q=1.,maximum_original_q=2.)
        validate_manifest_q_window(region,region,original_q_window(region))
        changed=dict(region,minimum_original_q_inclusive=False)
        with self.assertRaises(AssertionError):validate_manifest_q_window(region,changed,original_q_window(changed))
        with self.assertRaises(AssertionError):validate_manifest_q_window(dict(region,maximum_original_q=3.),region,original_q_window(region))

    def test_stream_open_interval_retains_full_draw_count_and_drops_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);_,_,_,rows,_=fixture(path,dict(minimum_original_q_inclusive=False,maximum_original_q_inclusive=False))
            with contextlib.redirect_stdout(io.StringIO()):result=analyze(path)
            self.assertEqual(result['estimate']['draws'],6);self.assertEqual(result['estimate']['nonzero'],1)
            self.assertEqual(result['independently_reconstructed_poses'],6)
            self.assertAlmostEqual(result['estimate']['logQ'],rows[2]['log_importance_weight']-math.log(6))
            self.assertFalse(result['original_q_window']['lower_inclusive']);self.assertFalse(result['original_q_window']['upper_inclusive'])

    def test_explicit_inclusive_flags_match_old_v2_estimates(self):
        results=[]
        for flags in [None,dict(minimum_original_q_inclusive=True,maximum_original_q_inclusive=True)]:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d);fixture(path,flags)
                with contextlib.redirect_stdout(io.StringIO()):results.append(analyze(path))
        self.assertEqual(results[0]['estimate'],results[1]['estimate'])
        self.assertEqual(results[0]['hard_region'],results[1]['hard_region'])
        self.assertEqual(results[0]['estimate']['nonzero'],3)
        self.assertNotIn('original_q_window',results[0])

    def test_rejects_manifest_and_boundary_tampering(self):
        for mutation in ('summary_manifest','both_manifests','row_boundary'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                path=Path(d);_,manifest,summary,rows,directory=fixture(path,dict(minimum_original_q_inclusive=False,maximum_original_q_inclusive=False))
                if mutation=='summary_manifest':summary['manifest']['minimum_original_q_inclusive']=True
                elif mutation=='both_manifests':
                    summary['manifest']['maximum_original_q_inclusive']=True
                    write(directory/'manifest.json',summary['manifest'])
                else:
                    rows[1]['region_valid']=True
                    (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    summary['samples_sha256']=sha(directory/'samples.jsonl')
                write(directory/'summary.json',summary)
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(AssertionError):analyze(path)


if __name__=='__main__':unittest.main()
