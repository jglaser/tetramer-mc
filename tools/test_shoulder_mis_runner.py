"""Frozen quota, source, seed, and geometric-preflight controls; fake executables only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from prepare_cayley_rms_cover import derive_model
import run_shoulder_mis_campaign as runner


def pose(position):return dict(position=position,orientation=[1.,0.,0.,0.])


def fixture(root,extra=()):
    shape=root/'shape.json';runner.write(shape,dict(atoms=[dict(center=[0.,0.,0.],radius=.1)]))
    fixed=[pose([10.,0.,0.]),pose([0.,10.,0.])]
    metric=dict(native_poses=[pose([0.,0.,0.])],rigid_members=[pose([3*x,3*y,3*z]) for x,y,z in
        [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]],member_error_scale=1.,angle_error_scale_deg=30.)
    cfg=dict(shape=str(shape),fixed_poses=fixed,capture_center=[0.,0.,0.],capture_radius=18.,
        depletant_radius=1.5,reservoir_density=.035,poisson_lambda_ratio=64.,metadata=metric)
    config=root/'config.json';runner.write(config,cfg)
    chart,proof=derive_model(metric,fixed[0],runner.sha(shape),2.)
    region=dict(gaussian_chart=chart,shape_sha256=runner.sha(shape),fixed_neighbor=fixed[0],physical_fixed_neighbors=fixed,
        capture_center=cfg['capture_center'],capture_radius=cfg['capture_radius'],activity=.035,depletant_radius=1.5,
        physical_metric=metric,mahalanobis_radius=proof['mahalanobis_radius'],minimum_original_q=1.,maximum_original_q=2.,
        minimum_original_q_inclusive=False,maximum_original_q_inclusive=False)
    region_path=root/'region.json';runner.write(region_path,region)
    model_path=root/'model.json';runner.write(model_path,chart)
    binary=root/'fake-kernel'
    binary.write_text(f'#!{sys.executable}\n'+'''import json,sys
from pathlib import Path
if '--help' in sys.argv:
    print('--q-min --q-max --q-lower-open --q-upper-open --model --region --cloud-replicates')
    raise SystemExit
def arg(name):return sys.argv[sys.argv.index(name)+1]
cfg=json.loads(Path(arg('--config')).read_text())
assert len(cfg['fixed_poses'])==2 and len(json.loads(Path(cfg['shape']).read_text())['atoms'])==1
out=Path(arg('--out'));out.mkdir(parents=True)
(out/'summary.json').write_text(json.dumps(dict(complete=True,samples=int(arg('--samples')),seed=int(arg('--seed')))))
''')
    binary.chmod(0o755)
    args=runner.parser().parse_args(['--out',str(root/'campaign'),'--config',str(config),'--model',str(model_path),
        '--region',str(region_path),'--native-binary',str(binary),'--latent-binary',str(binary),
        '--populations','2','--n-guide','3','--n-cover','5','--workers','2',
        '--guide-seed-base','123','--cover-seed-base','100023','--prepare-only',*extra])
    return args,cfg,region


class ShoulderMisRunnerTests(unittest.TestCase):
    def test_frozen_quotas_commands_physics_and_seal(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d);args,cfg,_=fixture(source);root,manifest=runner.prepare(args)
            runner.verify_archive(root,manifest)
            self.assertEqual(manifest['allocation'],dict(guide=3,cover=5,total=8,alpha_guide=3/8,alpha_cover=5/8))
            self.assertEqual(manifest['total_unconditional_draws'],16)
            self.assertEqual(manifest['population_count'],2);self.assertEqual(len(manifest['jobs']),4)
            self.assertEqual({j['seed'] for j in manifest['jobs']},{123,1132,100023,101032})
            self.assertEqual(manifest['physical'],{k:cfg[k] for k in runner.PHYSICAL_KEYS})
            self.assertEqual(manifest['q_window'],dict(minimum=1.,maximum=2.,lower_inclusive=False,upper_inclusive=False))
            self.assertEqual(runner.read(root/'provenance/input-config.json'),cfg)
            effective=runner.read(root/'provenance/config.json')
            self.assertEqual(effective['shape'],str(root/'provenance/shape.json'))
            self.assertEqual({k:v for k,v in effective.items() if k!='shape'},{k:v for k,v in cfg.items() if k!='shape'})
            for population in manifest['populations']:
                for family,quota in [('guide',3),('cover',5)]:
                    job=population[family];command=job['command']
                    self.assertEqual(job['samples'],quota)
                    self.assertEqual(command[command.index('--config')+1],str(root/'provenance/config.json'))
                    self.assertEqual(command[command.index('--samples')+1],str(quota))
                    self.assertEqual(command[command.index('--cloud-replicates')+1],'2')
                self.assertIn('--q-lower-open',population['guide']['command'])
                self.assertIn('--q-upper-open',population['guide']['command'])
            self.assertFalse(any((root/'runs').iterdir()))
            for name in ['prepare_cayley_rms_cover.py','prepare_native_confirmation_atlas.py','prepare_smc_normalizer_atlas.py']:
                self.assertIn(name,manifest['archive_sha256'])

    def test_preflight_rejects_mismatched_target_or_incomplete_cover(self):
        mutations=[lambda cfg,r:r.update(physical_fixed_neighbors=r['physical_fixed_neighbors'][:1]),
            lambda cfg,r:r.update(activity=.04),lambda cfg,r:r.update(minimum_original_q_inclusive=True),
            lambda cfg,r:r.update(mahalanobis_radius=r['mahalanobis_radius']*.9),
            lambda cfg,r:r['gaussian_chart']['covariances'][0][0].__setitem__(0,.9),
            lambda cfg,r:r.update(minimum_mahalanobis_radius=.1)]
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory() as d:
                path=Path(d);args,cfg,region=fixture(path);mutate(cfg,region);runner.write(args.region,region)
                with self.assertRaises(ValueError):runner.prepare(args)
                self.assertFalse(args.out.exists())

    def test_rejects_invalid_quotas_seed_overlap_and_binary_hash(self):
        for extra in [('--n-guide','0'),('--n-guide','1'),('--n-cover','1'),('--cover-seed-base','123'),('--guide-seed-base','-1'),
            ('--expected-native-sha256','0'*64)]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as d:
                args,_,_=fixture(Path(d),extra)
                with self.assertRaises(ValueError):runner.prepare(args)
                self.assertFalse(args.out.exists())

    def test_frozen_execution_ignores_changed_sources_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as d:
            args,_,_=fixture(Path(d));root,manifest=runner.prepare(args)
            args.config.write_text('{}');(Path(d)/'shape.json').write_text('{}');args.model.write_text('{}')
            runner.execute(root)
            status=runner.read(root/'runner-status.json')
            self.assertTrue(status['success']);self.assertTrue(status['complete']);self.assertFalse(status['running'])
            self.assertEqual(len(status['jobs']),4)
            for job in manifest['jobs']:
                summary=runner.read(Path(job['output'])/'summary.json')
                self.assertEqual(summary['samples'],job['samples']);self.assertEqual(summary['seed'],job['seed'])
            with self.assertRaises(ValueError):runner.execute(root)
            with self.assertRaises(ValueError):runner.prepare(args)

    def test_seal_and_archive_modifications_prevent_launch(self):
        for mutation in ('manifest','shape','region','output'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                args,_,_=fixture(Path(d));root,_=runner.prepare(args)
                if mutation=='manifest':(root/'manifest.json').write_text((root/'manifest.json').read_text()+' ')
                elif mutation=='output':(root/'runs/r00/guide').mkdir(parents=True)
                else:(root/'provenance'/f'{mutation}.json').write_text('{}')
                with self.assertRaises(ValueError):runner.execute(root)
                self.assertEqual(runner.read(root/'runner-status.json')['jobs'],{})


if __name__=='__main__':unittest.main()
