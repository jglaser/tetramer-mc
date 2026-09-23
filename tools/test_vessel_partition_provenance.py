"""Synthetic saved-label pipeline, including tampered provenance rejection."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from analyze_r4_smc_control import sha,write
from test_physical_latent_guide import fixture,independent_pose
from test_vessel_contact_partition import mass
from vessel_contact_partition import PRIMARY,SUPPORTS,enrich


def package(root):
    region,_,lower=fixture(.5,coupled=True)
    fixed=region['fixed_neighbor']
    metric=dict(native_poses=[fixed],rigid_members=[dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])],member_error_scale=3.,angle_error_scale_deg=90.)
    region.update(physical_fixed_neighbors=[fixed],physical_metric=metric,capture_center=[0.,0.,0.],capture_radius=1000.,minimum_original_q=0.,minimum_original_q_inclusive=True)
    config=dict(fixed_poses=region['physical_fixed_neighbors'],metadata=metric)
    source=root/'population';source.mkdir();write(source/'config.json',config)
    manifest=dict(samples=3,shape_sha256=region['shape_sha256']);write(source/'manifest.json',manifest)
    rows=[];labels=[];estimates={}
    for i,u in enumerate(([.2,0,0,0,0,0],[6.,0,0,0,0,0],[.3,0,0,0,0,0])):
        valid=i!=2;classes=dict(total=valid,**{p:valid and PRIMARY[i]==p for p in PRIMARY})
        rows.append(dict(draw=i,pose=independent_pose(u,region,lower)[0],hard_valid=valid,
                         log_importance_weight=math.log(i+2) if valid else None,log_hard_weight=0. if valid else None))
        labels.append(dict(draw=i,classes=classes))
    for name in ('total',*PRIMARY):
        estimates[name]={kind:mass([math.exp(r[key]) if label['classes'][name] else 0 for r,label in zip(rows,labels)])
            for kind,key in [('Qz','log_importance_weight'),('Q0','log_hard_weight')]}
    (source/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    regions={}
    for name in SUPPORTS:
        path=root/(name+'.json');write(path,region);regions[name]=path
    audit=root/'audit';audit.mkdir();(audit/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
    report=dict(schema='full-vessel-baseline-audit-v1',complete=True,population=str(source),manifest=manifest,
        native_binding={'definition_sha256':'synthetic'},raw_sample_binding={'sha256':sha(source/'samples.jsonl')},
        source_sha256={str(source/'config.json'):sha(source/'config.json')},reporting_guide_binding={'region_sha256':sha(regions['current_R4'])},estimates=estimates)
    write(audit/'analysis.json',report);write(audit/'freeze.json',dict(files={p.name:sha(p) for p in audit.iterdir() if p.is_file()}))
    return audit/'analysis.json',regions,source


class PartitionProvenanceTests(unittest.TestCase):
    def test_reuses_labels_all_attempts_and_exterior_mass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit,regions,_=package(root);result=enrich(audit,regions,root/'result')
            self.assertEqual(result['samples'],3);self.assertEqual(result['invalid_draws'],1)
            self.assertAlmostEqual(result['estimates']['total']['Qz']['logQ'],math.log(5/3))
            self.assertAlmostEqual(result['estimates']['outside_measured_pockets']['Qz']['logQ'],0.)
            self.assertAlmostEqual(result['estimates'][PRIMARY[0]+':inside_current_R4']['Qz']['logQ'],math.log(2/3))
            self.assertEqual(len((root/'result/labels.jsonl').read_text().splitlines()),3)

    def test_changed_effective_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit,regions,source=package(root);write(source/'config.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):enrich(audit,regions,root/'result')

    def test_wrong_current_chart_under_the_same_reporting_name_rejected(self):
        from analyze_r4_smc_control import read
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit,regions,_=package(root);r=copy.deepcopy(read(regions['current_R4']));r['mahalanobis_radius']=3.;write(regions['current_R4'],r)
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):enrich(audit,regions,root/'result')


if __name__=='__main__':unittest.main()
