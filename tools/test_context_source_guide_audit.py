"""Small complete-row/journal controls; no protein geometry or random draws."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_context_source_guide import (scalar_row, audit_generation, audit_events,
                                       role_seed, PANEL_ORDINALS)
from audit_context_candidate_bank import MapDensityReference
from source_guide_reference import SourceDensity, atlas_decode, mixture_log_density, pose
from test_context_candidate_bank_audit import preflight_row, reference_tokens, model
from test_source_guide_reference import spec


ID = pose([0., 0., 0.], np.eye(3))


def row(ordinal=0, valid=True):
    source = SourceDensity(spec(), ID, ID)
    tokens = [[16, 77, 'anchor', 'mobile']]+reference_tokens()
    result = preflight_row(ordinal, tokens, valid)
    d = result['density']; d['log_source'] = source.evaluate(ID); d['log_source_status'] = 'finite'
    d['log_q'] = mixture_log_density(d['log_u'], d['log_g'], d['log_source'])
    return result, source


class ContextSourceGuideAuditTests(unittest.TestCase):
    def test_complete_Q_source_density_and_hard_zero(self):
        for valid in (False, True):
            r, source = row(valid=valid)
            scalar_row(r, 0, 2., reference_tokens(), source, [.5, .25, .25])
            for field, value in [('log_q', r['density']['log_q']+.1),
                                 ('log_source', r['density']['log_source']+.1)]:
                bad = copy.deepcopy(r);bad['density'][field] = value
                with self.assertRaises(ValueError):
                    scalar_row(bad, 0, 2., reference_tokens(), source, [.5, .25, .25])
        r, source = row(valid=False);r['region'] = 'unbound'
        with self.assertRaises(ValueError):scalar_row(r, 0, 2., reference_tokens(), source, [.5, .25, .25])

    def test_generation_trace_and_role_identity_for_every_branch(self):
        cfg = dict(seed=98765, population=7, identity=dict(width_index=1, stream=3))
        digest = 'e'*64; source = SourceDensity(spec(), ID, ID); atlas = MapDensityReference(model())
        roles = {r:role_seed(cfg, digest, 0, r) for r in ('component','label','latent','uniform','cloud0','cloud1')}
        self.assertEqual(len(set(roles.values())), 6)
        for branch, coin in [('uniform', .2), ('context', .6), ('source', .9)]:
            item = dict(identity=cfg['identity'], population=7, component_uniform=coin, branch=branch,
                        role_seeds=roles, selected_virtual_label=None, latent=None,
                        cube_uniforms=None, quaternion_normals=None)
            if branch == 'uniform':
                item.update(cube_uniforms=[.1,.2,.3], quaternion_normals=[1.,0.,0.,0.],
                            proposed_pose=pose([-1.6,-1.2,-.8], np.eye(3)))
            else:
                item['latent'] = [.1,-.2,.3,-.4,.5,-.6]
                if branch == 'context':
                    item['selected_virtual_label'] = 1
                    item['proposed_pose'] = atlas_decode(atlas, ID, 1, item['latent'])
                else:item['proposed_pose'] = source.decode(item['latent'])
            audit_generation(item, cfg, digest, 0, 2., source, atlas, ID)
            bad = copy.deepcopy(item);bad['proposed_pose']['position'][0] += .01
            with self.assertRaises(ValueError):audit_generation(bad, cfg, digest, 0, 2., source, atlas, ID)
            bad = copy.deepcopy(item);bad['role_seeds']['cloud0'] = roles['cloud1']
            with self.assertRaises(ValueError):audit_generation(bad, cfg, digest, 0, 2., source, atlas, ID)

    def test_exhaustive_journal_and_no_replacement_panel(self):
        rows = [row(0, True)[0], row(1, False)[0]]
        events = [dict(kind='setup_begun'), dict(kind='source_begun'),
                  dict(kind='source_complete', patches=rows[0]['patches'])]
        for i, r in enumerate(rows):
            events.extend([dict(kind='candidate_begun', ordinal=i),
                           dict(kind='candidate_generated', ordinal=i, input=r['input']),
                           dict(kind='density_complete', ordinal=i, density=r['density']),
                           dict(kind='geometry_complete', ordinal=i, wall_valid=r['actual']['wall_valid'], core_valid=r['actual']['core_valid'])])
            if r['actual']['physical_valid']:
                events.extend([dict(kind='patches_complete', ordinal=i, patches=r['patches'], region=r['region']),
                               dict(kind='envelope_complete', ordinal=i, envelope=r['envelope'])])
            events.append(dict(kind='candidate_complete', ordinal=i))
        def write(path, items):
            path.write_text(''.join(json.dumps(dict(e, event_index=i))+'\n' for i,e in enumerate(items)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'events.jsonl';write(path, events)
            self.assertEqual(audit_events(path, rows, reference_tokens()), len(events))
            write(path, events[:-1])
            with self.assertRaises(ValueError):audit_events(path, rows, reference_tokens())
            extra = events+[dict(kind='cloud_progress')];write(path, extra)
            with self.assertRaises(ValueError):audit_events(path, rows, reference_tokens())
        self.assertEqual(PANEL_ORDINALS, tuple(range(0,512,16)))
        self.assertEqual(len(PANEL_ORDINALS),32)


if __name__ == '__main__':unittest.main()
