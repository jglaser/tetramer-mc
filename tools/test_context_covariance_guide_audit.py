"""Deterministic covariance-guide audit controls; no protein or random queries."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from audit_context_covariance_guide import (
    PANEL_ORDINALS, audit_events, audit_generation, scalar_row,
    sha, validate_admitted_config, validate_contract,
)
from audit_context_source_guide import role_seed
from source_guide_reference import SourceDensity, compose, mixture_log_density, pose
from test_context_source_guide_audit import row as legacy_row


def chart():
    lower = np.diag([.2, .3, .4, .12, .15, .18])
    lower[3, 0], lower[4, 1], lower[5, 2] = .04, -.05, .06
    lower[4, 0], lower[5, 3] = .02, .03
    return dict(angular_length=30., covariance=(lower@lower.T).tolist(),
                explicit_gaussian=dict(schema='source-gaussian-v1',
                    mean=[.03, -.04, .02, .01, -.02, .03],
                    provenance='synthetic frozen training streams0,1'))


def frozen_fixture(directory):
    """An exact local metadata closure, never a fallback physical fixture."""
    root = Path(directory); bindings = {}
    def save(name, value):
        path = root/name; path.write_text(json.dumps(value, allow_nan=False))
        bindings[str(path.resolve())] = sha(path)
        return dict(path=str(path.resolve()), sha256=sha(path))
    reduction = save('reduction.json', dict(complete=True, passed=True))
    guides = {}
    for arm in ('full', 'diagonal'):
        spec = chart()
        if arm == 'diagonal':
            spec['covariance'] = np.diag(np.diag(spec['covariance'])).tolist()
        guides[arm] = save(arm+'.json', dict(schema='context-covariance-frozen-guide-v1',
            arm=arm, source_chart=spec, training_streams=[0, 1], heldout_streams=[2, 3],
            reduction_sha256=reduction['sha256']))
    allocation = save('allocation.json', dict(schema='context-source-covariance-geometry-allocation-v1',
        arms=['full', 'diagonal'], streams_per_arm=4, draws_per_population=2048,
        populations=8, total_draws=16384, clouds=0, mixture=[.5, .25, .25],
        training_streams=[0, 1], heldout_streams=[2, 3],
        panel_ordinals=list(PANEL_ORDINALS), guides=guides, reduction=reduction))
    protocol = dict(schema='context-covariance-guide-audit-v1',
        panel_ordinals=list(PANEL_ORDINALS), expected_populations=8,
        draws_per_population=2048, input_sha256=bindings, allocation=allocation,
        guides=guides, reduction=reduction)
    return protocol, allocation['sha256']


class CovarianceGuideAuditTests(unittest.TestCase):
    def test_nonzero_mean_full_covariance_density_and_common_frame(self):
        anchor = pose([2., -.5, 1.], Rotation.from_rotvec([.3, -.2, .1]).as_matrix())
        origin = pose([-.2, .4, 3.], Rotation.from_rotvec([-.4, .1, .2]).as_matrix())
        spec = chart(); reference = SourceDensity(spec, origin, anchor)
        mean = np.asarray(spec['explicit_gaussian']['mean'])
        shift = pose([4., -3., 2.], Rotation.from_rotvec([.2, .6, -.3]).as_matrix())
        transformed = SourceDensity(spec, compose(shift, origin), compose(shift, anchor))
        for latent in (np.zeros(6), np.array([.4, -.3, .2, -.1, .5, -.6])):
            x = mean+reference.lower@latent
            c = x[3:]/reference.ell
            log_haar = -3*math.log(reference.ell)-2*math.log(math.pi)-2*math.log1p(c@c)
            expected = multivariate_normal.logpdf(x, mean=mean, cov=spec['covariance'])-log_haar
            value = reference.decode(latent)
            self.assertAlmostEqual(reference.evaluate(value), expected, places=9)
            self.assertAlmostEqual(transformed.evaluate(compose(shift, value)), expected, places=9)
        # A nonzero mean changes the decoded pose even at zero standard latent.
        self.assertGreater(np.linalg.norm(np.asarray(reference.decode(np.zeros(6))['position'])-origin['position']), .01)

    def test_exact_allocation_and_fixed_panel_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, expected = frozen_fixture(directory)
            allocation, guides = validate_contract(protocol, expected)
            self.assertEqual(allocation['total_draws'], 16384)
            self.assertEqual(set(guides), {'full', 'diagonal'})
            self.assertEqual(len(PANEL_ORDINALS), 64)
            for key, value in [('expected_populations', 7), ('draws_per_population', 1024),
                               ('panel_ordinals', list(range(1, 2048, 32)))]:
                bad = copy.deepcopy(protocol); bad[key] = value
                with self.assertRaises(ValueError): validate_contract(bad, expected)
            with self.assertRaises(ValueError): validate_contract(protocol, '0'*64)
            changed = copy.deepcopy(allocation); changed['clouds'] = 2
            path = Path(protocol['allocation']['path']); path.write_text(json.dumps(changed))
            protocol['allocation']['sha256'] = sha(path)
            protocol['input_sha256'][str(path)] = sha(path)
            with self.assertRaises(ValueError): validate_contract(protocol, expected)

    def test_changed_guide_cannot_replace_frozen_reduction_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, expected = frozen_fixture(directory)
            path = Path(protocol['guides']['full']['path']); value = json.loads(path.read_text())
            value['source_chart']['explicit_gaussian']['mean'][0] += .01
            path.write_text(json.dumps(value))
            protocol['guides']['full']['sha256'] = sha(path)
            protocol['input_sha256'][str(path)] = sha(path)
            with self.assertRaises(ValueError): validate_contract(protocol, expected)

    def test_admission_uses_actual_config_digest_exact_mean_and_covariance(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, expected = frozen_fixture(directory)
            _, guides = validate_contract(protocol, expected)
            digest = 'b'*64
            cfg = dict(population=1, seed=901, identity=dict(arm='full', stream=1,
                guide_sha256=guides['full']['asset_sha256']), source_chart=guides['full']['source_chart'])
            entry = dict(arm='full', stream=1, population=1, config_sha256=digest)
            used = set(); self.assertEqual(validate_admitted_config(entry, cfg, digest, guides, used), ('full', 1))
            with self.assertRaises(ValueError): validate_admitted_config(entry, cfg, digest, guides, used)
            with self.assertRaises(ValueError): validate_admitted_config(entry, cfg, 'c'*64, guides, set())
            bad = copy.deepcopy(cfg); bad['source_chart']['explicit_gaussian']['mean'][0] += .001
            with self.assertRaises(ValueError): validate_admitted_config(entry, bad, digest, guides, set())

    def test_source_decode_full_Q_and_hard_invalid_denominator(self):
        origin = pose([0., 0., 0.], np.eye(3)); source = SourceDensity(chart(), origin, origin)
        cfg = dict(seed=51, population=5, identity=dict(arm='diagonal', stream=1))
        digest = 'a'*64; latent = [.2, -.3, .4, -.2, .1, .3]
        item = dict(identity=cfg['identity'], population=5, ordinal=0,
            branch='source', component_uniform=.9, latent=latent,
            selected_virtual_label=None, cube_uniforms=None, quaternion_normals=None,
            proposed_pose=source.decode(latent),
            role_seeds={role:role_seed(cfg, digest, 0, role) for role in
                ('component', 'label', 'latent', 'uniform', 'cloud0', 'cloud1')})
        audit_generation(item, cfg, digest, 0, 2., source, None, origin)
        row, _ = legacy_row(valid=False); row['input'] = item
        row['density']['log_source'] = source.evaluate(item['proposed_pose'])
        row['density']['log_q'] = mixture_log_density(row['density']['log_u'],
            row['density']['log_g'], row['density']['log_source'])
        scalar_row(row, 0, 2., [], source, [.5, .25, .25])
        self.assertTrue(row['physical_zero'])
        bad = copy.deepcopy(row); bad['density']['log_q'] += .2
        with self.assertRaises(ValueError): scalar_row(bad, 0, 2., [], source, [.5, .25, .25])

    def test_truncated_journal_and_extra_cloud_fail_with_invalid_row_retained(self):
        from test_context_candidate_bank_audit import reference_tokens
        rows = [legacy_row(0, True)[0], legacy_row(1, False)[0]]
        events = [dict(kind='setup_begun'), dict(kind='source_begun'),
                  dict(kind='source_complete', patches=rows[0]['patches'])]
        for i, row in enumerate(rows):
            events.extend([dict(kind='candidate_begun', ordinal=i),
                dict(kind='candidate_generated', ordinal=i, input=row['input']),
                dict(kind='density_complete', ordinal=i, density=row['density']),
                dict(kind='geometry_complete', ordinal=i, wall_valid=row['actual']['wall_valid'], core_valid=row['actual']['core_valid'])])
            if row['actual']['physical_valid']:
                events.extend([dict(kind='patches_complete', ordinal=i, patches=row['patches'], region=row['region']),
                    dict(kind='envelope_complete', ordinal=i, envelope=row['envelope'])])
            events.append(dict(kind='candidate_complete', ordinal=i))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'events.jsonl'
            def write(values): path.write_text(''.join(json.dumps(dict(v, event_index=i))+'\n' for i,v in enumerate(values)))
            write(events); self.assertEqual(audit_events(path, rows, reference_tokens()), len(events))
            write(events[:-1])
            with self.assertRaises(ValueError): audit_events(path, rows, reference_tokens())
            write(events+[dict(kind='cloud_progress')])
            with self.assertRaises(ValueError): audit_events(path, rows, reference_tokens())


if __name__ == '__main__': unittest.main()
