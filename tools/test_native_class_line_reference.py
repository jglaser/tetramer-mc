"""Synthetic-only reference checks. No protein geometry or physical draws."""
import copy
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import norm

import native_class_line_reference as line
from native_contact_regions import NativeContactRegions, make_pose
from test_hard_free_line_reference import setup as hard_setup

IDENTITY = make_pose([0., 0., 0.], np.eye(3))


def observer(atoms=None, radii=None, residues=None, target=(3., 0., 0.)):
    """Complete toy classifier data; actual classify_pair is unmodified."""
    model = NativeContactRegions.__new__(NativeContactRegions)
    model.member_positions = np.array([[0., 10*i, 0.] for i in range(4)])
    model.member_rotations = np.tile(np.eye(3), (4, 1, 1))
    model.atoms = np.array([[0., 0., 0.]] if atoms is None else atoms)
    model.radii = np.array([.25]*len(model.atoms) if radii is None else radii)
    model.residues = np.array([0]*len(model.atoms) if residues is None else residues)
    model.residue_count = int(model.residues.max())+1
    model.tree = cKDTree(model.atoms)
    model.references = {'good': dict(label='good', family='toy', position=np.array(target),
        rotation=np.eye(3), native_residue_pairs={0}),
        'bad': dict(label='bad', family='toy', position=np.array(target),
        rotation=np.eye(3), native_residue_pairs={10**6})}
    model.motifs = [dict(id=0, member_contacts=[dict(member_i=0, member_j=0, directed_class='bad'),
                                            dict(member_i=1, member_j=1, directed_class='good')])]
    model.motif_positions = np.array([target], float)
    model.motif_rotations = np.array([np.eye(3)])
    return model


def hard_valid_monomers(model, anchor, moving):
    # Tests only compare the observer in its documented hard-valid domain.
    from native_contact_regions import pose_arrays
    ta, ra = pose_arrays(anchor); tm, rm = pose_arrays(moving)
    for i in range(4):
        fixed = (model.atoms @ model.member_rotations[i].T+model.member_positions[i]) @ ra.T+ta
        for j in range(4):
            mobile = (model.atoms @ model.member_rotations[j].T+model.member_positions[j]) @ rm.T+tm
            if np.any(np.linalg.norm(fixed[:, None, :]-mobile[None, :, :], axis=2)
                      < model.radii[:, None]+model.radii[None, :]-1e-12):
                return False
    return True


class IntervalTests(unittest.TestCase):
    def test_endpoint_union_difference_and_singleton(self):
        full = [line.interval(-1, 1)]
        self.assertEqual(line.difference(full, [line.interval(-1, 0, True, False),
                                               line.interval(0, 1, False, True)]), [line.interval(0, 0)])
        self.assertEqual(line.intersection(full, [line.interval(1, 2)]), [line.interval(1, 1)])
        self.assertEqual(line.intersection(full, [line.interval(1, 2, False, True)]), [])
        self.assertEqual(line.difference(full, full), [])

    def test_quadratics_strict_closed_constant_and_tangent(self):
        segment = [-4., 4.]
        self.assertEqual(line.distance_intervals([0, 2], [1, 0], 2, segment), [line.interval(0, 0)])
        self.assertEqual(line.distance_intervals([0, 2], [1, 0], 2, segment, inclusive=False), [])
        self.assertEqual(line.distance_intervals([0, 2], [0, 0], 2, segment), [line.interval(*segment)])
        self.assertEqual(line.distance_intervals([0, 2], [0, 0], 2, segment, inclusive=False), [])
        self.assertEqual(line.distance_intervals([1, 0], [2, 0], 2, segment), [line.interval(-1.5, .5)])
        for inclusive in (False, True):
            sets = line.distance_intervals([1, .3, -.4], [2, -.1, .2], 2, segment, inclusive=inclusive)
            for s in np.linspace(*segment, 1001):
                distance = np.linalg.norm(np.array([1, .3, -.4])+s*np.array([2, -.1, .2]))
                self.assertEqual(line.contains(sets, s), distance <= 2 if inclusive else distance < 2)

    def test_vectorized_leaf_union_matches_scalar_all_pairs(self):
        fixed = np.array([[0., 0., 0.], [1., .4, .8], [-2., .1, .3]])
        moving = np.array([[.2, -.4, .3], [1.2, .1, .7]])
        fr, mr, direction, segment = [.2, .7, .4], [.3, .5], [.7, .2, -.1], [-3., 4.]
        for inclusive in (False, True):
            for pairs in (None, [(0, 1), (2, 0)]):
                complete = [(i, j) for i in range(3) for j in range(2)] if pairs is None else pairs
                expected = line.union([v for i, j in complete for v in line.distance_intervals(
                    moving[j]-fixed[i], direction, fr[i]+mr[j]+.5, segment, inclusive=inclusive)])
                result = line.leaf_contact_intervals(fixed, moving, direction, fr, mr, .5, segment,
                                                     inclusive=inclusive, pairs=pairs)
                line.compare_intervals(result['intervals'], expected, 'Scalar versus vector roots', atol=1e-14)
                self.assertEqual(result['leaf_atom_pairs'], len(complete))

    def test_computational_chart_zero_sign_is_not_blas_factor_rounding(self):
        original = np.diag([1.5, .5, .7, .2, .3, .4])
        original[1, 0] = .15; original[3, 0] = .12; original[4, 1] = -.08
        covariance = original @ original.T
        lower, validation = line.chart_fp64_factor(covariance)
        self.assertEqual(lower[3, 1], 0.)
        self.assertNotEqual(np.linalg.cholesky(covariance)[3, 1], 0.)
        du = line.scalar_chart_solve(lower, np.eye(6)[1])
        self.assertEqual(du[3], 0.)
        self.assertEqual(line.orthant_intervals(np.zeros(6), du, 55, [-2., 2.]), [])
        self.assertLess(validation['relative_covariance_residual'], 1e-15)
        # Tiny genuinely represented coefficients are never clipped.
        lower[3, 1] = 1e-18
        self.assertLess(line.scalar_chart_solve(lower, np.eye(6)[1])[3], 0.)

    def test_correlated_whitened_orthants_and_zero_positive(self):
        u0 = np.array([0., 0., 1., 1., 1., 1.]); du = np.array([1., -2., 0., 0., 0., 0.])
        self.assertEqual(line.orthant_intervals(u0, du, 63, [-1., 1.]), [line.interval(0., 0.)])
        self.assertEqual(line.orthant_intervals(u0, du, 61, [-1., 1.]), [line.interval(0., 1., False, True)])
        self.assertEqual(line.orthant_intervals(u0, du, 62, [-1., 1.]), [line.interval(-1., 0., True, False)])
        for orthant in range(64):
            sets = line.orthant_intervals(u0, du, orthant, [-1., 1.])
            for s in [-1., -.1, 0., .1, 1.]:
                index = sum((1 << i) for i, value in enumerate(u0+du*s) if value >= 0)
                self.assertEqual(line.contains(sets, s), index == orthant)


class NativeGeometryTests(unittest.TestCase):
    def compare(self, model, origin, direction, segment, rotation=np.eye(3), anchors=(IDENTITY,)):
        result = line.NativeLineReference(model).all_anchors(anchors, origin, rotation, direction, segment)
        xs = list(np.linspace(*segment, 1001))
        for interval in result['intervals']:
            xs.extend([interval['lower'], interval['upper']])
        checked = 0
        for s in xs:
            moving = make_pose(np.asarray(origin)+s*np.asarray(direction), rotation)
            if not all(hard_valid_monomers(model, a, moving) for a in anchors):
                continue
            oracle = any(model.classify_pair(a, moving) for a in anchors)
            self.assertEqual(line.contains(result['intervals'], s), oracle, (s, result['intervals']))
            checked += 1
        self.assertGreater(checked, 0)
        return result

    def test_any_bond_not_every_bond_and_all_members(self):
        model = observer()
        result = self.compare(model, [0, 0, 0], [1, 0, 0], [-1, 6])
        self.assertEqual(result['intervals'], [line.interval(1., 2.5)])
        found = model.classify_pair(IDENTITY, make_pose([2, 0, 0], np.eye(3)))
        self.assertEqual(len(found), 1)
        self.assertEqual(len(found[0]['supporting_member_bonds']), 1)
        # Constant rotation changes member errors differently; the fourth
        # member must remain in the intersection despite the first passing.
        rot = Rotation.from_euler('z', 5, degrees=True).as_matrix()
        self.compare(model, [0, 0, 0], [1, 0, 0], [-1, 6], rotation=rot)
        self.assertFalse(line.contains(line.NativeLineReference(model).pair(IDENTITY, [0, 0, 0], rot, [1, 0, 0], [-1, 6])['intervals'], 2.))

    def test_wrong_residues_do_not_supply_native_contact(self):
        model = observer(atoms=[[0., 0., 0.], [0., 1., 0.]], residues=[0, 1])
        for ref in model.references.values():
            ref['native_residue_pairs'] = {10**6}
        result = self.compare(model, [0, 0, 0], [1, 0, 0], [.6, 6])
        self.assertEqual(result['intervals'], [])

    def test_body_and_monomer_angular_gates(self):
        model = observer()
        rot = Rotation.from_euler('x', 16, degrees=True).as_matrix()
        self.assertEqual(self.compare(model, [0, 0, 0], [1, 0, 0], [.6, 6], rotation=rot)['intervals'], [])
        model.references['good']['rotation'] = Rotation.from_euler('x', 21, degrees=True).as_matrix()
        self.assertEqual(self.compare(model, [0, 0, 0], [1, 0, 0], [.6, 6])['intervals'], [])

    def test_monomer_position_gate_is_required(self):
        model = observer()
        model.references['good']['position'] = np.array([-5., 0, 0])
        self.assertEqual(self.compare(model, [0, 0, 0], [1, 0, 0], [.6, 6])['intervals'], [])

    def test_both_anchors_and_overlapping_motifs_union(self):
        model = observer(); duplicate = copy.deepcopy(model.motifs[0]); duplicate['id'] = 17
        model.motifs.append(duplicate)
        model.motif_positions = np.vstack([model.motif_positions, model.motif_positions])
        model.motif_rotations = np.concatenate([model.motif_rotations, model.motif_rotations])
        anchors = [IDENTITY, make_pose([10., 0, 0], np.eye(3))]
        result = self.compare(model, [0, 0, 0], [1, 0, 0], [.6, 16], anchors=anchors)
        self.assertEqual(result['intervals'], [line.interval(1., 2.5), line.interval(11., 12.5)])
        self.assertEqual(len(result['anchors'][0]['motifs']), 2)

    def test_closed_native_tangent_and_strict_exclusion_difference(self):
        model = observer(target=(0., 2.5, 0.))
        result = self.compare(model, [0, 2.5, 0], [1, 0, 0], [-2, 2])
        self.assertEqual(result['intervals'], [line.interval(0., 0.)])
        contact = line.leaf_contact_intervals([[0, 0, 0]], [[0, 2.5, 0]], [1, 0, 0], [.25], [.25], 2., [-2, 2], inclusive=False)
        self.assertEqual(contact['intervals'], [])
        closed = line.leaf_contact_intervals([[0, 0, 0]], [[0, 2.5, 0]], [1, 0, 0], [.25], [.25], 2., [-2, 2], inclusive=True)
        self.assertEqual(closed['intervals'], result['intervals'])

    def test_complement_only_on_hard_free_and_contact_radius_distinction(self):
        native = [line.interval(1., 2.5)]
        hard_free = [line.interval(.5, 6)]
        contact = line.leaf_contact_intervals([[0, 0, 0]], [[0, 0, 0]], [1, 0, 0], [.25], [.25], 3., [0, 6], inclusive=False)['intervals']
        competing = line.difference(line.intersection(hard_free, contact), native)
        self.assertEqual(competing, [line.interval(.5, 1., True, False), line.interval(2.5, 3.5, False, False)])
        self.assertFalse(line.contains(competing, .2))
        self.assertFalse(line.contains(competing, 3.5))


class ConditionalLawTests(unittest.TestCase):
    def test_fallback_normalization_and_single_class_dominance(self):
        H, C = [line.interval(-2, 2)], [line.interval(-.1, .2)]
        for selected, floor, wanted in [(C, 1e-12, 'class'), ([], 1e-12, 'hard_free'), ([], .99, 'unconditional')]:
            result = line.conditional_branch(H, selected, .3, 1.2, floor, 0.)
            self.assertEqual(result['fallback'], wanted)
            domain = selected if wanted == 'class' else H if wanted == 'hard_free' else [line.interval(-20., 20.)]
            integral = sum(quad(lambda s: norm.pdf(s, .3, 1.2)*line.conditional_branch(H, selected, .3, 1.2, floor, s)['multiplier'], v['lower'], v['upper'])[0] for v in domain)
            self.assertAlmostEqual(integral, 1., places=11)
            if wanted != 'unconditional':
                for x in [0., .1]:
                    a = line.conditional_branch(H, selected, .3, 1.2, floor, x)
                    b = line.conditional_branch(H, H, .3, 1.2, floor, x)
                    self.assertGreaterEqual(a['multiplier'], b['multiplier'])

    def test_floor_equality_and_non_subset_rejected(self):
        H = [line.interval(-1., 1.)]; C = [line.interval(0., .2)]
        zc = float(line.interval_masses(C, 0., 1.))
        self.assertEqual(line.conditional_branch(H, C, 0., 1., zc, .1)['fallback'], 'hard_free')
        self.assertEqual(line.conditional_branch(H, C, 0., 1., np.nextafter(zc, 0.), .1)['fallback'], 'class')
        with self.assertRaisesRegex(ValueError, 'subset'):
            line.conditional_branch(H, [line.interval(3., 4.)], 0., 1., 1e-12, 3.5)

    def test_different_class_mixture_need_not_dominate_union(self):
        H = [line.interval(-4., 4.)]; C1 = [line.interval(-4., 0.)]; C2 = [line.interval(3., 4.)]
        # C1 is half the Normal mass but receives only 10% channel probability.
        f1 = line.conditional_branch(H, C1, 0., 1., 1e-12, -1.)['multiplier']
        f2 = line.conditional_branch(H, C2, 0., 1., 1e-12, -1.)['multiplier']
        h = line.conditional_branch(H, H, 0., 1., 1e-12, -1.)['multiplier']
        self.assertLess(.1*f1+.9*f2, h)

    def test_extreme_tail_and_tiny_interval_mass(self):
        for bounds in [(8., 8.01), (-9., -8.9), (.1, .1+1e-12), (0., 0.)]:
            mass = float(line.interval_masses([line.interval(*bounds)], 0., 1.))
            expected = quad(norm.pdf, *bounds, epsabs=1e-100, epsrel=1e-12)[0]
            self.assertAlmostEqual(mass/expected if expected else mass, 1. if expected else 0., places=10)

    def setup(self, axes=(0, 1, 2)):
        region, guide, config, shape = hard_setup(axes=axes)
        guide['schema'] = line.SCHEMA
        guide['class_channels'] = [{'class': 'hard_free', 'probability': .2},
                                   {'class': 'native', 'probability': .3},
                                   {'class': 'contact_without_native', 'probability': .5, 'orthant': 63}]
        config['depletant_radius'] = .4
        return region, guide, config, shape, observer()

    def test_near_unit_channel_probability_mass_is_normalized(self):
        args = self.setup()
        for c in args[1]['class_channels']: c['probability'] *= 1.+5e-13
        recon = line.Reconstructor(*args)
        self.assertAlmostEqual(sum(c['probability'] for c in recon.channels), 1., places=15)
        canonical = line.Reconstructor(*self.setup())
        self.assertAlmostEqual(recon.density(np.zeros(6))['log_density'], canonical.density(np.zeros(6))['log_density'], places=14)
        args[1]['class_channels'][0]['probability'] += 2e-12
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            line.Reconstructor(*args)

    def test_full_axis_component_channel_mixture_and_jacobian(self):
        args = self.setup(); r = line.Reconstructor(*args)
        u = np.array([.01, .01, .0, .1, .1, .1])
        result = r.density(u); parts = []
        for a in r.axes:
            for c in r.channels:
                reg, g, cfg, shape, model = copy.deepcopy(args)
                g['raw_translation_axes'] = [a]
                cc = copy.deepcopy(c); cc['probability'] = 1.; g['class_channels'] = [cc]
                parts.append(c['probability']*math.exp(line.Reconstructor(reg, g, cfg, shape, model).density(u)['log_density'])/len(r.axes))
        self.assertAlmostEqual(math.exp(result['log_density']), sum(parts), places=13)
        # The base independent reconstruction performs Schur conditioning and
        # exactly the same physical coordinate map/J independently of Rust.
        g = copy.deepcopy(args[1]); g['schema'] = 'defensive-hard-free-line-guide-v1'
        base = line.hard.Reconstructor(args[0], g, args[2], args[3])
        self.assertEqual(r.decode(u)[3], base.decode(u)[3])

    def test_summary_hard_free_density_all_beta_and_normalized_weight_cases(self):
        from summarize_native_class_line_probe import hard_free_log_density
        for alpha, beta in [(.5, 1.), (.5, .3), (.5, 0.), (1., 1.)]:
            args = self.setup()
            args[1]['class_channels'] = [{'class': 'hard_free', 'probability': 1.}]
            args[1]['defensive_uniform_shell_probability'] = alpha
            args[1]['conditional_probability'] = beta
            # The base fixture uses unnormalized weights 2 and 1.
            recon = line.Reconstructor(*args)
            for u in [np.zeros(6), np.array([8., 0., 0., 0., 0., 0.])]:
                density = recon.density(u)
                row = dict(latent=u.tolist(), raw_coordinates=recon.raw(u).tolist(),
                           shell_valid=bool(np.linalg.norm(u) <= recon.radius), density_details=density)
                got = hard_free_log_density(row, args[1], recon.radius)
                if density['log_density'] == -math.inf:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, density['log_density'], places=12)

    def test_hardfree_only_alpha_one_and_beta_zero_limits(self):
        args = self.setup(); args[1]['class_channels'] = [{'class': 'hard_free', 'probability': 1.}]
        r = line.Reconstructor(*args); g = copy.deepcopy(args[1]); g['schema'] = 'defensive-hard-free-line-guide-v1'
        base = line.hard.Reconstructor(args[0], g, args[2], args[3])
        for u in [np.zeros(6), np.array([.01, .01, .02, .3, -.1, .1])]:
            self.assertAlmostEqual(r.density(u)['log_density'], base.density(u)['log_density'], places=12)
        for key, value in [('conditional_probability', 0.), ('defensive_uniform_shell_probability', 1.)]:
            copyargs = copy.deepcopy(args); copyargs[1][key] = value
            result = line.Reconstructor(*copyargs).density(np.zeros(6))
            self.assertTrue(result['conditioning_disabled'])
            self.assertEqual(result['log_density'], result['baseline_log_density'])


if __name__ == '__main__':
    unittest.main()
