use super::*;
use crate::math::{IDENTITY, cayley, quaternion};
use serde_json::{Value, json};

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn fixture() -> Value {
    json!({"schema":"native-entry-compiled-v1","source_definition_sha256":"0".repeat(64),
    "source_input_sha256":{"tetramer-shape.json":"1".repeat(64)},
    "criteria":{"body_member_position_entry_A":2.,"body_orientation_entry_deg":15.,"monomer_position_entry_A":3.,"monomer_orientation_entry_deg":20.,"contact_entry_A":2.,"native_reference_patch_gap_A":1.,"minimum_shared_native_residue_pairs":1,"hard_overlap_tolerance_A":1e-8,"catalogue_cycle_position_tolerance_A":1e-6,"catalogue_cycle_angle_tolerance_deg":1e-6},
    "fixed_poses":[pose([0.,0.,0.]),pose([10.,0.,0.])],
    "members":[{"position":[0.,0.,0.],"rotation":IDENTITY}],
    "monomer_atoms":[{"center":[0.,0.,0.],"radius":1.,"residue":0}],"residue_count":1,
    "references":[{"label":"A","family":"A","position":[2.,0.,0.],"rotation":IDENTITY,"native_residue_pairs":[0]}],
    "motifs":[{"id":7,"position":[2.,0.,0.],"rotation":IDENTITY,"member_contacts":[{"member_i":0,"member_j":0,"directed_class":"A"}]}]})
}
fn model(v: &Value) -> CompleteNativeEntry {
    CompleteNativeEntry::from_bytes(&serde_json::to_vec(v).unwrap()).unwrap()
}
fn line(
    m: &CompleteNativeEntry,
    origin: Vec3,
    direction: Vec3,
    range: [f64; 2],
) -> NativeLineResult {
    m.translation_intervals(pose(origin), direction, range)
        .unwrap()
}
fn interval(a: f64, b: f64) -> Interval {
    Interval::closed(a, b).unwrap()
}

#[test]
fn all_anchors_and_motifs_with_any_supporting_bond() {
    let mut v = fixture();
    let mut absent = v["references"][0].clone();
    absent["label"] = json!("B");
    absent["position"] = json!([200., 0., 0.]);
    v["references"].as_array_mut().unwrap().push(absent);
    v["motifs"][0]["member_contacts"]
        .as_array_mut()
        .unwrap()
        .push(json!({"member_i":0,"member_j":0,"directed_class":"B"}));
    let m = model(&v);
    let answer = line(&m, [0., 0., 0.], [1., 0., 0.], [-8., 20.]);
    assert_eq!(
        answer.intervals.intervals(),
        &[interval(0., 4.), interval(10., 14.)]
    );
    assert_eq!(answer.counts.anchors_visited, 2);
    assert_eq!(answer.counts.bond_requests, 4);
    assert_eq!(answer.counts.bond_position_empty, 2);
    // ANY bond accepts although the B bond is absent for both anchors.
    assert!(m.classify(pose([3., 0., 0.])).unwrap().native_any);
    let mut second = v["motifs"][0].clone();
    second["id"] = json!(8);
    second["position"] = json!([-2., 0., 0.]);
    v["motifs"].as_array_mut().unwrap().push(second);
    // Reference A remains limited to x>=-1; a second reference permits the full
    // reflected support and proves union over distinct frozen motif positions.
    v["references"][1]["position"] = json!([-2., 0., 0.]);
    let answer = line(&model(&v), [0., 0., 0.], [1., 0., 0.], [-8., 20.]);
    assert_eq!(
        answer.intervals.intervals(),
        &[interval(-4., 4.), interval(6., 14.)]
    );
    assert_eq!(answer.counts.motifs_visited, 4);
    assert!(answer.counts.bond_cache_hits > 0);
}

#[test]
fn all_four_member_bounds_are_intersected() {
    let mut v = fixture();
    v["members"] =
        json!([0., 20., 5., 10.].map(|y| json!({"position":[0.,y,0.],"rotation":IDENTITY})));
    // Place the monomer atom at the last member's physical origin so the
    // sampled hard-valid segment also contains a supporting native contact.
    v["monomer_atoms"][0]["center"] = json!([0., -20., 0.]);
    v["motifs"][0]["member_contacts"][0]["member_i"] = json!(1);
    v["motifs"][0]["member_contacts"][0]["member_j"] = json!(1);
    let m = model(&v);
    let moving = Pose {
        position: [0.; 3],
        orientation: [
            (5_f64.to_radians()).cos(),
            0.,
            0.,
            (5_f64.to_radians()).sin(),
        ],
    };
    let result = m
        .translation_intervals_pair(pose([0.; 3]), moving, [1., 0., 0.], [0., 6.])
        .unwrap();
    assert_eq!(result.counts.member_constraints_tested, 4);
    assert!(!result.intervals.contains(2.));
    assert!(result.intervals.contains(3.9));
    for s in [2.1, 3., 3.5, 3.9, 4.1] {
        let p = Pose {
            position: [s, 0., 0.],
            ..moving
        };
        let rr = rotation(p.orientation);
        for a in &m.definition.members {
            for b in &m.definition.members {
                let anchor_atom = add(a.position, m.definition.monomer_atoms[0].center);
                let moving_atom = add(
                    p.position,
                    matvec(rr, add(b.position, m.definition.monomer_atoms[0].center)),
                );
                assert!(
                    norm(sub(anchor_atom, moving_atom)) >= 2.,
                    "hard invalid test pose at {s}"
                );
            }
        }
        assert_eq!(
            result.intervals.contains(s),
            !m.classify_pair(pose([0.; 3]), p).unwrap().is_empty()
        );
    }
}

#[test]
fn directed_reference_residues_exclude_unrelated_contacts() {
    let mut v = fixture();
    v["residue_count"] = json!(2);
    v["monomer_atoms"] = json!([
        {"center":[0.,0.,0.],"radius":1.,"residue":0},
        {"center":[10.,0.,0.],"radius":1.,"residue":1}]);
    v["motifs"][0]["position"] = json!([-12., 0., 0.]);
    v["references"][0]["position"] = json!([-12., 0., 0.]);
    v["references"][0]["native_residue_pairs"] = json!([1]);
    let m = model(&v);
    let answer = m
        .translation_intervals_pair(pose([0.; 3]), pose([0.; 3]), [1., 0., 0.], [-20., 0.])
        .unwrap();
    assert_eq!(answer.intervals.intervals(), &[interval(-14., -10.)]);
    assert!(
        !m.classify_pair(pose([0.; 3]), pose([-12., 0., 0.]))
            .unwrap()
            .is_empty()
    );
    v["references"][0]["native_residue_pairs"] = json!([2]);
    let m = model(&v);
    assert!(
        m.translation_intervals_pair(pose([0.; 3]), pose([0.; 3]), [1., 0., 0.], [-20., 0.])
            .unwrap()
            .intervals
            .is_empty()
    );
    assert!(
        m.classify_pair(pose([0.; 3]), pose([-12., 0., 0.]))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn body_and_monomer_angular_failures_are_constant() {
    let m = model(&fixture());
    let p = Pose {
        orientation: quaternion(cayley([0., 0., 0.2])),
        ..pose([0.; 3])
    };
    let answer = m
        .translation_intervals(p, [1., 0., 0.], [-5., 20.])
        .unwrap();
    assert!(answer.intervals.is_empty());
    assert_eq!(answer.counts.body_angle_rejected, 2);
    assert_eq!(answer.counts.bond_requests, 0);
    let mut v = fixture();
    v["references"][0]["rotation"] = json!(cayley([0., 0., 0.3]));
    let answer = line(&model(&v), [0.; 3], [1., 0., 0.], [-5., 20.]);
    assert!(answer.intervals.is_empty());
    assert_eq!(answer.counts.bond_angle_rejected, 2);
    assert_eq!(answer.counts.atom_pairs_tested, 0);
}

#[test]
fn inclusive_contact_tangent_and_native_complement() {
    let mut v = fixture();
    v["fixed_poses"] = json!([pose([0.; 3])]);
    v["motifs"][0]["position"] = json!([0., 4., 0.]);
    v["references"][0]["position"] = json!([0., 4., 0.]);
    let m = model(&v);
    let answer = line(&m, [0., 4., 0.], [1., 0., 0.], [-3., 3.]);
    assert_eq!(answer.intervals.intervals(), &[interval(0., 0.)]);
    assert!(m.classify(pose([0., 4., 0.])).unwrap().native_any);
    assert!(!m.classify(pose([0.1, 4., 0.])).unwrap().native_any);
    let complement = IntervalSet::segment([-3., 3.])
        .unwrap()
        .difference(&answer.intervals);
    assert_eq!(complement.intervals().len(), 2);
    assert!(!complement.contains(0.));
    assert!(!complement.intervals()[0].upper_closed);
    assert!(!complement.intervals()[1].lower_closed);
    let constant = line(&m, [0., 4., 0.], [0.; 3], [-3., 3.]);
    assert_eq!(constant.intervals, IntervalSet::segment([-3., 3.]).unwrap());
    assert!(
        line(&m, [0., 4.1, 0.], [0.; 3], [-3., 3.])
            .intervals
            .is_empty()
    );
}

#[test]
fn hard_invalid_parts_do_not_throw_but_require_external_intersection() {
    let m = model(&fixture());
    let answer = line(&m, [0.; 3], [1., 0., 0.], [-5., 20.]);
    assert!(answer.intervals.contains(1.));
    assert!(m.classify(pose([1., 0., 0.])).is_err());
    // Every x in these support intervals is hard valid against both fixed
    // monomer spheres. Equality with classify is claimed only on this support.
    let hard_free =
        IntervalSet::from_intervals(vec![interval(2., 8.), interval(12., 20.)]).unwrap();
    let native = answer.intervals.intersection(&hard_free);
    let competing = hard_free.difference(&answer.intervals);
    for i in 0..=180 {
        let s = 2. + i as f64 * 0.1;
        if !hard_free.contains(s) {
            continue;
        }
        let expected = m.classify(pose([s, 0., 0.])).unwrap().native_any;
        assert_eq!(native.contains(s), expected, "native at {s}");
        assert_eq!(competing.contains(s), !expected, "complement at {s}");
    }
}

#[test]
fn unrelated_residue_overlap_still_invalidates_reporting_classifier() {
    let mut v = fixture();
    v["residue_count"] = json!(2);
    v["monomer_atoms"] = json!([
        {"center":[0.,0.,0.],"radius":1.,"residue":0},
        {"center":[3.,0.,0.],"radius":1.,"residue":1}]);
    // Native 0→0 touches at x=2, while unrelated anchor residue1 overlaps
    // moving residue0. The interval must not masquerade as a hard certificate.
    let m = model(&v);
    let answer = m
        .translation_intervals_pair(pose([0.; 3]), pose([0.; 3]), [1., 0., 0.], [2., 4.])
        .unwrap();
    assert!(answer.intervals.contains(2.));
    assert!(
        m.classify_pair(pose([0.; 3]), pose([2., 0., 0.]))
            .unwrap_err()
            .to_string()
            .contains("motif")
    );
}

#[test]
fn transformed_nonunit_lines_match_the_complete_hard_valid_observer() {
    let mut v = fixture();
    let r = cayley([0.2, -0.1, 0.3]);
    let anchor = Pose {
        position: [8., -3., 5.],
        orientation: quaternion(r),
    };
    v["fixed_poses"] = json!([anchor]);
    let m = model(&v);
    let origin = Pose {
        position: add(anchor.position, matvec(r, [3., 0., 0.])),
        orientation: anchor.orientation,
    };
    let direction = matvec(r, [2., 0., 0.]);
    let answer = m
        .translation_intervals(origin, direction, [-0.4, 2.])
        .unwrap();
    for i in 0..=100 {
        let s = -0.4 + 2.4 * i as f64 / 100.;
        let moving = Pose {
            position: add(origin.position, direction.map(|x| x * s)),
            ..origin
        };
        assert_eq!(
            answer.intervals.contains(s),
            m.classify(moving).unwrap().native_any,
            "s={s}"
        );
    }
    assert!((answer.intervals.intervals()[0].upper - 0.5).abs() < 1e-13);
}

#[test]
fn conservative_projection_pruning_matches_all_leaf_roots() {
    let mut v = fixture();
    v["monomer_atoms"] = json!([
        {"center":[0.,0.,0.],"radius":1.,"residue":0},
        {"center":[0.,20.,0.],"radius":0.5,"residue":0},
        {"center":[10.,0.,20.],"radius":1.5,"residue":0},
        {"center":[-10.,0.,-20.],"radius":0.7,"residue":0}]);
    let m = model(&v);
    for direction in [[1., 0., 0.], [0., 1., 0.], [0.3, -0.7, 1.4], [0.; 3]] {
        for origin in [[0.; 3], [0., 4., 0.], [3., 0.1, 0.2]] {
            let p = pose(origin);
            let pruned = m
                .line_intervals_for_anchors(m.fixed_poses(), p, direction, [-4., 20.], true)
                .unwrap();
            let full = m
                .line_intervals_for_anchors(m.fixed_poses(), p, direction, [-4., 20.], false)
                .unwrap();
            assert_eq!(pruned.intervals, full.intervals);
            assert_eq!(
                pruned.counts.atom_pairs_considered,
                full.counts.atom_pairs_considered
            );
            assert_eq!(
                pruned.counts.atom_pairs_tested + pruned.counts.atom_pairs_pruned,
                full.counts.atom_pairs_tested
            );
        }
    }
    assert!(
        line(&m, [0.; 3], [1., 0., 0.], [-4., 20.])
            .counts
            .atom_pairs_pruned
            > 0
    );
}

#[test]
fn invalid_lines_and_overflow_fail_explicitly() {
    let m = model(&fixture());
    assert!(
        m.translation_intervals(pose([0.; 3]), [f64::NAN, 0., 0.], [0., 1.])
            .is_err()
    );
    assert!(
        m.translation_intervals(pose([0.; 3]), [1., 0., 0.], [1., 0.])
            .is_err()
    );
    assert!(
        m.translation_intervals(pose([0.; 3]), [1., 0., 0.], [0., f64::INFINITY])
            .is_err()
    );
    assert!(
        m.translation_intervals(pose([f64::MAX; 3]), [f64::MAX; 3], [-1., 1.])
            .is_err()
    );
}
