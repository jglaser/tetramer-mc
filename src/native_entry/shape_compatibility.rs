//! Static consistency check between the compiled native observer geometry and
//! the physical rigid sphere union. This never alters centers or hard radii.
use super::*;
use crate::geometry::{Atom, Shape};
use std::collections::VecDeque;

/// Explicit audit of a deterministic one-to-one sphere matching. The native
/// ordering is member-major then frozen monomer-atom order. Maximum errors are
/// over matched atoms; a successful compatibility claim requires a full bijection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeShapeCompatibility {
    pub compiled_sha256: String,
    pub expected_shape_sha256: String,
    pub native_atoms: usize,
    pub physical_atoms: usize,
    pub matched_atoms: usize,
    pub center_tolerance_a: f64,
    pub radius_tolerance_a: f64,
    pub matched_max_center_error_a: f64,
    pub matched_max_radius_error_a: f64,
    pub physical_index_by_native_atom: Vec<Option<usize>>,
    pub unmatched_native_atoms: Vec<usize>,
    pub unmatched_physical_atoms: Vec<usize>,
    pub compatible: bool,
    /// Bound on negative native atom gap allowed by nonoverlapping physical
    /// atoms under exact proper rigid transformations, before FP arithmetic.
    pub pair_overlap_slack_bound_a: Option<f64>,
    pub observer_hard_overlap_tolerance_a: f64,
    pub hard_valid_implication_within_tolerance: bool,
}

impl CompleteNativeEntry {
    /// Static diagnostic with explicit tolerances. Matching is a deterministic
    /// maximum-cardinality bipartite matching, not an order-dependent greedy
    /// nearest-neighbor assertion. Permutations and duplicate spheres are valid.
    /// A mismatch is returned as a report; malformed/unrepresentable arithmetic
    /// is an error. No protein pose, overlap query, or random draw is evaluated.
    pub fn shape_compatibility(
        &self,
        shape: &Shape,
        center_tolerance_a: f64,
        radius_tolerance_a: f64,
    ) -> Result<NativeShapeCompatibility> {
        ensure!(
            center_tolerance_a.is_finite()
                && center_tolerance_a >= 0.
                && radius_tolerance_a.is_finite()
                && radius_tolerance_a >= 0.,
            "Invalid native shape matching tolerances"
        );
        ensure!(!shape.atoms.is_empty(), "Physical shape has no atoms");
        for atom in &shape.atoms {
            ensure!(
                atom.center.iter().all(|x| x.is_finite())
                    && atom.radius.is_finite()
                    && atom.radius > 0.,
                "Invalid physical shape atom"
            );
        }
        let size = self
            .definition
            .members
            .len()
            .checked_mul(self.definition.monomer_atoms.len())
            .context("Native shape atom count overflow")?;
        let mut expanded = Vec::with_capacity(size);
        for member in &self.definition.members {
            for atom in &self.definition.monomer_atoms {
                let center = add(member.position, matvec(member.rotation, atom.center));
                ensure!(
                    center.iter().all(|x| x.is_finite()),
                    "Unrepresentable reconstructed native atom"
                );
                expanded.push(Atom {
                    center,
                    radius: atom.radius,
                });
            }
        }
        let mut x_order: Vec<_> = (0..shape.atoms.len()).collect();
        x_order.sort_by(|&i, &j| {
            shape.atoms[i].center[0]
                .total_cmp(&shape.atoms[j].center[0])
                .then(i.cmp(&j))
        });
        let mut neighbors = Vec::with_capacity(size);
        for atom in &expanded {
            // Padding only protects the search window; exact tolerance tests
            // below determine matching edges and never inflate physical radii.
            let guard = 64. * f64::EPSILON * (1. + atom.center[0].abs() + center_tolerance_a);
            let lo = atom.center[0] - center_tolerance_a - guard;
            let hi = atom.center[0] + center_tolerance_a + guard;
            ensure!(
                lo.is_finite() && hi.is_finite(),
                "Unrepresentable native matching search window"
            );
            let first = x_order.partition_point(|&i| shape.atoms[i].center[0] < lo);
            let end = x_order.partition_point(|&i| shape.atoms[i].center[0] <= hi);
            let mut edges = Vec::new();
            for &i in &x_order[first..end] {
                let actual = &shape.atoms[i];
                let distance = crate::math::norm(sub(atom.center, actual.center));
                let radius_error = (atom.radius - actual.radius).abs();
                ensure!(
                    distance.is_finite() && radius_error.is_finite(),
                    "Unrepresentable native matching distance"
                );
                if distance <= center_tolerance_a && radius_error <= radius_tolerance_a {
                    edges.push(i);
                }
            }
            edges.sort_unstable();
            neighbors.push(edges);
        }
        let mut native_match: Vec<Option<usize>> = vec![None; size];
        let mut physical_owner: Vec<Option<usize>> = vec![None; shape.atoms.len()];
        // Iterative augmenting paths avoid recursion and handle ambiguous local
        // matches that require reassignment of an earlier native atom.
        for start in 0..size {
            let mut visited = vec![false; size];
            let mut predecessor: Vec<Option<usize>> = vec![None; shape.atoms.len()];
            let mut queue = VecDeque::from([start]);
            visited[start] = true;
            let mut endpoint = None;
            'search: while let Some(i) = queue.pop_front() {
                for &j in &neighbors[i] {
                    if predecessor[j].is_some() {
                        continue;
                    }
                    predecessor[j] = Some(i);
                    if let Some(previous) = physical_owner[j] {
                        if !visited[previous] {
                            visited[previous] = true;
                            queue.push_back(previous);
                        }
                    } else {
                        endpoint = Some(j);
                        break 'search;
                    }
                }
            }
            if let Some(mut j) = endpoint {
                loop {
                    let i = predecessor[j].context("Broken native matching augmenting path")?;
                    let old = native_match[i];
                    native_match[i] = Some(j);
                    physical_owner[j] = Some(i);
                    if let Some(previous) = old {
                        j = previous;
                    } else {
                        break;
                    }
                }
            }
        }
        let unmatched_native_atoms: Vec<_> = native_match
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.is_none().then_some(i))
            .collect();
        let unmatched_physical_atoms: Vec<_> = physical_owner
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.is_none().then_some(i))
            .collect();
        let compatible = size == shape.atoms.len()
            && unmatched_native_atoms.is_empty()
            && unmatched_physical_atoms.is_empty();
        let mut max_center = 0_f64;
        let mut max_radius = 0_f64;
        let mut one_atom_slack = 0_f64;
        for (i, actual) in native_match.iter().enumerate() {
            if let Some(j) = actual {
                let a = &expanded[i];
                let b = &shape.atoms[*j];
                let distance = crate::math::norm(sub(a.center, b.center));
                max_center = max_center.max(distance);
                max_radius = max_radius.max((a.radius - b.radius).abs());
                one_atom_slack = one_atom_slack.max(distance + (a.radius - b.radius).max(0.));
            }
        }
        let bound = compatible.then_some(2. * one_atom_slack);
        ensure!(
            bound.is_none_or(f64::is_finite),
            "Unrepresentable native pair-gap discrepancy bound"
        );
        Ok(NativeShapeCompatibility {
            compiled_sha256: self.compiled_sha256().to_owned(),
            expected_shape_sha256: self.shape_sha256().to_owned(),
            native_atoms: size,
            physical_atoms: shape.atoms.len(),
            matched_atoms: size - unmatched_native_atoms.len(),
            center_tolerance_a,
            radius_tolerance_a,
            matched_max_center_error_a: max_center,
            matched_max_radius_error_a: max_radius,
            physical_index_by_native_atom: native_match,
            unmatched_native_atoms,
            unmatched_physical_atoms,
            compatible,
            pair_overlap_slack_bound_a: bound,
            observer_hard_overlap_tolerance_a: 1e-8,
            hard_valid_implication_within_tolerance: bound.is_some_and(|x| x <= 1e-8),
        })
    }

    /// Strict setup gate with frozen numerical tolerances (center 1e-10 Å,
    /// radius 1e-12 Å). This is an input-identity check, not a radius repair.
    /// The pair-gap bound leaves almost all of the observer's 1e-8 Å allowance
    /// for subsequent FP64 pose/rotation arithmetic, which remains an explicit
    /// implementation obligation rather than a real-arithmetic certificate.
    pub fn validate_shape_compatibility(&self, shape: &Shape) -> Result<NativeShapeCompatibility> {
        let report = self.shape_compatibility(shape, 1e-10, 1e-12)?;
        ensure!(
            report.compatible && report.hard_valid_implication_within_tolerance,
            "Native/physical shape mismatch: native_atoms={} physical_atoms={} matched={} unmatched_native={:?} unmatched_physical={:?} center_tolerance_A={} radius_tolerance_A={} pair_gap_bound_A={:?}",
            report.native_atoms,
            report.physical_atoms,
            report.matched_atoms,
            report.unmatched_native_atoms,
            report.unmatched_physical_atoms,
            report.center_tolerance_a,
            report.radius_tolerance_a,
            report.pair_overlap_slack_bound_a
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{IDENTITY, cayley};
    use serde_json::{Value, json};
    fn fixture() -> Value {
        json!({"schema":"native-entry-compiled-v1","source_definition_sha256":"0".repeat(64),
        "source_input_sha256":{"tetramer-shape.json":"1".repeat(64)},
        "criteria":{"body_member_position_entry_A":2.,"body_orientation_entry_deg":15.,"monomer_position_entry_A":3.,"monomer_orientation_entry_deg":20.,"contact_entry_A":2.,"native_reference_patch_gap_A":1.,"minimum_shared_native_residue_pairs":1,"hard_overlap_tolerance_A":1e-8,"catalogue_cycle_position_tolerance_A":1e-6,"catalogue_cycle_angle_tolerance_deg":1e-6},
        "fixed_poses":[{"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]}],
        "members":[{"position":[0.,0.,0.],"rotation":IDENTITY},{"position":[5.,2.,-1.],"rotation":cayley([0.1,0.2,-0.1])}],
        "monomer_atoms":[{"center":[0.,0.,0.],"radius":1.,"residue":0},{"center":[0.5,1.,0.],"radius":0.2,"residue":0}],"residue_count":1,
        "references":[{"label":"A","family":"A","position":[2.,0.,0.],"rotation":IDENTITY,"native_residue_pairs":[0]}],
        "motifs":[{"id":7,"position":[2.,0.,0.],"rotation":IDENTITY,"member_contacts":[{"member_i":0,"member_j":0,"directed_class":"A"}]}]})
    }
    fn model(v: &Value) -> CompleteNativeEntry {
        CompleteNativeEntry::from_bytes(&serde_json::to_vec(v).unwrap()).unwrap()
    }
    fn shape(m: &CompleteNativeEntry) -> Shape {
        Shape {
            name: "toy only".into(),
            volume: 0.,
            atoms: m
                .definition
                .members
                .iter()
                .flat_map(|member| {
                    m.definition.monomer_atoms.iter().map(|atom| Atom {
                        center: add(member.position, matvec(member.rotation, atom.center)),
                        radius: atom.radius,
                    })
                })
                .collect(),
        }
    }
    #[test]
    fn reconstructed_permutation_and_zero_tolerance_identity() {
        let m = model(&fixture());
        let mut s = shape(&m);
        s.atoms.reverse();
        let r = m.shape_compatibility(&s, 0., 0.).unwrap();
        assert!(r.compatible && r.hard_valid_implication_within_tolerance);
        assert_eq!(
            r.physical_index_by_native_atom,
            vec![Some(3), Some(2), Some(1), Some(0)]
        );
        assert_eq!(r.pair_overlap_slack_bound_a, Some(0.));
        assert_eq!(r.matched_max_center_error_a, 0.);
        m.validate_shape_compatibility(&s).unwrap();
    }
    #[test]
    fn counts_unrelated_centers_and_radii_cannot_pass() {
        let m = model(&fixture());
        let original = shape(&m);
        let mut s = original.clone();
        s.atoms.pop();
        let r = m.shape_compatibility(&s, 1e-10, 1e-12).unwrap();
        assert!(!r.compatible);
        assert_eq!(r.unmatched_native_atoms, vec![3]);
        assert!(r.pair_overlap_slack_bound_a.is_none());
        let mut s = original.clone();
        s.atoms.push(Atom {
            center: [100.; 3],
            radius: 1.,
        });
        let r = m.shape_compatibility(&s, 1e-10, 1e-12).unwrap();
        assert_eq!(r.unmatched_physical_atoms, vec![4]);
        assert!(!r.compatible);
        let mut s = original.clone();
        s.atoms[1].center[0] += 0.1;
        let r = m.shape_compatibility(&s, 1e-10, 1e-12).unwrap();
        assert_eq!(r.unmatched_native_atoms, vec![1]);
        assert!(!r.compatible);
        let mut s = original;
        s.atoms[1].radius += 0.01;
        assert!(m.validate_shape_compatibility(&s).is_err());
    }
    #[test]
    fn ambiguous_candidates_require_reassignment_not_greedy_matching() {
        let mut v = fixture();
        v["members"].as_array_mut().unwrap().truncate(1);
        v["monomer_atoms"] = json!([{"center":[0.,0.,0.],"radius":1.,"residue":0},{"center":[0.2,0.,0.],"radius":1.,"residue":0}]);
        let m = model(&v);
        let s = Shape {
            name: String::new(),
            volume: 0.,
            atoms: vec![
                Atom {
                    center: [0.1, 0., 0.],
                    radius: 1.,
                },
                Atom {
                    center: [-0.1, 0., 0.],
                    radius: 1.,
                },
            ],
        };
        let r = m.shape_compatibility(&s, 0.15, 0.).unwrap();
        assert!(r.compatible);
        assert_eq!(r.physical_index_by_native_atom, vec![Some(1), Some(0)]);
        assert!(!r.hard_valid_implication_within_tolerance);
    }
    #[test]
    fn explicit_tolerance_reports_error_and_bounds_gap_without_repairs() {
        let m = model(&fixture());
        let mut s = shape(&m);
        s.atoms[0].center[0] += 1e-9;
        s.atoms[0].radius -= 1e-9;
        let snapshot = serde_json::to_vec(&s).unwrap();
        let r = m.shape_compatibility(&s, 1e-8, 1e-8).unwrap();
        assert!(r.compatible && r.hard_valid_implication_within_tolerance);
        assert!((r.pair_overlap_slack_bound_a.unwrap() - 4e-9).abs() < 1e-15);
        assert_eq!(serde_json::to_vec(&s).unwrap(), snapshot);
        assert!(m.validate_shape_compatibility(&s).is_err());
        s.atoms[0].center[0] = 2e-8;
        let r = m.shape_compatibility(&s, 1e-7, 1e-8).unwrap();
        assert!(r.compatible);
        assert!(!r.hard_valid_implication_within_tolerance);
        // Larger physical radii strengthen hard exclusion, so only an excess
        // native radius (not |delta radius|) enters the discrepancy bound.
        let mut s = shape(&m);
        s.atoms[0].radius += 0.01;
        let r = m.shape_compatibility(&s, 0., 0.02).unwrap();
        assert!(r.compatible);
        assert_eq!(r.pair_overlap_slack_bound_a, Some(0.));
    }
    #[test]
    fn duplicate_atoms_and_invalid_input_are_explicit() {
        let mut v = fixture();
        v["members"][1] = v["members"][0].clone();
        let m = model(&v);
        let s = shape(&m);
        assert!(m.validate_shape_compatibility(&s).unwrap().compatible);
        assert!(m.shape_compatibility(&s, -1., 0.).is_err());
        assert!(m.shape_compatibility(&s, 0., f64::NAN).is_err());
        let mut bad = s;
        bad.atoms[0].center[0] = f64::NAN;
        assert!(m.validate_shape_compatibility(&bad).is_err());
    }
}
