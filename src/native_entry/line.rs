//! Complete frozen native-registration/contact sets on translation lines.
//!
//! These intervals do not certify hard validity. In particular the classifier's
//! `contacts` scans *all* atoms of a queried monomer pair and errors on a gap
//! below -1e-8, even if the overlapping residue is not a native reference pair.
//! The interval construction intentionally does not call that error-producing
//! observer on the whole line: callers intersect its result with independently
//! certified whole-body hard-free geometry. Only on that support is membership
//! equivalent to the complete instantaneous classifier. The compiled monomer
//! and member geometry must represent the same frozen physical whole-body shape.
//!
//! Roots are analytic for the specified sphere model. Rotations, distance sums,
//! angle thresholds, quadratic roots, endpoint comparisons and conservative
//! pruning guards use FP64; this is not certified real arithmetic. Inclusive
//! thresholds may differ in their last bit from a separately evaluated norm.
use super::*;
use crate::line_geometry::{Interval, IntervalSet, ball_interval};

/// Work counters include rejected motifs/bonds and cached requests. They are
/// diagnostics, never a subset of the frozen native catalogue or a work cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeLineCounts {
    pub anchors_visited: u64,
    pub motifs_visited: u64,
    pub body_angle_rejected: u64,
    pub member_constraints_tested: u64,
    pub body_position_empty: u64,
    pub bond_requests: u64,
    pub bond_cache_hits: u64,
    pub bond_angle_rejected: u64,
    pub bond_position_empty: u64,
    pub residue_pairs_visited: u64,
    pub atom_pairs_considered: u64,
    pub atom_pairs_pruned: u64,
    pub atom_pairs_tested: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeLineResult {
    pub intervals: IntervalSet,
    pub counts: NativeLineCounts,
}

fn closed_ball(
    offset: Vec3,
    direction: Vec3,
    radius: f64,
    segment: Interval,
) -> Result<IntervalSet> {
    IntervalSet::from_intervals(
        ball_interval(offset, direction, radius, segment, true)?
            .into_iter()
            .collect(),
    )
}

/// A conservative coordinate projection bound for a finite line segment.
/// This only rejects obviously distant atom pairs; leaf roots retain their
/// original radii with no tolerance or padding. Overflow fails explicitly.
fn projected_miss(offset: Vec3, direction: Vec3, radius: f64, segment: Interval) -> Result<bool> {
    ensure!(
        offset.iter().chain(direction.iter()).all(|x| x.is_finite()) && radius.is_finite(),
        "unrepresentable native atom-line geometry"
    );
    for axis in 0..3 {
        let a = direction[axis].mul_add(segment.lower, offset[axis]);
        let b = direction[axis].mul_add(segment.upper, offset[axis]);
        let extent = direction[axis].abs() * segment.lower.abs().max(segment.upper.abs());
        let guard = 1024. * f64::EPSILON * (1. + offset[axis].abs() + extent + radius);
        ensure!(
            a.is_finite() && b.is_finite() && guard.is_finite(),
            "unrepresentable native atom-line pruning bound"
        );
        if a.min(b) > radius + guard || a.max(b) < -radius - guard {
            return Ok(true);
        }
    }
    Ok(false)
}

impl CompleteNativeEntry {
    /// Union over every frozen anchor and motif, with inclusive body/member and
    /// native residue-contact thresholds. `origin + s*direction`, s in `range`,
    /// has fixed orientation; zero direction and closed singletons are valid.
    ///
    /// Intersect with independent whole-body hard-free intervals before treating
    /// this set as `classify(...).native_any`. No wall, source capture, reference
    /// pocket, periodic image or cycle constraint is introduced here.
    pub fn translation_intervals(
        &self,
        origin: Pose,
        direction: Vec3,
        range: [f64; 2],
    ) -> Result<NativeLineResult> {
        self.line_intervals_for_anchors(self.fixed_poses(), origin, direction, range, true)
    }

    /// Directed anchor→moving variant; the same hard-valid support precondition
    /// applies as to `classify_pair` and `translation_intervals`.
    pub fn translation_intervals_pair(
        &self,
        anchor: Pose,
        origin: Pose,
        direction: Vec3,
        range: [f64; 2],
    ) -> Result<NativeLineResult> {
        self.line_intervals_for_anchors(&[anchor], origin, direction, range, true)
    }

    fn line_intervals_for_anchors(
        &self,
        anchors: &[Pose],
        origin: Pose,
        direction: Vec3,
        range: [f64; 2],
        prune: bool,
    ) -> Result<NativeLineResult> {
        validate_pose(origin).context("native line origin")?;
        ensure!(
            direction.iter().all(|v| v.is_finite()),
            "nonfinite native line direction"
        );
        let whole = IntervalSet::segment(range)?;
        let segment = whole.intervals()[0];
        let mut result = NativeLineResult {
            intervals: IntervalSet::empty(),
            counts: NativeLineCounts::default(),
        };
        for anchor in anchors {
            validate_pose(*anchor).context("native line anchor")?;
            result.counts.anchors_visited += 1;
            let inverse_anchor = transpose(rotation(anchor.orientation));
            let r = matmul(inverse_anchor, rotation(origin.orientation));
            let d = matvec(inverse_anchor, sub(origin.position, anchor.position));
            let v = matvec(inverse_anchor, direction);
            ensure!(
                d.iter().chain(v.iter()).all(|x| x.is_finite()),
                "unrepresentable native line frame"
            );
            // Cache each directed member/reference subproblem over the complete
            // line range; then intersect separately with every motif's support.
            let mut bonds: BTreeMap<(usize, usize, usize), IntervalSet> = BTreeMap::new();
            for motif in &self.definition.motifs {
                result.counts.motifs_visited += 1;
                if angle_degrees(r, motif.rotation) > 15. {
                    result.counts.body_angle_rejected += 1;
                    continue;
                }
                let mut body = whole.clone();
                for member in &self.definition.members {
                    result.counts.member_constraints_tested += 1;
                    let offset = sub(
                        add(matvec(r, member.position), d),
                        add(matvec(motif.rotation, member.position), motif.position),
                    );
                    body = body.intersection(&closed_ball(offset, v, 2., segment)?);
                    if body.is_empty() {
                        break;
                    }
                }
                if body.is_empty() {
                    result.counts.body_position_empty += 1;
                    continue;
                }
                let mut support = IntervalSet::empty();
                for contact in &motif.member_contacts {
                    result.counts.bond_requests += 1;
                    let index = self.reference_indices[&contact.directed_class];
                    let key = (contact.member_i, contact.member_j, index);
                    if let Some(cached) = bonds.get(&key) {
                        result.counts.bond_cache_hits += 1;
                        support = support.union(cached);
                    } else {
                        let intervals = self
                            .bond_line_intervals(key, r, d, v, segment, prune, &mut result.counts)
                            .with_context(|| {
                                format!(
                                    "native line motif {} members {}→{} reference {}",
                                    motif.id, key.0, key.1, contact.directed_class
                                )
                            })?;
                        support = support.union(&intervals);
                        bonds.insert(key, intervals);
                    }
                }
                // ANY supporting bond, ALL body member displacement bounds.
                result.intervals = result.intervals.union(&body.intersection(&support));
            }
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn bond_line_intervals(
        &self,
        key: (usize, usize, usize),
        r: Mat3,
        d: Vec3,
        v: Vec3,
        segment: Interval,
        prune: bool,
        counts: &mut NativeLineCounts,
    ) -> Result<IntervalSet> {
        let i = &self.definition.members[key.0];
        let j = &self.definition.members[key.1];
        let inverse_i = transpose(i.rotation);
        let mr = matmul(matmul(inverse_i, r), j.rotation);
        let md = matvec(inverse_i, sub(add(matvec(r, j.position), d), i.position));
        let mv = matvec(inverse_i, v);
        let reference = &self.definition.references[key.2];
        if angle_degrees(mr, reference.rotation) > 20. {
            counts.bond_angle_rejected += 1;
            return Ok(IntervalSet::empty());
        }
        let position = closed_ball(sub(md, reference.position), mv, 3., segment)?;
        if position.is_empty() {
            counts.bond_position_empty += 1;
            return Ok(position);
        }
        // A ball intersected with a segment has exactly one interval, including
        // a possible singleton. Cache coordinate transforms once per bond.
        let bounded = position.intervals()[0];
        let atoms = &self.definition.monomer_atoms;
        let centers: Vec<_> = atoms
            .iter()
            .map(|a| add(matvec(mr, a.center), md))
            .collect();
        let mut hits = Vec::new();
        for &pair in &reference.native_residue_pairs {
            counts.residue_pairs_visited += 1;
            let left = pair / self.definition.residue_count;
            let right = pair % self.definition.residue_count;
            for &a in &self.residue_atom_indices[left] {
                for &b in &self.residue_atom_indices[right] {
                    counts.atom_pairs_considered += 1;
                    let offset = sub(centers[b], atoms[a].center);
                    // 2 Å total native atom gap, independent of depletant size.
                    let radius = atoms[a].radius + atoms[b].radius + 2.;
                    if prune && projected_miss(offset, mv, radius, bounded)? {
                        counts.atom_pairs_pruned += 1;
                        continue;
                    }
                    counts.atom_pairs_tested += 1;
                    if let Some(hit) = ball_interval(offset, mv, radius, bounded, true)? {
                        hits.push(hit);
                    }
                }
            }
        }
        IntervalSet::from_intervals(hits)
    }
}

#[cfg(test)]
mod tests;
