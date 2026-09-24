//! Complete instantaneous native-entry observer compiled from the frozen Python
//! definition. No physical target, proposal, pocket, or cycle restriction is added.
//!
//! `classify` and `classify_pair` require caller-verified whole-body hard validity
//! (and capture/wall conditions required by that caller). Queried monomer contacts
//! independently reject atomic overlaps, as Python does, but that partial check
//! cannot certify the entire configuration. Entry requires body registration AND
//! at least one prescribed registered monomer bond sharing a reference residue pair.
use crate::math::{Mat3, Pose, Vec3, add, matmul, matvec, rotation, sub, transpose};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Transform {
    position: Vec3,
    rotation: Mat3,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidueAtom {
    center: Vec3,
    radius: f64,
    residue: usize,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    label: String,
    family: String,
    position: Vec3,
    rotation: Mat3,
    native_residue_pairs: Vec<usize>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemberContact {
    member_i: usize,
    member_j: usize,
    directed_class: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Motif {
    id: i64,
    position: Vec3,
    rotation: Mat3,
    member_contacts: Vec<MemberContact>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledDefinition {
    schema: String,
    source_definition_sha256: String,
    source_input_sha256: BTreeMap<String, String>,
    criteria: BTreeMap<String, f64>,
    fixed_poses: Vec<Pose>,
    members: Vec<Transform>,
    monomer_atoms: Vec<ResidueAtom>,
    residue_count: usize,
    references: Vec<Reference>,
    motifs: Vec<Motif>,
}

/// Same supporting directed monomer-bond fields as the independent Python observer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeBond {
    pub members: [usize; 2],
    pub class_label: String,
    pub class_family: String,
    #[serde(rename = "position_error_A")]
    pub position_error_a: f64,
    pub orientation_error_deg: f64,
    #[serde(rename = "minimum_gap_A")]
    pub minimum_gap_a: f64,
    pub shared_reference_residue_pairs_entry: Vec<usize>,
}
/// All matches are retained in frozen catalogue order, including overlapping motifs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeMatch {
    pub motif_id: i64,
    #[serde(rename = "maximum_member_position_error_A")]
    pub maximum_member_position_error_a: f64,
    pub proper_orientation_error_deg: f64,
    pub supporting_member_bonds: Vec<NativeBond>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorMatches {
    pub anchor_index: usize,
    pub matched_motif_ids: Vec<i64>,
    pub matches: Vec<NativeMatch>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeDecision {
    pub native_any: bool,
    pub matched_anchor_indices: Vec<usize>,
    /// Empty anchors are included explicitly.
    pub per_anchor: Vec<AnchorMatches>,
}
#[derive(Clone, Debug)]
pub struct CompleteNativeEntry {
    definition: CompiledDefinition,
    compiled_sha256: String,
    reference_indices: BTreeMap<String, usize>,
    atom_x_order: Vec<usize>,
    maximum_atom_radius: f64,
}

fn norm(v: Vec3) -> f64 {
    // numpy.linalg.norm uses the square-root of the sum of squares, not hypot.
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn validate_pose(pose: Pose) -> Result<()> {
    ensure!(
        pose.position
            .iter()
            .chain(&pose.orientation)
            .all(|x| x.is_finite()),
        "nonfinite native-entry pose"
    );
    let qnorm = pose.orientation.iter().map(|x| x * x).sum::<f64>().sqrt();
    ensure!(
        (qnorm - 1.).abs() < 1e-8,
        "native-entry pose quaternion must be normalized (norm tolerance <1e-8)"
    );
    Ok(())
}
fn validate_transform(position: Vec3, r: Mat3, label: &str) -> Result<()> {
    ensure!(
        position
            .iter()
            .chain(r.iter().flatten())
            .all(|x| x.is_finite()),
        "nonfinite {label} transform"
    );
    let gram = matmul(transpose(r), r);
    ensure!(
        (0..3).all(|i| (0..3).all(|j| (gram[i][j] - if i == j { 1. } else { 0. }).abs() <= 1e-8)),
        "{label} rotation is not orthogonal"
    );
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    ensure!((det - 1.).abs() <= 1e-8, "{label} rotation must be proper");
    Ok(())
}
fn validate_hash(hash: &str, label: &str) -> Result<()> {
    ensure!(
        hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
        "{label} must be a 64-digit SHA256"
    );
    Ok(())
}
fn angle_degrees(a: Mat3, b: Mat3) -> f64 {
    let trace = (0..3)
        .flat_map(|i| (0..3).map(move |j| a[i][j] * b[i][j]))
        .sum::<f64>();
    ((trace - 1.) / 2.).clamp(-1., 1.).acos().to_degrees()
}

impl CompleteNativeEntry {
    /// The exporter authenticates source inputs; the caller must pin the resulting
    /// compiled hash in provenance. No external files are silently reinterpreted.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("read native-entry definition {}", path.display()))?;
        Self::from_bytes(&bytes)
            .with_context(|| format!("native-entry definition {}", path.display()))
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let definition: CompiledDefinition =
            serde_json::from_slice(bytes).context("decode compiled native-entry JSON")?;
        ensure!(
            definition.schema == "native-entry-compiled-v1",
            "unsupported compiled native-entry schema"
        );
        let expected: BTreeMap<String, f64> = [
            ("body_member_position_entry_A", 2.),
            ("body_orientation_entry_deg", 15.),
            ("monomer_position_entry_A", 3.),
            ("monomer_orientation_entry_deg", 20.),
            ("contact_entry_A", 2.),
            ("native_reference_patch_gap_A", 1.),
            ("minimum_shared_native_residue_pairs", 1.),
            ("hard_overlap_tolerance_A", 1e-8),
            ("catalogue_cycle_position_tolerance_A", 1e-6),
            ("catalogue_cycle_angle_tolerance_deg", 1e-6),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        ensure!(
            definition.criteria == expected,
            "unsupported native-entry criteria: require the complete frozen thresholds"
        );
        validate_hash(
            &definition.source_definition_sha256,
            "source_definition_sha256",
        )?;
        ensure!(
            definition
                .source_input_sha256
                .contains_key("tetramer-shape.json"),
            "source_input_sha256 lacks tetramer-shape.json"
        );
        for (key, hash) in &definition.source_input_sha256 {
            ensure!(
                !Path::new(key).is_absolute()
                    && !Path::new(key)
                        .components()
                        .any(|c| matches!(c, std::path::Component::ParentDir)),
                "unsafe source input path {key}"
            );
            validate_hash(hash, key)?;
        }
        ensure!(
            !definition.fixed_poses.is_empty(),
            "native-entry definition has no fixed anchors"
        );
        for (i, pose) in definition.fixed_poses.iter().enumerate() {
            validate_pose(*pose).with_context(|| format!("fixed anchor {i}"))?;
        }
        ensure!(
            !definition.members.is_empty(),
            "native-entry definition has no rigid members"
        );
        for (i, m) in definition.members.iter().enumerate() {
            validate_transform(m.position, m.rotation, &format!("member {i}"))?;
        }
        ensure!(
            !definition.monomer_atoms.is_empty(),
            "native-entry definition has no monomer atoms"
        );
        ensure!(
            definition.residue_count > 0,
            "residue_count must be positive"
        );
        let residue_pairs = definition
            .residue_count
            .checked_mul(definition.residue_count)
            .context("residue-pair count overflow")?;
        for (i, a) in definition.monomer_atoms.iter().enumerate() {
            ensure!(
                a.center.iter().all(|x| x.is_finite()) && a.radius.is_finite() && a.radius > 0.,
                "invalid monomer atom {i}: require finite center and positive radius"
            );
            ensure!(
                a.residue < definition.residue_count,
                "monomer atom {i} residue index out of range"
            );
        }
        let mut reference_indices = BTreeMap::new();
        ensure!(
            !definition.references.is_empty(),
            "native-entry definition has no reference patches"
        );
        for (i, r) in definition.references.iter().enumerate() {
            ensure!(
                !r.label.is_empty() && !r.family.is_empty(),
                "reference {i} has empty label/family"
            );
            ensure!(
                reference_indices.insert(r.label.clone(), i).is_none(),
                "duplicate reference label {}",
                r.label
            );
            validate_transform(r.position, r.rotation, &format!("reference {}", r.label))?;
            ensure!(
                !r.native_residue_pairs.is_empty(),
                "reference {} has an empty residue patch",
                r.label
            );
            ensure!(
                r.native_residue_pairs.iter().all(|&p| p < residue_pairs),
                "reference {} residue pair out of range",
                r.label
            );
            ensure!(
                r.native_residue_pairs.windows(2).all(|p| p[0] < p[1]),
                "reference {} residue pairs must be sorted and unique",
                r.label
            );
        }
        ensure!(
            !definition.motifs.is_empty(),
            "native-entry definition has no motifs"
        );
        let mut motif_ids = BTreeSet::new();
        for m in &definition.motifs {
            ensure!(motif_ids.insert(m.id), "duplicate native motif ID {}", m.id);
            validate_transform(m.position, m.rotation, &format!("motif {}", m.id))?;
            ensure!(
                !m.member_contacts.is_empty(),
                "native motif {} has no member contacts",
                m.id
            );
            for c in &m.member_contacts {
                ensure!(
                    c.member_i < definition.members.len() && c.member_j < definition.members.len(),
                    "native motif {} member index out of range",
                    m.id
                );
                ensure!(
                    reference_indices.contains_key(&c.directed_class),
                    "native motif {} has unknown directed class {}",
                    m.id,
                    c.directed_class
                );
            }
        }
        let mut atom_x_order: Vec<_> = (0..definition.monomer_atoms.len()).collect();
        atom_x_order.sort_by(|&i, &j| {
            definition.monomer_atoms[i].center[0]
                .total_cmp(&definition.monomer_atoms[j].center[0])
                .then(i.cmp(&j))
        });
        let maximum_atom_radius = definition
            .monomer_atoms
            .iter()
            .map(|a| a.radius)
            .fold(0., f64::max);
        Ok(Self {
            definition,
            compiled_sha256: format!("{:x}", Sha256::digest(bytes)),
            reference_indices,
            atom_x_order,
            maximum_atom_radius,
        })
    }
    pub fn definition_sha256(&self) -> &str {
        &self.definition.source_definition_sha256
    }
    pub fn compiled_sha256(&self) -> &str {
        &self.compiled_sha256
    }
    pub fn shape_sha256(&self) -> &str {
        &self.definition.source_input_sha256["tetramer-shape.json"]
    }
    pub fn fixed_poses(&self) -> &[Pose] {
        &self.definition.fixed_poses
    }

    /// Caller must establish whole-body hard validity first. No periodic image,
    /// region, wall, reference-pocket, or cycle restriction is introduced here.
    pub fn classify(&self, moving: Pose) -> Result<NativeDecision> {
        let mut per_anchor = Vec::with_capacity(self.fixed_poses().len());
        for (anchor_index, anchor) in self.fixed_poses().iter().enumerate() {
            let matches = self
                .classify_pair(*anchor, moving)
                .with_context(|| format!("native-entry anchor {anchor_index}"))?;
            per_anchor.push(AnchorMatches {
                anchor_index,
                matched_motif_ids: matches.iter().map(|m| m.motif_id).collect(),
                matches,
            });
        }
        let matched_anchor_indices: Vec<_> = per_anchor
            .iter()
            .filter(|a| !a.matches.is_empty())
            .map(|a| a.anchor_index)
            .collect();
        Ok(NativeDecision {
            native_any: !matched_anchor_indices.is_empty(),
            matched_anchor_indices,
            per_anchor,
        })
    }
    /// Directed open-space anchor→moving matches, in frozen catalogue order.
    /// Whole-body hard validity is a caller precondition, as for `classify`.
    pub fn classify_pair(&self, anchor: Pose, moving: Pose) -> Result<Vec<NativeMatch>> {
        validate_pose(anchor).context("native-entry anchor pose")?;
        validate_pose(moving).context("native-entry moving pose")?;
        let inverse_anchor = transpose(rotation(anchor.orientation));
        let r = matmul(inverse_anchor, rotation(moving.orientation));
        let d = matvec(inverse_anchor, sub(moving.position, anchor.position));
        let mut matches = Vec::new();
        let mut cache: BTreeMap<(usize, usize, usize), Option<NativeBond>> = BTreeMap::new();
        for motif in &self.definition.motifs {
            let body_angle = angle_degrees(r, motif.rotation);
            if body_angle > 15. {
                continue;
            }
            let body_error = self
                .definition
                .members
                .iter()
                .map(|m| {
                    norm(sub(
                        add(matvec(r, m.position), d),
                        add(matvec(motif.rotation, m.position), motif.position),
                    ))
                })
                .fold(0., f64::max);
            if body_error > 2. {
                continue;
            }
            let mut bonds = Vec::new();
            for contact in &motif.member_contacts {
                let index = self.reference_indices[&contact.directed_class];
                let key = (contact.member_i, contact.member_j, index);
                if !cache.contains_key(&key) {
                    let i = &self.definition.members[contact.member_i];
                    let j = &self.definition.members[contact.member_j];
                    let mr = matmul(matmul(transpose(i.rotation), r), j.rotation);
                    let md = matvec(
                        transpose(i.rotation),
                        sub(add(matvec(r, j.position), d), i.position),
                    );
                    let reference = &self.definition.references[index];
                    let position_error = norm(sub(md, reference.position));
                    let angle = angle_degrees(mr, reference.rotation);
                    let mut bond = None;
                    if position_error <= 3. && angle <= 20. {
                        let (minimum_gap, pairs) = self.contacts(md, mr).with_context(|| {
                            format!(
                                "motif {} members {}→{} reference {}",
                                motif.id, contact.member_i, contact.member_j, reference.label
                            )
                        })?;
                        let common: Vec<_> = reference
                            .native_residue_pairs
                            .iter()
                            .copied()
                            .filter(|p| pairs.contains(p))
                            .collect();
                        if !common.is_empty() {
                            bond = Some(NativeBond {
                                members: [contact.member_i, contact.member_j],
                                class_label: reference.label.clone(),
                                class_family: reference.family.clone(),
                                position_error_a: position_error,
                                orientation_error_deg: angle,
                                minimum_gap_a: minimum_gap
                                    .context("shared residue contact has no minimum gap")?,
                                shared_reference_residue_pairs_entry: common,
                            });
                        }
                    }
                    cache.insert(key, bond);
                }
                if let Some(bond) = &cache[&key] {
                    bonds.push(bond.clone());
                }
            }
            if !bonds.is_empty() {
                matches.push(NativeMatch {
                    motif_id: motif.id,
                    maximum_member_position_error_a: body_error,
                    proper_orientation_error_deg: body_angle,
                    supporting_member_bonds: bonds,
                });
            }
        }
        Ok(matches)
    }
    fn contacts(&self, d: Vec3, r: Mat3) -> Result<(Option<f64>, BTreeSet<usize>)> {
        let mut minimum_gap: Option<f64> = None;
        let mut pairs = BTreeSet::new();
        let atoms = &self.definition.monomer_atoms;
        // Same broad radius as Python's cKDTree. Sorted x only prunes candidates;
        // exact three-dimensional distances and atom radii decide contacts.
        let reach = 2. + 2. * self.maximum_atom_radius;
        for moving in atoms {
            let center = add(matvec(r, moving.center), d);
            let guard = 32. * f64::EPSILON * (1. + reach + center[0].abs());
            let first = self
                .atom_x_order
                .partition_point(|&i| atoms[i].center[0] < center[0] - reach - guard);
            let end = self
                .atom_x_order
                .partition_point(|&i| atoms[i].center[0] <= center[0] + reach + guard);
            for &i in &self.atom_x_order[first..end] {
                let anchor = &atoms[i];
                let distance = norm(sub(anchor.center, center));
                if distance > reach {
                    continue;
                }
                let gap = distance - anchor.radius - moving.radius;
                if gap < -1e-8 {
                    bail!(
                        "native observer requires hard-valid poses: atom gap {gap:.17e} Å < -1e-8 Å"
                    );
                }
                if gap <= 2. {
                    minimum_gap = Some(minimum_gap.map_or(gap, |v| v.min(gap)));
                    pairs.insert(anchor.residue * self.definition.residue_count + moving.residue);
                }
            }
        }
        Ok((minimum_gap, pairs))
    }
}

#[cfg(test)]
mod tests {
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
        "fixed_poses":[pose([0.,0.,0.]),pose([10.,0.,0.])],"members":[{"position":[0.,0.,0.],"rotation":IDENTITY}],
        "monomer_atoms":[{"center":[0.,0.,0.],"radius":1.,"residue":0}],"residue_count":1,
        "references":[{"label":"A","family":"A","position":[2.,0.,0.],"rotation":IDENTITY,"native_residue_pairs":[0]}],
        "motifs":[{"id":7,"position":[2.,0.,0.],"rotation":IDENTITY,"member_contacts":[{"member_i":0,"member_j":0,"directed_class":"A"}]}]})
    }
    fn model(v: &Value) -> CompleteNativeEntry {
        CompleteNativeEntry::from_bytes(&serde_json::to_vec(v).unwrap()).unwrap()
    }
    #[test]
    fn exact_threshold_and_empty_matches() {
        let m = model(&fixture());
        for x in [2., 4.] {
            assert_eq!(
                m.classify(pose([x, 0., 0.]))
                    .unwrap()
                    .matched_anchor_indices,
                vec![0]
            );
        }
        assert!(!m.classify(pose([4. + 1e-10, 0., 0.])).unwrap().native_any);
        assert_eq!(
            m.classify(pose([12., 0., 0.]))
                .unwrap()
                .matched_anchor_indices,
            vec![1]
        );
        let matches = m
            .classify_pair(pose([0., 0., 0.]), pose([2., 0., 0.]))
            .unwrap();
        assert_eq!(
            matches[0].supporting_member_bonds[0].shared_reference_residue_pairs_entry,
            vec![0]
        );
        assert_eq!(matches[0].supporting_member_bonds[0].minimum_gap_a, 0.);
    }
    #[test]
    fn all_motifs_retained_and_pose_frame_invariance() {
        let mut v = fixture();
        let mut other = v["motifs"][0].clone();
        other["id"] = json!(3);
        v["motifs"].as_array_mut().unwrap().push(other);
        let m = model(&v);
        let r = cayley([0.2, -0.1, 0.3]);
        let anchor = Pose {
            position: [8., -3., 5.],
            orientation: quaternion(r),
        };
        let moving = Pose {
            position: add(anchor.position, matvec(r, [2., 0., 0.])),
            orientation: anchor.orientation,
        };
        let matches = m.classify_pair(anchor, moving).unwrap();
        assert_eq!(
            matches.iter().map(|x| x.motif_id).collect::<Vec<_>>(),
            vec![7, 3]
        );
        assert!(matches[0].maximum_member_position_error_a < 1e-12);
        let mut negative = moving;
        negative.orientation = negative.orientation.map(|x| -x);
        assert_eq!(m.classify_pair(anchor, negative).unwrap().len(), 2);
    }
    #[test]
    fn body_registration_is_not_sufficient() {
        let mut v = fixture();
        v["references"][0]["position"] = json!([20., 0., 0.]);
        assert!(!model(&v).classify(pose([2., 0., 0.])).unwrap().native_any);
        v = fixture();
        v["residue_count"] = json!(2);
        v["references"][0]["native_residue_pairs"] = json!([1]);
        assert!(!model(&v).classify(pose([2., 0., 0.])).unwrap().native_any);
        v = fixture();
        v["references"][0]["rotation"] = json!(cayley([0., 0., 0.5]));
        assert!(!model(&v).classify(pose([2., 0., 0.])).unwrap().native_any);
    }
    #[test]
    fn body_angle_and_contact_cutoffs() {
        let m = model(&fixture());
        let rotated = |angle: f64| Pose {
            position: [2., 0., 0.],
            orientation: [
                (angle.to_radians() / 2.).cos(),
                0.,
                0.,
                (angle.to_radians() / 2.).sin(),
            ],
        };
        assert!(m.classify(rotated(15. - 1e-8)).unwrap().native_any);
        assert!(!m.classify(rotated(15. + 1e-8)).unwrap().native_any);
        let mut v = fixture();
        v["motifs"][0]["position"] = json!([4., 0., 0.]);
        v["references"][0]["position"] = json!([4., 0., 0.]);
        let m = model(&v);
        assert!(m.classify(pose([4., 0., 0.])).unwrap().native_any);
        assert!(!m.classify(pose([4. + 1e-10, 0., 0.])).unwrap().native_any);
    }
    #[test]
    fn overlap_and_invalid_definition_fail_explicitly() {
        let m = model(&fixture());
        assert!(
            format!("{:#}", m.classify(pose([1.9, 0., 0.])).unwrap_err()).contains("hard-valid")
        );
        let mut bad = pose([2., 0., 0.]);
        bad.orientation = [0.; 4];
        assert!(m.classify(bad).is_err());
        let mut v = fixture();
        v["motifs"][0]["member_contacts"][0]["directed_class"] = json!("missing");
        assert!(
            CompleteNativeEntry::from_bytes(&serde_json::to_vec(&v).unwrap())
                .unwrap_err()
                .to_string()
                .contains("unknown directed class")
        );
        v = fixture();
        v["members"][0]["rotation"] = json!([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        assert!(CompleteNativeEntry::from_bytes(&serde_json::to_vec(&v).unwrap()).is_err());
        v = fixture();
        v["criteria"]["contact_entry_A"] = json!(2.1);
        assert!(
            CompleteNativeEntry::from_bytes(&serde_json::to_vec(&v).unwrap())
                .unwrap_err()
                .to_string()
                .contains("criteria")
        );
    }
    #[test]
    fn nonzero_member_frames_and_directed_residue_pairs() {
        let mut v = fixture();
        v["members"] = json!([{"position":[1.,2.,0.],"rotation":cayley([0.,0.,1.])}]);
        v["references"][0]["position"] = json!([0., -2., 0.]);
        assert!(model(&v).classify(pose([2., 0., 0.])).unwrap().native_any);
        v = fixture();
        v["residue_count"] = json!(2);
        v["monomer_atoms"] = json!([{"center":[0.,0.,0.],"radius":1.,"residue":0},{"center":[10.,0.,0.],"radius":1.,"residue":1}]);
        v["motifs"][0]["position"] = json!([-12., 0., 0.]);
        v["references"][0]["position"] = json!([-12., 0., 0.]);
        v["references"][0]["native_residue_pairs"] = json!([1]);
        assert_eq!(
            model(&v)
                .classify_pair(pose([0., 0., 0.]), pose([-12., 0., 0.]))
                .unwrap()
                .len(),
            1
        );
        v["references"][0]["native_residue_pairs"] = json!([2]);
        assert!(
            model(&v)
                .classify_pair(pose([0., 0., 0.]), pose([-12., 0., 0.]))
                .unwrap()
                .is_empty()
        );
    }
}
