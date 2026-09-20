//! Frozen Gaussian relative-pose proposals with an explicit periodic null move.
//!
//! The chosen other-particle anchor is retained as an auxiliary label. Learned
//! displacements outside its unique lab-frame image cube are null proposals;
//! they are never wrapped, retried, or renormalized conditionally on the box.

use anyhow::{Context, Result, bail, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, fs, path::Path};

use crate::math::{
    Mat3, Pose, Vec3, cayley, matmul, matvec, minimum_image, quaternion, rotation, transpose, wrap,
};

type Vec6 = [f64; 6];
type Mat6 = [[f64; 6]; 6];

/// Public, lossless Gaussian chart parameters in anchor-body-relative coordinates.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GaussianComponentParameters {
    pub anchor_position: Vec3,
    pub anchor_rotation: Mat3,
    pub mean: [f64; 6],
    pub covariance: [[f64; 6]; 6],
    pub weight: f64,
}

#[derive(Deserialize)]
struct RawAnchor {
    position: Vec3,
    rotation: Mat3,
}

#[derive(Deserialize)]
struct RawModel {
    angular_length: f64,
    anchors: Vec<RawAnchor>,
    means: Vec<Vec6>,
    #[serde(default)]
    covariances: Option<Vec<Mat6>>,
    #[serde(default)]
    scales: Option<Vec<Mat6>>,
    #[serde(default)]
    dfs: Option<Vec<Option<f64>>>,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    shape_sha256: String,
    coordinate_convention: String,
}

#[derive(Clone, Debug)]
struct Gaussian {
    mean: Vec6,
    lower: Mat6,
    log_normalizer: f64,
    anchor_position: Vec3,
    anchor_rotation: Mat3,
    weight: f64,
    log_weight: f64,
}

fn finite3(v: Vec3) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn validate_rotation(matrix: Mat3) -> Result<()> {
    ensure!(
        matrix.iter().flatten().all(|x| x.is_finite()),
        "Nonfinite rotation"
    );
    let gram = matmul(transpose(matrix), matrix);
    for (i, row) in gram.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            ensure!(
                (value - if i == j { 1.0 } else { 0.0 }).abs() <= 1e-10,
                "Rotation matrix is not orthogonal"
            );
        }
    }
    let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    ensure!((det - 1.0).abs() <= 1e-10, "Rotation matrix must be proper");
    Ok(())
}

fn validate_pose(pose: &Pose) -> Result<()> {
    // Geometry and proposal must accept the same ingress tolerance. The common
    // rotation() normalizes before constructing a rigid transform.
    pose.validate()
}

fn prepare_cholesky(covariance: Mat6) -> Result<(Mat6, f64)> {
    ensure!(
        covariance.iter().flatten().all(|v| v.is_finite()),
        "Nonfinite Gaussian covariance"
    );
    let scale = covariance
        .iter()
        .flatten()
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    ensure!(scale > 0.0, "Gaussian covariance is not positive definite");
    let mut symmetric = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            ensure!(
                (covariance[i][j] - covariance[j][i]).abs()
                    <= 1e-12 * (scale + covariance[j][i].abs()),
                "Gaussian covariance is not symmetric"
            );
            symmetric[i][j] = 0.5 * (covariance[i][j] + covariance[j][i]);
        }
    }
    let mut lower = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut remainder = symmetric[i][j];
            for k in 0..j {
                remainder -= lower[i][k] * lower[j][k];
            }
            if i == j {
                ensure!(
                    remainder.is_finite() && remainder > 0.0,
                    "Gaussian covariance is not strictly positive definite (no jitter is added)"
                );
                lower[i][j] = remainder.sqrt();
            } else {
                lower[i][j] = remainder / lower[j][j];
                ensure!(
                    lower[i][j].is_finite(),
                    "Unrepresentable Gaussian Cholesky factor"
                );
            }
        }
    }
    let log_normalizer = -3.0 * (2.0 * PI).ln() - (0..6).map(|i| lower[i][i].ln()).sum::<f64>();
    ensure!(
        log_normalizer.is_finite(),
        "Unrepresentable Gaussian normalizer"
    );
    Ok((lower, log_normalizer))
}

impl Gaussian {
    fn log_density(&self, latent: Vec6) -> f64 {
        let mut residual = [0.0; 6];
        let mut norm2 = 0.0;
        for i in 0..6 {
            let mut value = latent[i] - self.mean[i];
            for (j, previous) in residual.iter().enumerate().take(i) {
                value -= self.lower[i][j] * previous;
            }
            residual[i] = value / self.lower[i][i];
            norm2 += residual[i] * residual[i];
        }
        if !norm2.is_finite() {
            f64::NEG_INFINITY
        } else {
            self.log_normalizer - 0.5 * norm2
        }
    }

    fn draw(&self, rng: &mut StdRng) -> Option<Vec6> {
        let normal: Vec6 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let result: Vec6 = std::array::from_fn(|i| {
            self.mean[i] + (0..=i).map(|j| self.lower[i][j] * normal[j]).sum::<f64>()
        });
        result.iter().all(|x| x.is_finite()).then_some(result)
    }
}

/// The finite Cayley coordinate; a measure-zero pi seam has Gaussian density zero.
fn cayley_inverse(matrix: Mat3) -> Option<Vec3> {
    let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    let mut vector = [0.0; 3];
    let scalar;
    if trace > 0.0 {
        scalar = 0.5 * (1.0 + trace).sqrt();
        vector = [
            matrix[2][1] - matrix[1][2],
            matrix[0][2] - matrix[2][0],
            matrix[1][0] - matrix[0][1],
        ];
        for v in &mut vector {
            *v /= 4.0 * scalar;
        }
    } else {
        let mut i = 0;
        for j in 1..3 {
            if matrix[j][j] > matrix[i][i] {
                i = j;
            }
        }
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let factor = 2.0
            * (1.0 + matrix[i][i] - matrix[j][j] - matrix[k][k])
                .max(0.0)
                .sqrt();
        if factor == 0.0 {
            return None;
        }
        vector[i] = 0.25 * factor;
        vector[j] = (matrix[i][j] + matrix[j][i]) / factor;
        vector[k] = (matrix[i][k] + matrix[k][i]) / factor;
        scalar = (matrix[k][j] - matrix[j][k]) / factor;
    }
    if scalar == 0.0 {
        return None;
    }
    let c = vector.map(|v| v / scalar);
    finite3(c).then_some(c)
}

fn log_haar_jacobian(c: Vec3) -> f64 {
    let norm = c.iter().fold(1.0_f64, |a, v| a.hypot(*v));
    -2.0 * PI.ln() - 4.0 * norm.ln()
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let high = a.max(b);
    high + ((a - high).exp() + (b - high).exp()).ln()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalBranch {
    Uniform,
    Learned,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProposalOutcome {
    pub moving_index: usize,
    pub anchor_index: usize,
    pub branch: ProposalBranch,
    pub candidate: Option<Pose>,
    pub null_reason: Option<String>,
    pub old_log_density: f64,
    pub new_log_density: Option<f64>,
    pub log_reverse_forward: Option<f64>,
    pub component_index: Option<usize>,
}

/// Immutable normalized Gaussian mixture plus periodic uniform/null extension.
#[derive(Clone, Debug)]
pub struct FrozenRelativePoseProposal {
    components: Vec<Gaussian>,
    angular_length: f64,
    log_coordinate_scale: f64,
    box_lengths: Vec3,
    uniform_weight: f64,
    log_uniform_density: f64,
    log_learned_weight: f64,
    shape_sha256: String,
    periodic: bool,
}

impl FrozenRelativePoseProposal {
    /// Construct an open-space normalized mixture without a serialized atlas.
    /// Empty components are supported exactly when the proposal is uniform-only.
    pub fn from_components_open(
        parameters: Vec<GaussianComponentParameters>,
        angular_length: f64,
        uniform_cube_lengths: Vec3,
        uniform_weight: f64,
        shape_sha256: &str,
        expected_shape_sha256: &str,
    ) -> Result<Self> {
        ensure!(
            shape_sha256.len() == 64
                && shape_sha256.bytes().all(|x| x.is_ascii_hexdigit())
                && shape_sha256 == expected_shape_sha256,
            "Proposal shape SHA256 does not match physical geometry"
        );
        ensure!(
            uniform_cube_lengths
                .iter()
                .all(|x| x.is_finite() && *x > 0.),
            "Uniform cube lengths must be finite and positive"
        );
        ensure!(
            angular_length.is_finite() && angular_length > 0.,
            "Invalid angular_length"
        );
        ensure!(
            uniform_weight.is_finite() && uniform_weight > 0. && uniform_weight <= 1.,
            "Uniform weight must lie in (0,1]"
        );
        ensure!(
            !parameters.is_empty() || uniform_weight == 1.,
            "An empty mixture must have unit uniform weight"
        );
        let weight_sum: f64 = parameters.iter().map(|c| c.weight).sum();
        ensure!(
            parameters.is_empty() || (weight_sum - 1.).abs() <= 2e-12,
            "Component weights must sum to one"
        );
        let mut components = Vec::with_capacity(parameters.len());
        for p in parameters {
            ensure!(
                finite3(p.anchor_position) && p.mean.iter().all(|x| x.is_finite()),
                "Nonfinite Gaussian chart position or mean"
            );
            validate_rotation(p.anchor_rotation)?;
            ensure!(
                p.weight.is_finite() && p.weight > 0.,
                "Invalid Gaussian weight"
            );
            let (lower, log_normalizer) = prepare_cholesky(p.covariance)?;
            let weight = p.weight / weight_sum;
            components.push(Gaussian {
                mean: p.mean,
                lower,
                log_normalizer,
                anchor_position: p.anchor_position,
                anchor_rotation: p.anchor_rotation,
                weight,
                log_weight: weight.ln(),
            });
        }
        Ok(Self {
            components,
            angular_length,
            log_coordinate_scale: 3. * angular_length.ln(),
            box_lengths: uniform_cube_lengths,
            uniform_weight,
            log_uniform_density: uniform_weight.ln()
                - uniform_cube_lengths.iter().map(|x| x.ln()).sum::<f64>(),
            log_learned_weight: if uniform_weight == 1. {
                f64::NEG_INFINITY
            } else {
                (-uniform_weight).ln_1p()
            },
            shape_sha256: shape_sha256.into(),
            periodic: false,
        })
    }

    /// The K=0 model, independent of any learned model file.
    pub fn uniform_only_open(uniform_cube_lengths: Vec3, shape_sha256: &str) -> Result<Self> {
        Self::from_components_open(
            Vec::new(),
            1.,
            uniform_cube_lengths,
            1.,
            shape_sha256,
            shape_sha256,
        )
    }

    pub fn component_parameters(&self) -> Vec<GaussianComponentParameters> {
        self.components
            .iter()
            .map(|c| GaussianComponentParameters {
                anchor_position: c.anchor_position,
                anchor_rotation: c.anchor_rotation,
                mean: c.mean,
                covariance: std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        (0..=i.min(j)).map(|d| c.lower[i][d] * c.lower[j][d]).sum()
                    })
                }),
                weight: c.weight,
            })
            .collect()
    }

    pub fn from_path(
        path: impl AsRef<Path>,
        box_lengths: Vec3,
        uniform_weight: f64,
        expected_shape_sha256: &str,
    ) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Read model {}", path.as_ref().display()))?;
        Self::from_json_str(&text, box_lengths, uniform_weight, expected_shape_sha256)
    }

    pub fn from_json_str(
        text: &str,
        box_lengths: Vec3,
        uniform_weight: f64,
        expected_shape_sha256: &str,
    ) -> Result<Self> {
        let raw: RawModel =
            serde_json::from_str(text).context("Read frozen Gaussian pose model")?;
        ensure!(
            raw.coordinate_convention == "anchor-body-relative",
            "Model coordinate_convention must be anchor-body-relative"
        );
        ensure!(
            raw.shape_sha256.len() == 64 && raw.shape_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "Model shape_sha256 must be a 64-digit SHA256"
        );
        ensure!(
            raw.shape_sha256 == expected_shape_sha256,
            "Frozen model shape SHA256 does not match physical geometry"
        );
        ensure!(
            box_lengths.iter().all(|v| v.is_finite() && *v > 0.0),
            "Box lengths must be finite and positive"
        );
        ensure!(
            uniform_weight.is_finite() && uniform_weight > 0.0 && uniform_weight <= 1.0,
            "Uniform weight must lie in (0,1]"
        );
        ensure!(
            raw.angular_length.is_finite() && raw.angular_length > 0.0,
            "Invalid angular_length"
        );
        let count = raw.anchors.len();
        ensure!(
            count > 0 && raw.means.len() == count,
            "One mean is required per nonempty component"
        );
        if let Some(dfs) = &raw.dfs {
            ensure!(
                dfs.len() == count,
                "One dfs entry is required per component"
            );
            ensure!(
                dfs.iter().all(Option::is_none),
                "Student-t components are unsupported: this Rust model is Gaussian-only"
            );
        }
        let covariances = match (raw.covariances, raw.scales) {
            (Some(c), None) | (None, Some(c)) => c,
            (Some(_), Some(_)) => bail!("Specify covariances or scales, never both"),
            (None, None) => bail!("Missing Gaussian covariances"),
        };
        ensure!(
            covariances.len() == count,
            "One covariance is required per component"
        );
        let mut weights = raw
            .weights
            .unwrap_or_else(|| vec![1.0 / count as f64; count]);
        ensure!(
            weights.len() == count && weights.iter().all(|w| w.is_finite() && *w > 0.0),
            "Component weights must be finite, positive and normalized"
        );
        let weight_sum: f64 = weights.iter().sum();
        ensure!(
            (weight_sum - 1.0).abs() <= 2e-12,
            "Component weights must sum to one"
        );
        for w in &mut weights {
            *w /= weight_sum;
        }
        let mut components = Vec::with_capacity(count);
        for (i, ((anchor, mean), covariance)) in raw
            .anchors
            .into_iter()
            .zip(raw.means)
            .zip(covariances)
            .enumerate()
        {
            ensure!(
                finite3(anchor.position),
                "Nonfinite component anchor position"
            );
            validate_rotation(anchor.rotation)?;
            ensure!(
                mean.iter().all(|v| v.is_finite()),
                "Nonfinite Gaussian mean"
            );
            let (lower, log_normalizer) =
                prepare_cholesky(covariance).with_context(|| format!("Gaussian component {i}"))?;
            components.push(Gaussian {
                mean,
                lower,
                log_normalizer,
                anchor_position: anchor.position,
                anchor_rotation: anchor.rotation,
                weight: weights[i],
                log_weight: weights[i].ln(),
            });
        }
        Ok(Self {
            components,
            angular_length: raw.angular_length,
            log_coordinate_scale: 3.0 * raw.angular_length.ln(),
            box_lengths,
            uniform_weight,
            log_uniform_density: uniform_weight.ln()
                - box_lengths.iter().map(|v| v.ln()).sum::<f64>(),
            log_learned_weight: if uniform_weight == 1.0 {
                f64::NEG_INFINITY
            } else {
                (-uniform_weight).ln_1p()
            },
            shape_sha256: raw.shape_sha256,
            periodic: true,
        })
    }

    /// Open-space Gaussian mixture with a defensive uniform centered cube.
    /// The physical wall is checked by the caller, without retrying rejected draws.
    pub fn from_json_str_open(
        text: &str,
        uniform_cube_lengths: Vec3,
        uniform_weight: f64,
        expected_shape_sha256: &str,
    ) -> Result<Self> {
        let mut result = Self::from_json_str(
            text,
            uniform_cube_lengths,
            uniform_weight,
            expected_shape_sha256,
        )?;
        result.periodic = false;
        Ok(result)
    }

    pub fn is_periodic(&self) -> bool {
        self.periodic
    }
    pub fn angular_length(&self) -> f64 {
        self.angular_length
    }

    /// A fresh immutable mixture, shifting each mean by its original Cholesky L times offset.
    pub fn with_whitened_mean_offsets(&self, offsets: &[[f64; 6]]) -> Result<Self> {
        ensure!(
            offsets.len() == self.components.len(),
            "One mean offset required per component"
        );
        ensure!(
            offsets.iter().flatten().all(|x| x.is_finite()),
            "Nonfinite mean offset"
        );
        let mut result = self.clone();
        for (component, offset) in result.components.iter_mut().zip(offsets) {
            for i in 0..6 {
                component.mean[i] += (0..=i)
                    .map(|j| component.lower[i][j] * offset[j])
                    .sum::<f64>();
            }
            ensure!(
                component.mean.iter().all(|x| x.is_finite()),
                "Unrepresentable shifted mean"
            );
        }
        Ok(result)
    }

    /// Standardized component residual; pi-chart seams and numerical overflow return None.
    pub fn component_residual(
        &self,
        relative_t: Vec3,
        relative_r: Mat3,
        k: usize,
    ) -> Option<[f64; 6]> {
        let component = self.components.get(k)?;
        let c = cayley_inverse(matmul(relative_r, transpose(component.anchor_rotation)))?;
        let mut residual = [0.; 6];
        for i in 0..6 {
            let latent = if i < 3 {
                relative_t[i] - component.anchor_position[i]
            } else {
                self.angular_length * c[i - 3]
            };
            let value = latent
                - component.mean[i]
                - (0..i)
                    .map(|j| component.lower[i][j] * residual[j])
                    .sum::<f64>();
            residual[i] = value / component.lower[i][i];
        }
        residual.iter().all(|x| x.is_finite()).then_some(residual)
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }
    /// Fixed dictionary probabilities for auxiliary component-label births.
    pub fn component_weights(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.weight).collect()
    }

    /// Append one normalized Gaussian chart at every current memory pose.
    /// Atlas order and weights are fixed: original base first with mass 1-m,
    /// then M memory slots with mass m/M each. Pose changes affect chart centers
    /// and orientations only. Build from the ORIGINAL frozen base every time.
    /// Covariance is full rank, independent translation/Cayley coordinates;
    /// the angular input is a per-axis small-angle standard deviation.
    pub fn with_contact_components(
        &self,
        poses: &[Pose],
        mass: f64,
        translation_std: f64,
        small_angle_std_degrees: f64,
    ) -> Result<Self> {
        ensure!(
            !poses.is_empty(),
            "Contact-memory dictionary cannot be empty"
        );
        ensure!(
            mass.is_finite() && mass > 0. && mass < 1.,
            "Contact-memory mass must lie strictly between zero and one"
        );
        ensure!(
            [translation_std, small_angle_std_degrees]
                .iter()
                .all(|x| x.is_finite() && *x > 0.),
            "Invalid contact-memory Gaussian widths"
        );
        let angular_std = self.angular_length * small_angle_std_degrees.to_radians() / 2.;
        let mut covariance = [[0.; 6]; 6];
        for (i, row) in covariance.iter_mut().enumerate() {
            row[i] = if i < 3 {
                translation_std * translation_std
            } else {
                angular_std * angular_std
            };
        }
        let (lower, log_normalizer) = prepare_cholesky(covariance)?;
        let memory_weight = mass / poses.len() as f64;
        ensure!(
            memory_weight.is_finite() && memory_weight > 0.,
            "Unrepresentable contact-memory component weight"
        );
        let mut result = self.clone();
        for component in &mut result.components {
            component.weight *= 1. - mass;
            ensure!(
                component.weight.is_finite() && component.weight > 0.,
                "Unrepresentable contact-memory base weight"
            );
            component.log_weight = component.weight.ln();
        }
        for pose in poses {
            pose.validate()?;
            result.components.push(Gaussian {
                mean: [0.; 6],
                lower,
                log_normalizer,
                anchor_position: pose.position,
                anchor_rotation: rotation(pose.orientation),
                weight: memory_weight,
                log_weight: memory_weight.ln(),
            });
        }
        Ok(result)
    }

    /// Ordered active components with replacement, each with weight 1/K.
    /// Duplicate labels are distinct auxiliary slots; never sort or merge them.
    pub fn selected_components(&self, labels: &[usize]) -> Result<Self> {
        ensure!(
            !labels.is_empty() && labels.iter().all(|&i| i < self.components.len()),
            "Invalid active component labels"
        );
        let mut result = self.clone();
        let weight = 1. / labels.len() as f64;
        result.components = labels
            .iter()
            .map(|&i| {
                let mut component = self.components[i].clone();
                component.weight = weight;
                component.log_weight = weight.ln();
                component
            })
            .collect();
        Ok(result)
    }
    pub fn shape_sha256(&self) -> &str {
        &self.shape_sha256
    }
    pub fn box_lengths(&self) -> Vec3 {
        self.box_lengths
    }
    pub fn uniform_weight(&self) -> f64 {
        self.uniform_weight
    }

    fn relative_log_density_unchecked(&self, position: Vec3, orientation: Mat3) -> f64 {
        let mut total = f64::NEG_INFINITY;
        for component in &self.components {
            let Some(c) = cayley_inverse(matmul(orientation, transpose(component.anchor_rotation)))
            else {
                continue;
            };
            let latent: Vec6 = std::array::from_fn(|i| {
                if i < 3 {
                    position[i] - component.anchor_position[i]
                } else {
                    self.angular_length * c[i - 3]
                }
            });
            let gaussian = component.log_density(latent);
            total = log_add(
                total,
                component.log_weight + gaussian + self.log_coordinate_scale - log_haar_jacobian(c),
            );
        }
        total
    }

    /// Learned G alone in anchor-body relative coordinates, before the box null.
    pub fn relative_log_density(
        &self,
        relative_position: Vec3,
        relative_rotation: Mat3,
    ) -> Result<f64> {
        ensure!(finite3(relative_position), "Nonfinite relative center");
        validate_rotation(relative_rotation)?;
        Ok(self.relative_log_density_unchecked(relative_position, relative_rotation))
    }

    /// Continuous off-diagonal subdensity eta/V+(1-eta)G at the unique image.
    pub fn log_density(&self, pose: &Pose, anchor: &Pose) -> Result<f64> {
        validate_pose(pose)?;
        validate_pose(anchor)?;
        let anchor_rotation = rotation(anchor.orientation);
        let inverse_anchor = transpose(anchor_rotation);
        let raw_displacement = std::array::from_fn(|i| pose.position[i] - anchor.position[i]);
        let displacement = if self.periodic {
            minimum_image(raw_displacement, self.box_lengths)
        } else {
            raw_displacement
        };
        let relative_t = matvec(inverse_anchor, displacement);
        let relative_r = matmul(inverse_anchor, rotation(pose.orientation));
        let learned = self.relative_log_density_unchecked(relative_t, relative_r);
        Ok(log_add(
            if self.periodic
                || (0..3).all(|i| {
                    pose.position[i] >= -0.5 * self.box_lengths[i]
                        && pose.position[i] < 0.5 * self.box_lengths[i]
                })
            {
                self.log_uniform_density
            } else {
                f64::NEG_INFINITY
            },
            self.log_learned_weight + learned,
        ))
    }

    /// One proposal, with one retained uniform anchor label and no retry loop.
    pub fn propose(
        &self,
        rng: &mut StdRng,
        poses: &[Pose],
        moving_index: usize,
    ) -> Result<ProposalOutcome> {
        ensure!(
            poses.len() >= 2 && moving_index < poses.len(),
            "Require a valid moving index and at least two bodies"
        );
        let selected = rng.random_range(0..poses.len() - 1);
        let anchor_index = if selected < moving_index {
            selected
        } else {
            selected + 1
        };
        let anchor = &poses[anchor_index];
        let old_log_density = self.log_density(&poses[moving_index], anchor)?;
        let branch = if rng.random::<f64>() < self.uniform_weight {
            ProposalBranch::Uniform
        } else {
            ProposalBranch::Learned
        };
        let mut outcome = ProposalOutcome {
            moving_index,
            anchor_index,
            branch,
            candidate: None,
            null_reason: None,
            old_log_density,
            new_log_density: None,
            log_reverse_forward: None,
            component_index: None,
        };
        let candidate = match branch {
            ProposalBranch::Uniform => {
                let position = std::array::from_fn(|i| {
                    (rng.random::<f64>() - if self.periodic { 0. } else { 0.5 })
                        * self.box_lengths[i]
                });
                let mut orientation: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
                let norm = orientation.iter().fold(0.0_f64, |a, v| a.hypot(*v));
                if !norm.is_finite() || norm == 0.0 {
                    outcome.null_reason = Some("uniform_quaternion_unrepresentable".into());
                    return Ok(outcome);
                }
                for v in &mut orientation {
                    *v /= norm;
                }
                Pose {
                    position,
                    orientation,
                }
            }
            ProposalBranch::Learned => {
                let draw = rng.random::<f64>();
                let mut cumulative = 0.0;
                let mut index = self.components.len() - 1;
                for (i, component) in self.components.iter().enumerate() {
                    cumulative += component.weight;
                    if draw < cumulative {
                        index = i;
                        break;
                    }
                }
                outcome.component_index = Some(index);
                let component = &self.components[index];
                let Some(latent) = component.draw(rng) else {
                    outcome.null_reason = Some("learned_numerical_null".into());
                    return Ok(outcome);
                };
                let relative_t = std::array::from_fn(|i| component.anchor_position[i] + latent[i]);
                let c = std::array::from_fn(|i| latent[i + 3] / self.angular_length);
                if !finite3(relative_t) || !finite3(c) {
                    outcome.null_reason = Some("learned_numerical_null".into());
                    return Ok(outcome);
                }
                let anchor_rotation = rotation(anchor.orientation);
                let displacement = matvec(anchor_rotation, relative_t);
                if !finite3(displacement)
                    || (self.periodic
                        && (0..3).any(|i| {
                            displacement[i] < -0.5 * self.box_lengths[i]
                                || displacement[i] >= 0.5 * self.box_lengths[i]
                        }))
                {
                    outcome.null_reason = Some("learned_outside_unique_image_cube".into());
                    return Ok(outcome);
                }
                let relative_r = matmul(cayley(c), component.anchor_rotation);
                let orientation = quaternion(matmul(anchor_rotation, relative_r));
                let raw_position = std::array::from_fn(|i| anchor.position[i] + displacement[i]);
                if !finite3(raw_position) {
                    outcome.null_reason = Some("learned_numerical_null".into());
                    return Ok(outcome);
                }
                let position = if self.periodic {
                    wrap(raw_position, self.box_lengths)
                } else {
                    raw_position
                };
                Pose {
                    position,
                    orientation,
                }
            }
        };
        let new_log_density = self.log_density(&candidate, anchor)?;
        outcome.new_log_density = Some(new_log_density);
        outcome.log_reverse_forward = Some(old_log_density - new_log_density);
        outcome.candidate = Some(candidate);
        Ok(outcome)
    }
}
