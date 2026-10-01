//! Normalized two-atomic-contact translation proposals at a fixed orientation.
//! The original angular Gaussian marginal is never retried or truncated.
use super::*;
use crate::contact_distances::{AzimuthParameters, ContactFrame, RadiusPolygon};
use rand::distr::Open01;
use std::cell::RefCell;

struct AngularConditional {
    mean: [f64; 6],
    lower: [[f64; 6]; 6], // reordered: raw angle, then raw translation
    angular_normalizer: f64,
    world_covariance: Mat3,
}

impl AngularConditional {
    fn new(component: &ImportanceComponent, chart: &Chart) -> Result<Self> {
        let mean = chart.coordinates(component.mean);
        let product: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| chart.lower[i][k] * component.lower[k][j])
                    .sum()
            })
        });
        let order = [3, 4, 5, 0, 1, 2];
        let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| product[order[i]][k] * product[order[j]][k])
                    .sum()
            })
        });
        let mut lower = [[0.; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let value =
                    covariance[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    ensure!(
                        value.is_finite() && value > 0.,
                        "Invalid angular/translation covariance pivot"
                    );
                    value.sqrt()
                } else {
                    value / lower[j][j]
                };
            }
        }
        let conditional: Mat3 = std::array::from_fn(|i| {
            std::array::from_fn(|j| (3..6).map(|k| lower[i + 3][k] * lower[j + 3][k]).sum())
        });
        let r = rotation(chart.fixed.orientation);
        Ok(Self {
            mean,
            angular_normalizer: -1.5 * (2. * PI).ln()
                - (0..3).map(|i| lower[i][i].ln()).sum::<f64>(),
            lower,
            world_covariance: matmul(matmul(r, conditional), transpose(r)),
        })
    }

    fn conditional(&self, raw: [f64; 6], chart: &Chart) -> (f64, Vec3) {
        let mut z = [0.; 3];
        for i in 0..3 {
            z[i] = (raw[i + 3]
                - self.mean[i + 3]
                - (0..i).map(|j| self.lower[i][j] * z[j]).sum::<f64>())
                / self.lower[i][i];
        }
        let mean = std::array::from_fn(|i| {
            self.mean[i] + (0..3).map(|j| self.lower[i + 3][j] * z[j]).sum::<f64>()
        });
        (
            self.angular_normalizer - 0.5 * dot(z, z),
            chart.fixed.apply(add(chart.anchor_position, mean)),
        )
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomContact {
    neighbor_index: usize,
    moving_atom: usize,
    fixed_atom: usize,
}

struct ContactLabel {
    moving: [Vec3; 2],
    fixed: [Vec3; 2],
    core: [f64; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AzimuthConfig {
    localized_probability: f64,
    radius_floor: f64,
    projection_floor: f64,
    gamma_min: f64,
    gamma_max: f64,
}

pub(super) struct ContactDistanceGuide {
    pub(super) base: ImportanceGuide,
    pub(super) widths: Vec<f64>,
    pub(super) contacts: Vec<usize>,
    beta: f64,
    minimum_distance: f64,
    minimum_area: f64,
    azimuth: AzimuthParameters,
    conditionals: Vec<AngularConditional>,
    labels: Vec<ContactLabel>,
    pub(super) last_draw: RefCell<Value>,
}

fn latent(chart: &Chart, raw: [f64; 6]) -> [f64; 6] {
    let mut u = [0.; 6];
    for i in 0..6 {
        u[i] = (raw[i] - chart.mean[i] - (0..i).map(|j| chart.lower[i][j] * u[j]).sum::<f64>())
            / chart.lower[i][i];
    }
    u
}

fn gaussian_log(component: &ImportanceComponent, u: [f64; 6]) -> f64 {
    let mut z = [0.; 6];
    for i in 0..6 {
        z[i] =
            (u[i] - component.mean[i] - (0..i).map(|j| component.lower[i][j] * z[j]).sum::<f64>())
                / component.lower[i][i];
    }
    component.log_normalizer - 0.5 * z.iter().map(|z| z * z).sum::<f64>()
}

impl ContactDistanceGuide {
    pub(super) fn from_bytes(
        raw: &[u8],
        region_hash: &str,
        chart: &Chart,
        inner: f64,
        cfg: &DockingConfig,
        tree: &SphereTree,
    ) -> Result<Self> {
        ensure!(
            inner == 0.,
            "Contact-distance guide requires a complete latent ball"
        );
        let mut data: Value = serde_json::from_slice(raw)?;
        ensure!(
            data["schema"] == "defensive-contact-distance-guide-v1",
            "Wrong contact-distance schema"
        );
        let widths: Vec<f64> = serde_json::from_value(data["contact_widths_A"].clone())?;
        let contacts: Vec<usize> =
            serde_json::from_value(data["contact_neighbor_indices"].clone())?;
        let pair_labels: Vec<[AtomContact; 2]> =
            serde_json::from_value(data["component_contact_pairs"].clone())?;
        let beta = data["conditional_probability"]
            .as_f64()
            .context("Missing conditional probability")?;
        let minimum_distance = data["minimum_center_distance"]
            .as_f64()
            .context("Missing center-distance floor")?;
        let minimum_area = data["minimum_polygon_area"]
            .as_f64()
            .context("Missing polygon-area floor")?;
        let az: AzimuthConfig = serde_json::from_value(data["azimuth"].clone())?;
        ensure!(
            !widths.is_empty() && widths.iter().all(|w| w.is_finite() && *w > 0.),
            "Invalid contact widths"
        );
        ensure!(
            contacts.len() == 2
                && contacts[0] != contacts[1]
                && contacts.iter().all(|i| *i < cfg.fixed_poses.len()),
            "Invalid scaffold labels"
        );
        ensure!(
            beta.is_finite()
                && (0. ..=1.).contains(&beta)
                && minimum_distance.is_finite()
                && minimum_distance > 0.
                && minimum_area.is_finite()
                && minimum_area > 0.,
            "Invalid conditional controls"
        );
        ensure!(
            [
                az.localized_probability,
                az.radius_floor,
                az.projection_floor,
                az.gamma_min,
                az.gamma_max
            ]
            .iter()
            .all(|x| x.is_finite())
                && (0. ..1.).contains(&az.localized_probability)
                && az.radius_floor > 0.
                && az.projection_floor > 0.
                && az.gamma_min > 0.
                && az.gamma_min <= az.gamma_max
                && az.gamma_max <= PI,
            "Invalid azimuth controls"
        );
        data["schema"] = json!("defensive-latent-shell-guide-v1");
        for name in [
            "contact_widths_A",
            "contact_neighbor_indices",
            "component_contact_pairs",
            "conditional_probability",
            "minimum_center_distance",
            "minimum_polygon_area",
            "azimuth",
        ] {
            data.as_object_mut().unwrap().remove(name);
        }
        let base = ImportanceGuide::from_bytes(&serde_json::to_vec(&data)?, region_hash)?;
        ensure!(
            pair_labels.len() == base.components.len(),
            "One pair label required per component"
        );
        let mut labels = Vec::new();
        for pairs in pair_labels {
            for (j, pair) in pairs.iter().enumerate() {
                ensure!(
                    pair.neighbor_index == contacts[j]
                        && pair.moving_atom < tree.shape.atoms.len()
                        && pair.fixed_atom < tree.shape.atoms.len(),
                    "Invalid atomic contact label"
                );
            }
            labels.push(ContactLabel {
                moving: std::array::from_fn(|j| tree.shape.atoms[pairs[j].moving_atom].center),
                fixed: std::array::from_fn(|j| {
                    cfg.fixed_poses[pairs[j].neighbor_index]
                        .apply(tree.shape.atoms[pairs[j].fixed_atom].center)
                }),
                core: std::array::from_fn(|j| {
                    tree.shape.atoms[pairs[j].moving_atom].radius
                        + tree.shape.atoms[pairs[j].fixed_atom].radius
                }),
            });
        }
        let conditionals = base
            .components
            .iter()
            .map(|c| AngularConditional::new(c, chart))
            .collect::<Result<_>>()?;
        Ok(Self {
            base,
            widths,
            contacts,
            beta,
            minimum_distance,
            minimum_area,
            labels,
            conditionals,
            azimuth: AzimuthParameters {
                localized_probability: az.localized_probability,
                rho_floor: az.radius_floor,
                projection_floor: az.projection_floor,
                gamma_min: az.gamma_min,
                gamma_max: az.gamma_max,
            },
            last_draw: RefCell::new(Value::Null),
        })
    }

    fn frame(&self, component: usize, pose: Pose) -> Result<Option<ContactFrame>> {
        let label = &self.labels[component];
        let r = rotation(pose.orientation);
        let centers: [Vec3; 2] =
            std::array::from_fn(|j| sub(label.fixed[j], matvec(r, label.moving[j])));
        let distance = norm(sub(centers[1], centers[0]));
        ensure!(distance.is_finite(), "Nonfinite contact-center distance");
        if distance <= self.minimum_distance {
            return Ok(None);
        }
        Ok(Some(ContactFrame::new(centers[0], centers[1])?))
    }

    fn polygon(
        &self,
        component: usize,
        width: usize,
        frame: &ContactFrame,
    ) -> Result<RadiusPolygon> {
        let core = self.labels[component].core;
        RadiusPolygon::new(frame.distance, core.map(|r| [r, r + self.widths[width]]))
    }

    pub(super) fn density_details(
        &self,
        u: [f64; 6],
        inside: bool,
        log_volume: f64,
        chart: &Chart,
        _outer: f64,
    ) -> Result<(f64, Value)> {
        let baseline = self.base.log_density(u, inside, log_volume);
        if self.beta == 0. || self.base.alpha == 1. {
            return Ok((baseline, json!({"conditioning_disabled":true})));
        }
        let raw = chart.coordinates(u);
        let (pose, _) = chart.decode(u);
        let mut result = if inside {
            self.base.alpha.ln() - log_volume
        } else {
            f64::NEG_INFINITY
        };
        let mut details = Vec::new();
        let mut fallback_branches = 0;
        for (ci, component) in self.base.components.iter().enumerate() {
            let g = gaussian_log(component, u);
            let normal = &self.conditionals[ci];
            let (angular, world_mean) = normal.conditional(raw, chart);
            let frame = self.frame(ci, pose)?;
            let mut h = f64::NEG_INFINITY;
            let mut width_details = Vec::new();
            for wi in 0..self.widths.len() {
                let mut d = json!({"width_index":wi,"width_A":self.widths[wi]});
                let term = if let Some(frame) = &frame {
                    let polygon = self.polygon(ci, wi, frame)?;
                    d["distance"] = json!(frame.distance);
                    d["polygon_area"] = json!(polygon.area);
                    if polygon.area <= self.minimum_area {
                        d["fallback"] = json!(true);
                        d["fallback_reason"] = json!("polygon_area");
                        fallback_branches += 1;
                        g
                    } else {
                        d["fallback"] = json!(false);
                        let mut translation = f64::NEG_INFINITY;
                        if let Some(coordinates) = frame.encode(pose.position)? {
                            if polygon.contains(coordinates.radii) {
                                let circle = frame.circle(coordinates.radii)?;
                                let law = circle.azimuth_from_gaussian(
                                    world_mean,
                                    normal.world_covariance,
                                    self.azimuth,
                                )?;
                                translation =
                                    frame.cartesian_log_density(pose.position, &polygon, law)?;
                            }
                        }
                        ensure!(
                            !translation.is_nan() && translation != f64::INFINITY,
                            "Invalid contact translation density"
                        );
                        d["translation_log_density"] = if translation.is_finite() {
                            json!(translation)
                        } else {
                            Value::Null
                        };
                        chart.log_det + angular + translation
                    }
                } else {
                    fallback_branches += 1;
                    d["fallback"] = json!(true);
                    d["fallback_reason"] = json!("center_distance");
                    g
                };
                h = log_add(h, term - (self.widths.len() as f64).ln());
                width_details.push(d);
            }
            let combined = log_add((1. - self.beta).ln() + g, self.beta.ln() + h);
            result = log_add(
                result,
                (1. - self.base.alpha).ln() + component.weight.ln() + combined,
            );
            details.push(json!({"component":ci,"angular_log_density":angular,"conditional_world_mean":world_mean,
                "widths":width_details}));
        }
        Ok((
            result,
            json!({"raw_coordinates":raw,"components":details,"baseline_log_density":baseline,
            "component_branches":self.base.components.len()*self.widths.len(),"fallback_component_branches":fallback_branches}),
        ))
    }

    pub(super) fn draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        outer: f64,
        inner: f64,
        fraction: f64,
    ) -> Result<([f64; 6], f64, Option<usize>, Option<bool>)> {
        let (original, radius, component) = self.base.draw(rng, outer, inner, fraction)?;
        *self.last_draw.borrow_mut() = json!({"conditional":false});
        let Some(ci) = component else {
            return Ok((original, radius, None, None));
        };
        if self.beta == 0. || rng.random::<f64>() >= self.beta {
            return Ok((original, radius, Some(ci), None));
        }
        let wi = rng.random_range(0..self.widths.len());
        let mut raw = chart.coordinates(original);
        let (pose, _) = chart.decode(original);
        let frame = self.frame(ci, pose)?;
        let polygon = frame
            .as_ref()
            .map(|f| self.polygon(ci, wi, f))
            .transpose()?;
        let reason = if frame.is_none() {
            Some("center_distance")
        } else if polygon.as_ref().unwrap().area <= self.minimum_area {
            Some("polygon_area")
        } else {
            None
        };
        *self.last_draw.borrow_mut() = json!({"conditional":true,"component":ci,"width_index":wi,
            "width_A":self.widths[wi],"fallback":reason.is_some(),"fallback_reason":reason,"original_latent":original,
            "retained_raw_angles":[raw[3],raw[4],raw[5]],"polygon_area":polygon.as_ref().map(|p|p.area)});
        if reason.is_some() {
            return Ok((original, radius, Some(ci), Some(true)));
        }
        let frame = frame.unwrap();
        let polygon = polygon.unwrap();
        let radius_uniforms: [f64; 3] = std::array::from_fn(|_| rng.sample(Open01));
        let radii = polygon.sample_radii(radius_uniforms)?;
        let circle = frame.circle(radii)?;
        let (angular, world_mean) = self.conditionals[ci].conditional(raw, chart);
        let law = circle.azimuth_from_gaussian(
            world_mean,
            self.conditionals[ci].world_covariance,
            self.azimuth,
        )?;
        let azimuth_uniforms: [f64; 2] = std::array::from_fn(|_| rng.sample(Open01));
        let phi = law.sample(azimuth_uniforms)?;
        let position = circle.point(phi)?;
        let translation = sub(chart.fixed.inverse(position), chart.anchor_position);
        raw[..3].copy_from_slice(&translation);
        let u = latent(chart, raw);
        ensure!(
            u.iter().all(|x| x.is_finite()),
            "Nonfinite distance proposal; never retry"
        );
        {
            let mut trace = self.last_draw.borrow_mut();
            trace["distance"] = json!(frame.distance);
            trace["radii"] = json!(radii);
            trace["polygon_vertices"] = json!(polygon.vertices);
            trace["azimuth_law"] = json!(law);
            trace["circle_radius"] = json!(circle.radius);
            trace["circle_center"] = json!(circle.center);
            trace["phi"] = json!(phi);
            trace["radius_uniforms"] = json!(radius_uniforms);
            trace["azimuth_uniforms"] = json!(azimuth_uniforms);
            trace["angular_log_density"] = json!(angular);
            trace["conditional_world_mean"] = json!(world_mean);
            trace["world_translation"] = json!(position);
            trace["azimuth_log_density"] = json!(law.log_density(phi)?);
        }
        let radius = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok((u, radius, Some(ci), Some(false)))
    }
}

#[cfg(test)]
#[path = "contact_distance_tests.rs"]
mod tests;
