//! Exact one-coordinate conditional Gaussian guides for cooperative contacts.
//!
//! Raw chart coordinates are used: changing a whitened latent coordinate would
//! generally couple translation and rotation. The other five raw coordinates
//! retain their complete Gaussian marginal, including empty-line fallbacks.
use super::*;
use crate::line_geometry::{Interval, IntervalSet, translation_intervals};
use crate::native_entry::CompleteNativeEntry;
use rand::distr::Open01;
use serde::Serialize;
use std::cell::RefCell;

#[link(name = "m")]
unsafe extern "C" {
    fn erfc(x: f64) -> f64;
}

/// Positive interval probability, with reflected tails and a close-endpoint
/// quadrature branch. No subtraction of two almost equal central CDFs.
pub(crate) fn normal_mass(lo: f64, hi: f64) -> Result<f64> {
    ensure!(
        lo.is_finite() && hi.is_finite() && lo <= hi,
        "Invalid normal interval"
    );
    if lo == hi {
        return Ok(0.);
    }
    let width = hi - lo;
    let middle = lo + 0.5 * width;
    let mass = if width * (1. + middle.abs()) < 0.1 {
        let nodes = [
            0.1834346424956498,
            0.5255324099163290,
            0.7966664774136267,
            0.9602898564975363,
        ];
        let weights = [
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ];
        let half = 0.5 * width;
        half / (2. * PI).sqrt()
            * nodes
                .into_iter()
                .zip(weights)
                .map(|(n, w)| {
                    let a = middle - half * n;
                    let b = middle + half * n;
                    w * ((-0.5 * a * a).exp() + (-0.5 * b * b).exp())
                })
                .sum::<f64>()
    } else {
        let sf = |z: f64| unsafe { 0.5 * erfc(z / 2_f64.sqrt()) };
        if lo >= 0. {
            sf(lo) - sf(hi)
        } else if hi <= 0. {
            sf(-hi) - sf(-lo)
        } else {
            1. - sf(hi) - sf(-lo)
        }
    };
    ensure!(
        mass.is_finite() && (0. ..=1.).contains(&mass),
        "Invalid conditional Gaussian mass; never silently fall back"
    );
    Ok(mass)
}

struct ConditionalNormal {
    mean: [f64; 6],
    lower: [[f64; 6]; 6],
    order: [usize; 6],
}
impl ConditionalNormal {
    fn new(component: &ImportanceComponent, chart: &Chart, axis: usize) -> Result<Self> {
        let mean = chart.coordinates(component.mean);
        let product: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| chart.lower[i][k] * component.lower[k][j])
                    .sum()
            })
        });
        let indices: Vec<_> = (0..6).filter(|&i| i != axis).chain([axis]).collect();
        let order: [usize; 6] = indices.try_into().unwrap();
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
                let v = covariance[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    ensure!(
                        v.is_finite() && v > 0.,
                        "Unresolved conditional covariance pivot"
                    );
                    v.sqrt()
                } else {
                    v / lower[j][j]
                };
            }
        }
        Ok(Self { mean, lower, order })
    }
    fn conditional(&self, x: [f64; 6]) -> (f64, f64) {
        let mut z = [0.; 5];
        for i in 0..5 {
            z[i] = (x[self.order[i]]
                - self.mean[self.order[i]]
                - (0..i).map(|j| self.lower[i][j] * z[j]).sum::<f64>())
                / self.lower[i][i];
        }
        (
            self.mean[self.order[5]] + (0..5).map(|i| self.lower[5][i] * z[i]).sum::<f64>(),
            self.lower[5][5],
        )
    }
}

/// Each channel is a normalized scalar conditional, never a target restriction.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ClassChannel {
    #[serde(rename = "class")]
    kind: String,
    probability: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    orthant: Option<u8>,
}
struct ClassConditioning {
    native: CompleteNativeEntry,
    channels: Vec<ClassChannel>,
    depletant_radius: f64,
    shape_compatibility: Value,
}
struct ClassGeometry {
    hard_free: IntervalSet,
    channels: Vec<IntervalSet>,
    detail: Value,
}
/// A branch falls back to the hard-free law first, then the original Normal.
/// This choice is component-specific and must be repeated when scoring q.
struct ClassLaw<'a> {
    intervals: Option<&'a IntervalSet>,
    masses: Vec<f64>,
    class_mass: f64,
    hard_free_mass: f64,
    effective_mass: f64,
    target: &'static str,
}
impl ClassLaw<'_> {
    fn multiplier(&self, coordinate: f64) -> f64 {
        match self.intervals {
            None => 1.,
            Some(set) if set.contains(coordinate) => 1. / self.effective_mass,
            Some(_) => 0.,
        }
    }
}

/// Original whitened signs, including all rotation/translation correlations.
/// Zero belongs to the nonnegative bit; a negative-bit boundary is open.
fn orthant_intervals(
    u0: [f64; 6],
    du: [f64; 6],
    orthant: u8,
    range: [f64; 2],
) -> Result<IntervalSet> {
    ensure!(orthant < 64, "Invalid six-dimensional orthant");
    let mut result = IntervalSet::segment(range)?;
    for i in 0..6 {
        ensure!(
            u0[i].is_finite() && du[i].is_finite(),
            "Nonfinite orthant line"
        );
        let positive = orthant & (1 << i) != 0;
        if du[i] == 0. {
            if (u0[i] >= 0.) != positive {
                return Ok(IntervalSet::empty());
            }
            continue;
        }
        let zero = -u0[i] / du[i];
        ensure!(zero.is_finite(), "Unrepresentable orthant crossing");
        let high = positive == (du[i] > 0.);
        let mut iv = Interval::closed(range[0], range[1])?;
        if high {
            if zero > iv.upper || (zero == iv.upper && !positive) {
                return Ok(IntervalSet::empty());
            }
            if zero >= iv.lower {
                iv.lower = zero;
                iv.lower_closed = positive;
            }
        } else {
            if zero < iv.lower || (zero == iv.lower && !positive) {
                return Ok(IntervalSet::empty());
            }
            if zero <= iv.upper {
                iv.upper = zero;
                iv.upper_closed = positive;
            }
        }
        result = result.intersection(&IntervalSet::from_intervals(vec![iv])?);
    }
    Ok(result)
}

pub(super) struct ContactLineGuide {
    pub(super) base: ImportanceGuide,
    hard_free_only: bool,
    classes: Option<ClassConditioning>,
    pose_coordinates: bool,
    axes: Vec<usize>,
    widths: Vec<f64>,
    beta: f64,
    minimum_mass: f64,
    contacts: Vec<usize>,
    conditionals: Vec<Vec<ConditionalNormal>>,
    tree: SphereTree,
    fixed: Vec<Pose>,
    capture_center: Vec3,
    capture_radius: f64,
    pub(super) last_draw: RefCell<Value>,
}

fn raw_to_latent(chart: &Chart, x: [f64; 6]) -> [f64; 6] {
    let mut u = [0.; 6];
    for i in 0..6 {
        u[i] = (x[i] - chart.mean[i] - (0..i).map(|j| chart.lower[i][j] * u[j]).sum::<f64>())
            / chart.lower[i][i];
    }
    u
}

/// Finite line/ball chord; tangent sets have zero one-dimensional measure.
fn ball_chord<const N: usize>(
    origin: [f64; N],
    direction: [f64; N],
    radius: f64,
) -> Option<[f64; 2]> {
    let aa = direction.iter().map(|d| d * d).sum::<f64>();
    let center = -origin
        .iter()
        .zip(direction)
        .map(|(o, d)| o * d)
        .sum::<f64>()
        / aa;
    let minimum = (0..N)
        .map(|i| (origin[i] + center * direction[i]).powi(2))
        .sum::<f64>();
    let residual = radius * radius - minimum;
    if !(aa > 0. && aa.is_finite() && residual > 0.) {
        return None;
    }
    let half = (residual / aa).sqrt();
    let range = [center - half, center + half];
    (range[0].is_finite() && range[1].is_finite() && range[0] < range[1]).then_some(range)
}

/// Class mode must distinguish a certified empty chord from invalid arithmetic.
/// Closed tangencies are retained even though their Normal probability is zero.
fn checked_ball_chord<const N: usize>(
    origin: [f64; N],
    direction: [f64; N],
    radius: f64,
) -> Result<Option<[f64; 2]>> {
    ensure!(
        origin.iter().chain(&direction).all(|x| x.is_finite()) && radius.is_finite() && radius > 0.,
        "Invalid class chord inputs"
    );
    let aa = direction.iter().map(|d| d * d).sum::<f64>();
    let dot = origin
        .iter()
        .zip(direction)
        .map(|(o, d)| o * d)
        .sum::<f64>();
    ensure!(
        aa.is_finite() && aa > 0. && dot.is_finite(),
        "Unrepresentable class chord projection"
    );
    let center = -dot / aa;
    let minimum = (0..N)
        .map(|i| (origin[i] + center * direction[i]).powi(2))
        .sum::<f64>();
    let residual = radius * radius - minimum;
    ensure!(
        center.is_finite() && minimum.is_finite() && residual.is_finite(),
        "Unrepresentable class chord distance"
    );
    if residual < 0. {
        return Ok(None);
    }
    let half = (residual / aa).sqrt();
    let range = [center - half, center + half];
    ensure!(
        half.is_finite()
            && range.iter().all(|x| x.is_finite())
            && (if residual == 0. {
                range[0] <= range[1]
            } else {
                range[0] < range[1]
            }),
        "Unrepresentable class chord endpoints"
    );
    Ok(Some(range))
}

impl ContactLineGuide {
    pub(super) fn class_shape_compatibility(&self) -> Option<&Value> {
        self.classes.as_ref().map(|c| &c.shape_compatibility)
    }

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
            "Contact-line guide currently requires a complete latent ball"
        );
        let mut data: Value = serde_json::from_slice(raw)?;
        let class_mode = data["schema"] == "defensive-native-class-line-guide-v1";
        let pose_coordinates = data["schema"] == "defensive-hard-free-pose-line-guide-v1";
        let hard_free_only =
            pose_coordinates || data["schema"] == "defensive-hard-free-line-guide-v1";
        ensure!(
            class_mode || hard_free_only || data["schema"] == "defensive-contact-line-guide-v1",
            "Wrong contact-line schema"
        );
        let axis_key = if pose_coordinates {
            "raw_pose_axes"
        } else {
            "raw_translation_axes"
        };
        let other_key = if pose_coordinates {
            "raw_translation_axes"
        } else {
            "raw_pose_axes"
        };
        ensure!(
            data.get(other_key).is_none(),
            "Ambiguous coordinate-axis fields"
        );
        let axes: Vec<usize> = serde_json::from_value(data[axis_key].clone())?;
        let (widths, contacts): (Vec<f64>, Vec<usize>) = if hard_free_only || class_mode {
            ensure!(
                data.get("contact_widths_A").is_none()
                    && data.get("contact_neighbor_indices").is_none(),
                "Hard-free guide must not specify contact widths or neighbor labels"
            );
            // One set per axis. Zero is an internal sentinel only: no contact
            // query or additional contact predicate is applied in this mode.
            (vec![0.], Vec::new())
        } else {
            (
                serde_json::from_value(data["contact_widths_A"].clone())?,
                serde_json::from_value(data["contact_neighbor_indices"].clone())?,
            )
        };
        let beta = data["conditional_probability"]
            .as_f64()
            .context("Missing conditional probability")?;
        let minimum_mass = data["minimum_conditional_mass"]
            .as_f64()
            .context("Missing conditional mass floor")?;
        ensure!(
            !axes.is_empty()
                && axes
                    .iter()
                    .all(|&a| a < if pose_coordinates { 6 } else { 3 })
                && axes.iter().enumerate().all(|(i, a)| !axes[..i].contains(a)),
            "Invalid/duplicate raw chart axes"
        );
        ensure!(
            (hard_free_only || class_mode)
                || (!widths.is_empty() && widths.iter().all(|w| w.is_finite() && *w > 0.)),
            "Invalid contact widths"
        );
        ensure!(
            (hard_free_only || class_mode)
                || (!contacts.is_empty()
                    && contacts.iter().all(|&i| i < cfg.fixed_poses.len())
                    && contacts
                        .iter()
                        .enumerate()
                        .all(|(i, a)| !contacts[..i].contains(a))),
            "Invalid contact-neighbor labels"
        );
        ensure!(
            beta.is_finite()
                && (0. ..=1.).contains(&beta)
                && minimum_mass.is_finite()
                && minimum_mass > 0.
                && minimum_mass < 1.,
            "Invalid conditional controls"
        );
        let classes = if class_mode {
            let channels: Vec<ClassChannel> =
                serde_json::from_value(data["class_channels"].clone())?;
            ensure!(
                !channels.is_empty()
                    && channels.iter().all(|c| matches!(
                        c.kind.as_str(),
                        "hard_free" | "native" | "contact_without_native"
                    ) && c.probability.is_finite()
                        && c.probability > 0.
                        && c.orthant.is_none_or(|v| v < 64)),
                "Invalid class channels"
            );
            let mass: f64 = channels.iter().map(|c| c.probability).sum();
            ensure!(
                (mass - 1.).abs() <= 1e-12,
                "Class probabilities must sum to one"
            );
            // Normalize roundoff only, so the implemented sampling and density
            // use precisely the same probabilities.
            let channels = channels
                .into_iter()
                .map(|mut c| {
                    c.probability /= mass;
                    c
                })
                .collect();
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Binding {
                path: PathBuf,
                sha256: String,
            }
            let binding: Binding = serde_json::from_value(data["compiled_native"].clone())?;
            ensure!(
                binding.path.is_absolute(),
                "Compiled native path must be absolute"
            );
            let native = CompleteNativeEntry::load(&binding.path)?;
            ensure!(
                native.compiled_sha256() == binding.sha256,
                "Compiled native hash mismatch"
            );
            let shape_hash = hash_file(&cfg.shape)?;
            ensure!(
                data["shape_sha256"] == shape_hash && native.shape_sha256() == shape_hash,
                "Class guide/native/physical shape mismatch"
            );
            ensure!(
                data["fixed_poses"] == json!(cfg.fixed_poses)
                    && native.fixed_poses() == cfg.fixed_poses,
                "Class guide/native/physical fixed poses mismatch"
            );
            ensure!(
                data["capture_center"] == json!(cfg.capture_center)
                    && data["capture_radius"] == cfg.capture_radius
                    && data["depletant_radius"] == cfg.depletant_radius,
                "Class guide physical capture or depletant radius mismatch"
            );
            ensure!(
                cfg.depletant_radius.is_finite() && cfg.depletant_radius >= 0.,
                "Invalid class depletant radius"
            );
            for key in [
                "class_channels",
                "compiled_native",
                "shape_sha256",
                "fixed_poses",
                "capture_center",
                "capture_radius",
                "depletant_radius",
            ] {
                data.as_object_mut().unwrap().remove(key);
            }
            let shape_compatibility =
                serde_json::to_value(native.validate_shape_compatibility(&tree.shape)?)?;
            Some(ClassConditioning {
                native,
                shape_compatibility,
                channels,
                depletant_radius: cfg.depletant_radius,
            })
        } else {
            None
        };
        data["schema"] = json!("defensive-latent-shell-guide-v1");
        for key in [
            "raw_translation_axes",
            "raw_pose_axes",
            "contact_widths_A",
            "contact_neighbor_indices",
            "conditional_probability",
            "minimum_conditional_mass",
        ] {
            data.as_object_mut().unwrap().remove(key);
        }
        let base = ImportanceGuide::from_bytes(&serde_json::to_vec(&data)?, region_hash)?;
        let conditionals = axes
            .iter()
            .map(|&axis| {
                base.components
                    .iter()
                    .map(|c| ConditionalNormal::new(c, chart, axis))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            base,
            hard_free_only,
            classes,
            pose_coordinates,
            axes,
            widths,
            beta,
            minimum_mass,
            contacts,
            conditionals,
            tree: tree.clone(),
            fixed: cfg.fixed_poses.clone(),
            capture_center: cfg.capture_center,
            capture_radius: cfg.capture_radius,
            last_draw: RefCell::new(Value::Null),
        })
    }

    fn class_geometry(
        &self,
        x: [f64; 6],
        chart: &Chart,
        outer: f64,
        axis: usize,
    ) -> Result<ClassGeometry> {
        let classes = self.classes.as_ref().context("Missing class law")?;
        let mut raw = x;
        raw[axis] = 0.;
        let u0 = raw_to_latent(chart, raw);
        let mut du = [0.; 6];
        for i in 0..6 {
            du[i] = ((if i == axis { 1. } else { 0. })
                - (0..i).map(|j| chart.lower[i][j] * du[j]).sum::<f64>())
                / chart.lower[i][i];
        }
        let (origin, _) = chart.decode(u0);
        let direction = matvec(
            rotation(chart.fixed.orientation),
            std::array::from_fn(|i| if i == axis { 1. } else { 0. }),
        );
        let empty = |reason: &str| ClassGeometry {
            hard_free: IntervalSet::empty(),
            channels: vec![IntervalSet::empty(); classes.channels.len()],
            detail: json!({"axis":axis,"origin":origin,"direction":direction,
                "latent_line_origin":u0,"latent_line_direction":du,"empty_reason":reason,
                "hard_free_intervals":[],"native_intervals":[],"exclusion_contact_intervals":[],
                "channels":classes.channels.iter().enumerate().map(|(i,c)|json!({
                    "channel":i,"class":c.kind,"probability":c.probability,"orthant":c.orthant,"intervals":[]
                })).collect::<Vec<_>>() }),
        };
        let Some(mut segment) = checked_ball_chord(u0, du, outer)? else {
            return Ok(empty("no_R4_chord"));
        };
        let Some(capture) = checked_ball_chord(
            sub(origin.position, self.capture_center),
            direction,
            self.capture_radius,
        )?
        else {
            return Ok(empty("no_capture_chord"));
        };
        segment[0] = segment[0].max(capture[0]);
        segment[1] = segment[1].min(capture[1]);
        if segment[0] > segment[1] {
            return Ok(empty("disjoint_chords"));
        }
        let geometry = translation_intervals(
            &self.tree,
            origin,
            direction,
            &self.tree,
            &self.fixed,
            segment,
            Some(2. * classes.depletant_radius),
            None,
        )?;
        let hard_free = IntervalSet::segment(segment)?.difference(&geometry.hard_overlap);
        // Complete native intervals share the full frozen catalogue and BOTH
        // anchors. No chosen contact/motif can replace this union.
        let native = classes
            .native
            .translation_intervals(origin, direction, segment)?;
        let contacts = geometry
            .contact_overlap
            .as_ref()
            .context("Missing exclusion intervals")?;
        let native_hard = hard_free.intersection(&native.intervals);
        let competing = hard_free
            .intersection(contacts)
            .difference(&native.intervals);
        let mut sets = Vec::new();
        let mut channels = Vec::new();
        for (index, channel) in classes.channels.iter().enumerate() {
            let mut set = match channel.kind.as_str() {
                "hard_free" => hard_free.clone(),
                "native" => native_hard.clone(),
                "contact_without_native" => competing.clone(),
                _ => anyhow::bail!("Invalid class after validation"),
            };
            let orthant = channel
                .orthant
                .map(|o| orthant_intervals(u0, du, o, segment))
                .transpose()?;
            if let Some(orthant) = &orthant {
                set = set.intersection(orthant);
            }
            channels.push(json!({"channel":index,"class":channel.kind,"probability":channel.probability,
                "orthant":channel.orthant,"orthant_intervals":orthant.as_ref().map(IntervalSet::intervals),
                "intervals":set.intervals()}));
            sets.push(set);
        }
        let detail = json!({"axis":axis,"segment":segment,"origin":origin,"direction":direction,
            "latent_line_origin":u0,"latent_line_direction":du,"core_counts":geometry.counts,
            "hard_free_intervals":hard_free.intervals(),"native_intervals":native.intervals.intervals(),
            "native_counts":native.counts,"exclusion_contact_intervals":contacts.intervals(),
            "channels":channels});
        Ok(ClassGeometry {
            hard_free,
            channels: sets,
            detail,
        })
    }

    fn class_law<'a>(
        &self,
        hard: &'a IntervalSet,
        class: &'a IntervalSet,
        mean: f64,
        sigma: f64,
    ) -> Result<ClassLaw<'a>> {
        let (hard_masses, hard_free_mass) = Self::masses(hard, mean, sigma)?;
        let (class_masses, class_mass) = Self::masses(class, mean, sigma)?;
        ensure!(
            class_mass <= hard_free_mass + 2e-12,
            "Class mass exceeds hard-free mass"
        );
        if class_mass > self.minimum_mass {
            Ok(ClassLaw {
                intervals: Some(class),
                masses: class_masses,
                class_mass,
                hard_free_mass,
                effective_mass: class_mass,
                target: "class",
            })
        } else if hard_free_mass > self.minimum_mass {
            Ok(ClassLaw {
                intervals: Some(hard),
                masses: hard_masses,
                class_mass,
                hard_free_mass,
                effective_mass: hard_free_mass,
                target: "hard_free",
            })
        } else {
            Ok(ClassLaw {
                intervals: None,
                masses: Vec::new(),
                class_mass,
                hard_free_mass,
                effective_mass: 1.,
                target: "unconditional",
            })
        }
    }

    fn class_density_trace(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
        _full_trace: bool,
    ) -> Result<(f64, Value)> {
        let base = self.base.log_density(u, inside, volume);
        if self.beta == 0. || self.base.alpha == 1. {
            return Ok((base, json!({"conditioning_disabled":true})));
        }
        let classes = self.classes.as_ref().unwrap();
        let x = chart.coordinates(u);
        let mut factors = vec![0.; self.base.components.len()];
        let mut axes = Vec::new();
        for (ai, &axis) in self.axes.iter().enumerate() {
            let mut geometry = self.class_geometry(x, chart, outer, axis)?;
            let mut components = Vec::new();
            for (ci, conditional) in self.conditionals[ai].iter().enumerate() {
                let (mean, sigma) = conditional.conditional(x);
                let mut channels = Vec::new();
                let mut mixture = 0.;
                for (hi, channel) in classes.channels.iter().enumerate() {
                    let law =
                        self.class_law(&geometry.hard_free, &geometry.channels[hi], mean, sigma)?;
                    let multiplier = law.multiplier(x[axis]);
                    mixture += channel.probability * multiplier;
                    channels.push(json!({"channel":hi,"class_mass":law.class_mass,
                        "hard_free_mass":law.hard_free_mass,"effective_mass":law.effective_mass,
                        "fallback_target":law.target,"query_coordinate_allowed":law.intervals.is_none_or(|v|v.contains(x[axis])),
                        "multiplier":multiplier}));
                }
                factors[ci] += mixture / self.axes.len() as f64;
                components.push(
                    json!({"component":ci,"conditional_mean":mean,"conditional_sigma":sigma,
                    "channels":channels,"channel_mixture_multiplier":mixture}),
                );
            }
            geometry.detail["components"] = json!(components);
            axes.push(geometry.detail);
        }
        let mut result = if inside {
            self.base.alpha.ln() - volume
        } else {
            f64::NEG_INFINITY
        };
        for (ci, c) in self.base.components.iter().enumerate() {
            let correction = (1. - self.beta) + self.beta * factors[ci];
            if correction == 0. {
                continue;
            }
            ensure!(
                correction.is_finite() && correction > 0.,
                "Invalid class mixture multiplier"
            );
            let mut z = [0.; 6];
            for i in 0..6 {
                z[i] = (u[i] - c.mean[i] - (0..i).map(|j| c.lower[i][j] * z[j]).sum::<f64>())
                    / c.lower[i][i];
            }
            let log_g = c.log_normalizer - 0.5 * z.iter().map(|v| v * v).sum::<f64>();
            ensure!(log_g.is_finite(), "Unrepresentable active Gaussian density");
            result = log_add(
                result,
                (1. - self.base.alpha).ln() + c.weight.ln() + log_g + correction.ln(),
            );
        }
        Ok((
            result,
            json!({"raw_coordinates":x,"axes":axes,"component_mixture_multipliers":factors,
            "baseline_log_density":base,"class_scope":"complete native; optional original latent orthants; no old-R5 restriction"}),
        ))
    }

    fn class_draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        outer: f64,
        inner: f64,
        fraction: f64,
    ) -> Result<([f64; 6], f64, Option<usize>, Option<bool>)> {
        let (original, radius, component) = self.base.draw(rng, outer, inner, fraction)?;
        *self.last_draw.borrow_mut() = json!({"conditional":false,"original_latent":original});
        let Some(component) = component else {
            return Ok((original, radius, None, None));
        };
        if self.beta == 0. || rng.random::<f64>() >= self.beta {
            return Ok((original, radius, Some(component), None));
        }
        let classes = self.classes.as_ref().unwrap();
        let ai = rng.random_range(0..self.axes.len());
        let axis = self.axes[ai];
        let choice: f64 = rng.sample(Open01);
        let mut cumulative = 0.;
        let mut hi = classes.channels.len() - 1;
        for (index, channel) in classes.channels.iter().enumerate() {
            cumulative += channel.probability;
            if choice < cumulative {
                hi = index;
                break;
            }
        }
        let mut x = chart.coordinates(original);
        let geometry = self.class_geometry(x, chart, outer, axis)?;
        let (mean, sigma) = self.conditionals[ai][component].conditional(x);
        let law = self.class_law(&geometry.hard_free, &geometry.channels[hi], mean, sigma)?;
        *self.last_draw.borrow_mut() = json!({"conditional":true,"original_latent":original,"axis":axis,
            "channel":hi,"class":classes.channels[hi],"uniform_channel_selection":choice,"component":component,
            "conditional_mean":mean,"conditional_sigma":sigma,"class_mass":law.class_mass,
            "hard_free_mass":law.hard_free_mass,"effective_mass":law.effective_mass,
            "fallback":law.target != "class","fallback_target":law.target,"geometry":geometry.detail});
        let Some(intervals) = law.intervals else {
            return Ok((original, radius, Some(component), Some(true)));
        };
        let draw: f64 = rng.sample(Open01);
        let mut probability = draw * law.effective_mass;
        let mut chosen = None;
        for (interval, &mass) in intervals.intervals().iter().zip(&law.masses) {
            if mass <= 0. {
                continue;
            }
            chosen = Some((interval, mass));
            if probability < mass {
                break;
            }
            probability -= mass;
        }
        let (interval, mass) = chosen.context("Positive class mass without interval")?;
        let lo = (interval.lower - mean) / sigma;
        let within: f64 = rng.sample(Open01);
        let target = within * mass;
        ensure!(
            target > 0. && target < mass,
            "Unresolved class inverse probability"
        );
        let (mut left, mut right) = (interval.lower, interval.upper);
        for _ in 0..80 {
            let mid = left + 0.5 * (right - left);
            if mid == left || mid == right {
                break;
            }
            if normal_mass(lo, (mid - mean) / sigma)? < target {
                left = mid;
            } else {
                right = mid;
            }
        }
        x[axis] = left + 0.5 * (right - left);
        ensure!(
            intervals.contains(x[axis]),
            "Class inverse outside selected intervals"
        );
        {
            let mut trace = self.last_draw.borrow_mut();
            trace["uniform_interval_selection"] = json!(draw);
            trace["selected_interval"] = json!(interval);
            trace["selected_interval_mass"] = json!(mass);
            trace["uniform_within_interval"] = json!(within);
            trace["returned_raw_coordinate"] = json!(x[axis]);
            trace["inverse_probability_error"] =
                json!((normal_mass(lo, (x[axis] - mean) / sigma)? / mass - within).abs());
        }
        let u = raw_to_latent(chart, x);
        ensure!(
            u.iter().all(|v| v.is_finite()),
            "Nonfinite class-conditioned coordinates"
        );
        let radius = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok((u, radius, Some(component), Some(law.target != "class")))
    }

    /// A single geometry set per axis/width, shared by ALL mixture components.
    fn intervals(
        &self,
        x: [f64; 6],
        chart: &Chart,
        outer: f64,
        axis: usize,
    ) -> Result<(Vec<IntervalSet>, Value)> {
        let mut raw = x;
        raw[axis] = 0.;
        let u0 = raw_to_latent(chart, raw);
        let mut du = [0.; 6];
        for i in 0..6 {
            du[i] = ((if i == axis { 1. } else { 0. })
                - (0..i).map(|j| chart.lower[i][j] * du[j]).sum::<f64>())
                / chart.lower[i][i];
        }
        let (origin, _) = chart.decode(u0);
        let direction = matvec(
            rotation(chart.fixed.orientation),
            std::array::from_fn(|i| if i == axis { 1. } else { 0. }),
        );
        let empty = || vec![IntervalSet::default(); self.widths.len()];
        let Some(mut segment) = ball_chord(u0, du, outer) else {
            return Ok((empty(), json!({"axis":axis,"empty_reason":"no_R4_chord"})));
        };
        if axis >= 3 {
            // A raw Cayley coordinate rotates the body at a fixed center.
            // Capture is therefore one Boolean, never a zero-direction chord.
            let center = chart
                .fixed
                .apply(add(chart.anchor_position, [raw[0], raw[1], raw[2]]));
            if norm(sub(center, self.capture_center)) > self.capture_radius {
                return Ok((
                    empty(),
                    json!({"axis":axis,"empty_reason":"no_capture_at_fixed_center"}),
                ));
            }
            let family = crate::cayley_axis_geometry::CayleyAxis {
                world_center: center,
                chart_orientation: chart.fixed.orientation,
                anchor_rotation: chart.anchor_rotation,
                fixed_cayley: [raw[3] / chart.ell, raw[4] / chart.ell, raw[5] / chart.ell],
                axis: axis - 3,
                length_scale: chart.ell,
            };
            let core = crate::cayley_axis_geometry::cayley_axis_intervals(
                &self.tree,
                &family,
                &self.tree,
                &self.fixed,
                segment,
            )?;
            let detail = json!({"axis":axis,"coordinate_kind":"raw-scaled-Cayley",
                "segment":segment,"world_center":center,"fixed_cayley":family.fixed_cayley,
                "length_scale":chart.ell,"core_counts":core.counts,
                "hard_free_intervals":core.hard_free.intervals(),
                "empty_reason":if core.hard_free.length() == 0. { Some("no_positive_hard_free_length") } else { None }});
            return Ok((vec![core.hard_free], detail));
        }
        let Some(capture) = ball_chord(
            sub(origin.position, self.capture_center),
            direction,
            self.capture_radius,
        ) else {
            return Ok((
                empty(),
                json!({"axis":axis,"empty_reason":"no_capture_chord"}),
            ));
        };
        segment[0] = segment[0].max(capture[0]);
        segment[1] = segment[1].min(capture[1]);
        if segment[0] >= segment[1] {
            return Ok((
                empty(),
                json!({"axis":axis,"empty_reason":"disjoint_chords"}),
            ));
        }
        let core = translation_intervals(
            &self.tree,
            origin,
            direction,
            &self.tree,
            &self.fixed,
            segment,
            None,
            None,
        )?;
        let hard_free =
            IntervalSet::from_intervals(vec![Interval::closed(segment[0], segment[1])?])?
                .difference(&core.hard_overlap);
        if hard_free.length() == 0. {
            return Ok((
                empty(),
                json!({"axis":axis,"empty_reason":"no_positive_hard_free_length",
                "segment":segment,"core_counts":core.counts,"origin":origin,"direction":direction}),
            ));
        }
        if self.hard_free_only {
            let detail = json!({"axis":axis,"segment":segment,"core_counts":core.counts,
                "origin":origin,"direction":direction,"hard_free_intervals":hard_free.intervals()});
            return Ok((vec![hard_free], detail));
        }
        let mut sets = Vec::new();
        let mut details = Vec::new();
        for &width in &self.widths {
            let mut joint = hard_free.clone();
            let mut counts = Vec::new();
            for &index in &self.contacts {
                let query = translation_intervals(
                    &self.tree,
                    origin,
                    direction,
                    &self.tree,
                    &[self.fixed[index]],
                    segment,
                    Some(width),
                    None,
                )?;
                joint = joint.intersection(query.contact_overlap.as_ref().unwrap());
                counts.push(json!(query.counts));
            }
            details.push(
                json!({"width_A":width,"intervals":joint.intervals(),"contact_queries":counts}),
            );
            sets.push(joint);
        }
        Ok((
            sets,
            json!({"axis":axis,"segment":segment,"core_counts":core.counts,
            "origin":origin,"direction":direction,"widths":details}),
        ))
    }

    fn masses(intervals: &IntervalSet, mean: f64, sigma: f64) -> Result<(Vec<f64>, f64)> {
        ensure!(
            mean.is_finite() && sigma.is_finite() && sigma > 0.,
            "Invalid conditional Normal parameters"
        );
        let masses = intervals
            .intervals()
            .iter()
            .map(|r| normal_mass((r.lower - mean) / sigma, (r.upper - mean) / sigma))
            .collect::<Result<Vec<_>>>()?;
        let total = masses.iter().sum::<f64>();
        ensure!(
            total.is_finite() && total >= 0. && total <= 1. + 1e-12,
            "Invalid union Gaussian mass"
        );
        Ok((masses, total))
    }

    pub(super) fn density_details(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value)> {
        self.density_trace(u, inside, volume, chart, outer, true)
    }

    /// Production audit trace: retain every axis's exact interval union, but
    /// omit per-component diagnostics and nondeterministic timing fields.
    pub(super) fn density_compact(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value)> {
        self.density_trace(u, inside, volume, chart, outer, false)
    }

    /// World-pose queries can have genuine zero conditional support. Establish
    /// that support from intervals and fallback predicates, independently of
    /// floating-point Gaussian values. Geometry is traversed only once; this
    /// check reuses the saved intervals and repeats only cheap Normal algebra.
    /// The existing regional/diagnostic methods and their arithmetic are intact.
    pub(super) fn density_compact_checked(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value, bool)> {
        ensure!(
            self.hard_free_only,
            "Checked vessel scorer requires hard-free law"
        );
        let (density, detail) = self.density_compact(u, inside, volume, chart, outer)?;
        let mut active = vec![self.beta < 1.; self.base.components.len()];
        if self.base.alpha < 1. && self.beta == 1. {
            let x = chart.coordinates(u);
            let axes = detail["axes"]
                .as_array()
                .context("Missing hard-free axes")?;
            ensure!(axes.len() == self.axes.len(), "Incomplete hard-free axes");
            for (ai, &axis) in self.axes.iter().enumerate() {
                let intervals = IntervalSet::from_intervals(serde_json::from_value(
                    axes[ai]["hard_free_intervals"].clone(),
                )?)?;
                let allowed = intervals.contains(x[axis]);
                for (ci, normal) in self.conditionals[ai].iter().enumerate() {
                    let (mean, sigma) = normal.conditional(x);
                    let (_, mass) = Self::masses(&intervals, mean, sigma)?;
                    active[ci] |= mass <= self.minimum_mass || allowed;
                }
            }
        }
        if self.base.alpha < 1. {
            for (component, &present) in self.base.components.iter().zip(&active) {
                if !present {
                    continue;
                }
                let mut z = [0.; 6];
                for i in 0..6 {
                    z[i] = (u[i]
                        - component.mean[i]
                        - (0..i).map(|j| component.lower[i][j] * z[j]).sum::<f64>())
                        / component.lower[i][i];
                }
                let log_g = component.log_normalizer - 0.5 * z.iter().map(|v| v * v).sum::<f64>();
                ensure!(
                    log_g.is_finite(),
                    "Unrepresentable active Gaussian log density, not a support zero"
                );
            }
        }
        let positive =
            (inside && self.base.alpha > 0.) || (self.base.alpha < 1. && active.iter().any(|v| *v));
        ensure!(
            (positive && density.is_finite()) || (!positive && density == f64::NEG_INFINITY),
            "Conditional density disagrees with analytic support"
        );
        Ok((density, detail, !positive))
    }

    fn density_trace(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
        full_trace: bool,
    ) -> Result<(f64, Value)> {
        if self.classes.is_some() {
            return self.class_density_trace(u, inside, volume, chart, outer, full_trace);
        }
        let base = self.base.log_density(u, inside, volume);
        if self.beta == 0. || self.base.alpha == 1. {
            return Ok((base, json!({"conditioning_disabled":true})));
        }
        let x = chart.coordinates(u);
        let gaussian_logs: Vec<_> = self
            .base
            .components
            .iter()
            .map(|c| {
                let mut z = [0.; 6];
                for k in 0..6 {
                    z[k] = (u[k] - c.mean[k] - (0..k).map(|j| c.lower[k][j] * z[j]).sum::<f64>())
                        / c.lower[k][k];
                }
                c.log_normalizer - 0.5 * z.iter().map(|a| a * a).sum::<f64>()
            })
            .collect();
        let compose = |factors: &[f64], branches: f64| {
            let mut result = if inside {
                self.base.alpha.ln() - volume
            } else {
                f64::NEG_INFINITY
            };
            for (i, c) in self.base.components.iter().enumerate() {
                let correction = (1. - self.beta) + self.beta * factors[i] / branches;
                if correction > 0. {
                    result = log_add(
                        result,
                        (1. - self.base.alpha).ln()
                            + c.weight.ln()
                            + gaussian_logs[i]
                            + correction.ln(),
                    );
                }
            }
            result
        };
        let mut factor = vec![0.; self.base.components.len()];
        let mut geometry = Vec::new();
        let mut fallback = 0;
        for (ai, &axis) in self.axes.iter().enumerate() {
            let geometry_start = full_trace.then(cpu_seconds);
            let (sets, mut detail) = self.intervals(x, chart, outer, axis)?;
            let geometry_cpu = geometry_start.map(|start| cpu_seconds() - start);
            let mut axis_factors = vec![0.; self.base.components.len()];
            let mut components = Vec::new();
            for (ci, normal) in self.conditionals[ai].iter().enumerate() {
                let (mean, sigma) = normal.conditional(x);
                for intervals in &sets {
                    let (_, mass) = Self::masses(intervals, mean, sigma)?;
                    let is_fallback = mass <= self.minimum_mass;
                    let allowed = intervals.contains(x[axis]);
                    let multiplier = if is_fallback {
                        fallback += 1;
                        1.
                    } else if allowed {
                        1. / mass
                    } else {
                        0.
                    };
                    factor[ci] += multiplier;
                    axis_factors[ci] += multiplier;
                    if self.hard_free_only && full_trace {
                        components.push(json!({"component":ci,"gaussian_log_density":gaussian_logs[ci],
                            "conditional_mean":mean,"conditional_sigma":sigma,"conditional_mass":mass,
                            "fallback":is_fallback,"query_coordinate_allowed":allowed}));
                    }
                }
            }
            if self.hard_free_only {
                detail["hard_free_intervals"] = json!(sets[0].intervals());
                detail["axis_log_proposal_density"] = json!(compose(&axis_factors, 1.));
                if full_trace {
                    detail["components"] = json!(components);
                    detail["geometry_cpu_seconds"] = json!(geometry_cpu.unwrap());
                }
            }
            geometry.push(detail);
        }
        let branches = (self.axes.len() * self.widths.len()) as f64;
        let result = compose(&factor, branches);
        Ok((
            result,
            json!({"raw_coordinates":x,"axes":geometry,
            "fallback_component_branches":fallback,"component_branches":branches as usize*self.base.components.len(),
            "baseline_log_density":base}),
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
        if self.classes.is_some() {
            return self.class_draw(rng, chart, outer, inner, fraction);
        }
        let (original, radius, component) = self.base.draw(rng, outer, inner, fraction)?;
        *self.last_draw.borrow_mut() = json!({"conditional":false});
        if self.hard_free_only {
            self.last_draw.borrow_mut()["original_latent"] = json!(original);
        }
        let Some(component) = component else {
            return Ok((original, radius, None, None));
        };
        if self.beta == 0. || rng.random::<f64>() >= self.beta {
            return Ok((original, radius, Some(component), None));
        }
        let ai = rng.random_range(0..self.axes.len());
        let axis = self.axes[ai];
        let wi = rng.random_range(0..self.widths.len());
        let mut x = chart.coordinates(original);
        let (sets, mut geometry) = self.intervals(x, chart, outer, axis)?;
        let intervals = &sets[wi];
        let (mean, sigma) = self.conditionals[ai][component].conditional(x);
        let (masses, total) = Self::masses(intervals, mean, sigma)?;
        let fallback = total <= self.minimum_mass;
        *self.last_draw.borrow_mut() = json!({"conditional":true,"axis":axis,"width_index":wi,
            "width_A":self.widths[wi],"component":component,"conditional_mean":mean,"conditional_sigma":sigma,
            "conditional_mass":total,"fallback":fallback,"geometry":geometry});
        if self.hard_free_only {
            geometry["hard_free_intervals"] = json!(intervals.intervals());
            let mut trace = self.last_draw.borrow_mut();
            trace.as_object_mut().unwrap().remove("width_index");
            trace.as_object_mut().unwrap().remove("width_A");
            trace["original_latent"] = json!(original);
            trace["geometry"] = geometry;
        }
        if fallback {
            return Ok((original, radius, Some(component), Some(true)));
        }
        let draw: f64 = rng.sample(Open01);
        let mut probability = draw * total;
        let mut chosen = None;
        for (interval, mass) in intervals.intervals().iter().zip(masses) {
            if mass <= 0. {
                continue;
            }
            chosen = Some((interval, mass));
            if probability < mass {
                break;
            }
            probability -= mass;
        }
        let (interval, mass) = chosen.context("Positive mass without conditional interval")?;
        let lo = (interval.lower - mean) / sigma;
        let within: f64 = rng.sample(Open01);
        let target = within * mass;
        ensure!(
            target > 0. && target < mass,
            "Unresolved interior conditional probability"
        );
        // Bisect directly in raw translation to avoid rounding a rescaled
        // standardized root outside the geometric interval. Never clip.
        let (mut left, mut right) = (interval.lower, interval.upper);
        for _ in 0..80 {
            let mid = left + 0.5 * (right - left);
            if mid == left || mid == right {
                break;
            }
            if normal_mass(lo, (mid - mean) / sigma)? < target {
                left = mid;
            } else {
                right = mid;
            }
        }
        x[axis] = left + 0.5 * (right - left);
        ensure!(
            intervals.contains(x[axis]),
            "Conditional inverse landed outside the retained interval"
        );
        {
            let mut trace = self.last_draw.borrow_mut();
            if self.hard_free_only {
                trace["uniform_interval_selection"] = json!(draw);
            }
            trace["selected_interval"] = json!(interval);
            trace["selected_interval_mass"] = json!(mass);
            trace["uniform_within_interval"] = json!(within);
            trace["returned_raw_coordinate"] = json!(x[axis]);
            trace["inverse_probability_error"] =
                json!((normal_mass(lo, (x[axis] - mean) / sigma)? / mass - within).abs());
        }
        let u = raw_to_latent(chart, x);
        ensure!(
            u.iter().all(|v| v.is_finite()),
            "Nonfinite conditioned coordinates"
        );
        let radius = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok((u, radius, Some(component), Some(false)))
    }
}

/// Diagnostic only: no Poisson clouds or physical weights. Class guides also
/// retain the independent pointwise native classification on hard-valid poses.
pub struct AuditOptions {
    pub config: PathBuf,
    pub region: PathBuf,
    pub importance_guide: PathBuf,
    pub out: PathBuf,
    pub samples: u64,
    pub seed: u64,
    pub probes: Option<PathBuf>,
}

fn audited_query<T>(
    out: &Path,
    journal: &mut File,
    ordinal: u64,
    kind: &str,
    id: &Value,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    writeln!(
        journal,
        "{}",
        json!({"ordinal":ordinal,"kind":kind,"id":id,"state":"begin"})
    )?;
    journal.flush()?;
    match work() {
        Ok(value) => Ok(value),
        Err(error) => {
            save(
                &out.join("failure.json"),
                &json!({"complete":false,"ordinal":ordinal,
                "kind":kind,"id":id,"error":format!("{error:#}")}),
            )?;
            Err(error)
        }
    }
}

/// Share the diagnostic bookkeeping without changing either sampling law.
enum DiagnosticGuide {
    Line(ContactLineGuide),
    Distances(super::contact_distance::ContactDistanceGuide),
}
impl DiagnosticGuide {
    fn from_bytes(
        raw: &[u8],
        hash: &str,
        chart: &Chart,
        inner: f64,
        cfg: &DockingConfig,
        tree: &SphereTree,
    ) -> Result<Self> {
        let value: Value = serde_json::from_slice(raw)?;
        match value["schema"].as_str() {
            Some(
                "defensive-contact-line-guide-v1"
                | "defensive-hard-free-line-guide-v1"
                | "defensive-hard-free-pose-line-guide-v1"
                | "defensive-native-class-line-guide-v1",
            ) => Ok(Self::Line(ContactLineGuide::from_bytes(
                raw, hash, chart, inner, cfg, tree,
            )?)),
            Some("defensive-contact-distance-guide-v1") => Ok(Self::Distances(
                super::contact_distance::ContactDistanceGuide::from_bytes(
                    raw, hash, chart, inner, cfg, tree,
                )?,
            )),
            _ => anyhow::bail!("Unsupported contact audit guide"),
        }
    }
    fn base(&self) -> &ImportanceGuide {
        match self {
            Self::Line(g) => &g.base,
            Self::Distances(g) => &g.base,
        }
    }
    fn widths(&self) -> &[f64] {
        match self {
            Self::Line(g) => &g.widths,
            Self::Distances(g) => &g.widths,
        }
    }
    fn contacts(&self) -> &[usize] {
        match self {
            Self::Line(g) => &g.contacts,
            Self::Distances(g) => &g.contacts,
        }
    }
    fn last_draw(&self) -> &RefCell<Value> {
        match self {
            Self::Line(g) => &g.last_draw,
            Self::Distances(g) => &g.last_draw,
        }
    }
    fn schema(&self) -> &'static str {
        match self {
            Self::Line(g) if g.classes.is_some() => "native-class-line-guide-audit-v1",
            Self::Line(g) if g.pose_coordinates => "hard-free-pose-line-guide-audit-v1",
            Self::Line(g) if g.hard_free_only => "hard-free-line-guide-audit-v1",
            Self::Line(_) => "contact-line-guide-audit-v1",
            Self::Distances(_) => "contact-distance-guide-audit-v1",
        }
    }
    fn role(&self) -> &'static str {
        match self {
            Self::Line(g) if g.classes.is_some() => "native-class-line-proposal-audit",
            Self::Line(g) if g.pose_coordinates => "hard-free-pose-line-proposal-audit",
            Self::Line(g) if g.hard_free_only => "hard-free-line-proposal-audit",
            Self::Line(_) => "contact-line-proposal-audit",
            Self::Distances(_) => "contact-distance-proposal-audit",
        }
    }
    fn draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        outer: f64,
        inner: f64,
        fraction: f64,
    ) -> Result<([f64; 6], f64, Option<usize>, Option<bool>)> {
        match self {
            Self::Line(g) => g.draw(rng, chart, outer, inner, fraction),
            Self::Distances(g) => g.draw(rng, chart, outer, inner, fraction),
        }
    }
    fn density_details(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value)> {
        match self {
            Self::Line(g) => g.density_details(u, inside, volume, chart, outer),
            Self::Distances(g) => g.density_details(u, inside, volume, chart, outer),
        }
    }
}

pub fn run_audit(options: AuditOptions) -> Result<Value> {
    ensure!(!options.out.exists(), "Audit output must be new");
    let config_raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&config_raw)?;
    cfg.validate()?;
    ensure!(
        cfg.target_region.is_none(),
        "Unsupported docking-only target restriction"
    );
    if cfg.shape.is_relative() {
        cfg.shape = options.config.parent().unwrap().join(&cfg.shape);
    }
    let region_raw = fs::read(&options.region)?;
    let region: Value = serde_json::from_slice(&region_raw)?;
    let guide_raw = fs::read(&options.importance_guide)?;
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_hash = hash_bytes(&shape_raw);
    let fixed: Pose = serde_json::from_value(region["fixed_neighbor"].clone())?;
    let physical: Vec<Pose> = if let Some(v) = region.get("physical_fixed_neighbors") {
        serde_json::from_value(v.clone())?
    } else {
        vec![fixed]
    };
    ensure!(
        region["shape_sha256"] == shape_hash
            && cfg.fixed_poses == physical
            && physical.contains(&fixed),
        "Audit shape or physical scaffold mismatch"
    );
    ensure!(
        region["capture_center"] == json!(cfg.capture_center)
            && region["capture_radius"] == cfg.capture_radius
            && region["activity"] == cfg.reservoir_density
            && region["depletant_radius"] == cfg.depletant_radius
            && region["physical_metric"] == cfg.metadata,
        "Audit physical definition mismatch"
    );
    let radius = region["mahalanobis_radius"]
        .as_f64()
        .context("Missing region radius")?;
    let inner = region
        .get("minimum_mahalanobis_radius")
        .and_then(Value::as_f64)
        .unwrap_or(0.);
    ensure!(
        radius.is_finite() && radius > 0. && inner == 0.,
        "Invalid audit region"
    );
    let chart = Chart::new(&region, &shape_hash, cfg.capture_radius, fixed)?;
    let tree = SphereTree::new(serde_json::from_slice(&shape_raw)?)?;
    let guide = DiagnosticGuide::from_bytes(
        &guide_raw,
        &hash_bytes(&region_raw),
        &chart,
        inner,
        &cfg,
        &tree,
    )?;
    let native = match &guide {
        DiagnosticGuide::Line(g) => g.classes.as_ref().map(|c| &c.native),
        _ => None,
    };
    let native_raw = if let Some(native) = native {
        let guide_definition: Value = serde_json::from_slice(&guide_raw)?;
        let path = guide_definition["compiled_native"]["path"]
            .as_str()
            .context("Missing native definition path")?;
        let bytes = fs::read(path)?;
        ensure!(
            hash_bytes(&bytes) == native.compiled_sha256(),
            "Native definition changed after guide construction"
        );
        Some(bytes)
    } else {
        None
    };
    let exclusion_tree = if native.is_some() {
        let mut shape = tree.shape.clone();
        for atom in &mut shape.atoms {
            atom.radius += cfg.depletant_radius;
        }
        Some(SphereTree::new(shape)?)
    } else {
        None
    };
    let log_volume = 3. * PI.ln() + 6. * radius.ln() - 6_f64.ln();
    let placed: Vec<_> = physical.iter().copied().map(Placed::new).collect();
    let inflated = guide
        .widths()
        .iter()
        .map(|&width| {
            let mut shape = tree.shape.clone();
            for atom in &mut shape.atoms {
                atom.radius += width / 2.;
            }
            SphereTree::new(shape)
        })
        .collect::<Result<Vec<_>>>()?;
    fs::create_dir_all(options.out.join("provenance"))?;
    for (name, bytes) in [
        ("config.json", config_raw.as_slice()),
        ("region.json", region_raw.as_slice()),
        ("importance-guide.json", guide_raw.as_slice()),
        ("shape.json", shape_raw.as_slice()),
    ] {
        fs::write(options.out.join("provenance").join(name), bytes)?;
    }
    let bundle = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), bundle)?;
    if let Some(bytes) = &native_raw {
        fs::write(options.out.join("provenance/compiled-native.json"), bytes)?;
    }
    let probe_raw = options.probes.as_ref().map(fs::read).transpose()?;
    if let Some(bytes) = &probe_raw {
        fs::write(options.out.join("provenance/probes.jsonl"), bytes)?;
    }
    let mut probe_items = Vec::<Value>::new();
    let mut probe_ids = std::collections::BTreeSet::new();
    if let Some(bytes) = &probe_raw {
        for line in std::str::from_utf8(bytes)?.lines() {
            let item: Value = serde_json::from_str(line)?;
            ensure!(
                !item["id"].is_null() && probe_ids.insert(item["id"].to_string()),
                "Missing or duplicated saved query ID"
            );
            probe_items.push(item);
        }
    }
    let mut manifest = json!({"schema":guide.schema(),"samples":options.samples,"seed":options.seed,
        "scope":"Proposal and saved-pose geometry/density only; no Poisson sampling or physical mass",
        "config_sha256":hash_bytes(&config_raw),"region_sha256":hash_bytes(&region_raw),
        "guide_sha256":hash_bytes(&guide_raw),"shape_sha256":shape_hash,
        "probes_sha256":probe_raw.as_ref().map(|b|hash_bytes(b)),
        "executable_sha256":hash_file(&std::env::current_exe()?)?,"source_bundle_sha256":hash_bytes(bundle.as_bytes()),
        "log_latent_ball_volume":log_volume,"latent_radius":radius,"physical_jobs":0});
    if let Some(native) = native {
        manifest["compiled_native_sha256"] = json!(native.compiled_sha256());
        manifest["native_definition_sha256"] = json!(native.definition_sha256());
        manifest["classification_scope"] = json!(
            "Pointwise complete native classifier and physical exclusion contacts only on hard-valid in-domain poses; all invalid/exterior attempts retained with null labels"
        );
    }
    if let DiagnosticGuide::Line(g) = &guide {
        if let Some(report) = g.class_shape_compatibility() {
            manifest["native_shape_compatibility"] = report.clone();
        }
    }
    save(&options.out.join("manifest.json"), &manifest)?;
    let mut writer = BufWriter::new(File::create(options.out.join("samples.jsonl"))?);
    let mut journal = File::create(options.out.join("attempts.jsonl"))?;
    let started = cpu_seconds();
    let mut draw_cpu = 0.;
    let mut density_cpu = 0.;
    let mut nonzero = 0;
    let mut conditional = 0;
    let mut fallback = 0;
    let mut maximum_backmap = 0_f64;
    let evaluate = |u: [f64; 6],
                    kind: &str,
                    id: Value,
                    draw: Value,
                    draw_elapsed: f64|
     -> Result<(Value, f64)> {
        let radial = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inside = radial <= radius;
        let tick = cpu_seconds();
        let (log_density, details) =
            guide.density_details(u, inside, log_volume, &chart, radius)?;
        let elapsed = cpu_seconds() - tick;
        ensure!(
            !log_density.is_nan() && log_density != f64::INFINITY,
            "Invalid scored proposal density"
        );
        let (pose, jacobian) = chart.decode(u);
        pose.validate()?;
        let backmap = chart.encode(pose)?;
        let error = u
            .iter()
            .zip(backmap)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        ensure!(
            error < 2e-7 * (1. + radial.max(radius)),
            "Audit chart inverse failed"
        );
        let body = Placed::new(pose);
        let hard_valid = !placed.iter().any(|p| tree.overlaps(&body, p));
        let contacts: Vec<_> = inflated
            .iter()
            .map(|shape| {
                guide
                    .contacts()
                    .iter()
                    .map(|&i| shape.overlaps(&body, &placed[i]))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut row = json!({"kind":kind,"id":id,"latent":u,"latent_radius":radial,"raw_coordinates":chart.coordinates(u),
            "pose":pose,"shell_valid":inside,"capture_valid":cfg.contains(pose),"hard_valid":hard_valid,
            "log_proposal_density":log_density,"baseline_log_density":guide.base().log_density(u,inside,log_volume),
            "log_physical_jacobian":jacobian,"backmap_error":error,"width_contacts":contacts,
            "draw":draw,"density_details":details,"draw_cpu_seconds":draw_elapsed,"density_cpu_seconds":elapsed});
        if let Some(native) = native {
            let tick = cpu_seconds();
            row["native_decision"] = Value::Null;
            row["exclusion_contact_by_anchor"] = Value::Null;
            if hard_valid && inside && cfg.contains(pose) {
                row["native_decision"] = serde_json::to_value(native.classify(pose)?)?;
                row["exclusion_contact_by_anchor"] = json!(
                    placed
                        .iter()
                        .map(|p| exclusion_tree.as_ref().unwrap().overlaps(&body, p))
                        .collect::<Vec<_>>()
                );
            }
            row["observer_cpu_seconds"] = json!(cpu_seconds() - tick);
        }
        Ok((row, elapsed))
    };
    for index in 0..options.samples {
        let (row, cpu, elapsed) = audited_query(
            &options.out,
            &mut journal,
            index,
            "fresh",
            &json!(index),
            || {
                let mut rng = stream(options.seed, index, 0, guide.role());
                let tick = cpu_seconds();
                let (u, _, component, _) = guide.draw(&mut rng, &chart, radius, 0., 1.)?;
                let elapsed = cpu_seconds() - tick;
                let mut info = guide.last_draw().borrow().clone();
                info["component"] = json!(component);
                let (row, cpu) = evaluate(u, "fresh", json!(index), info, elapsed)?;
                ensure!(
                    row["log_proposal_density"]
                        .as_f64()
                        .is_some_and(f64::is_finite),
                    "Generated draw has zero or invalid proposal density"
                );
                Ok((row, cpu, elapsed))
            },
        )?;
        draw_cpu += elapsed;
        conditional += u64::from(row["draw"]["conditional"] == true);
        fallback += u64::from(row["draw"]["fallback"] == true);
        density_cpu += cpu;
        nonzero += u64::from(
            row["hard_valid"] == true && row["shell_valid"] == true && row["capture_valid"] == true,
        );
        maximum_backmap = maximum_backmap.max(row["backmap_error"].as_f64().unwrap());
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    writer.flush()?;
    let fresh_cpu = cpu_seconds() - started;
    let mut probe_writer = BufWriter::new(File::create(options.out.join("probes.jsonl"))?);
    let mut probes = 0;
    let probe_start = cpu_seconds();
    for (index, item) in probe_items.iter().enumerate() {
        let ordinal = options
            .samples
            .checked_add(index as u64)
            .context("Audit ordinal overflow")?;
        let (row, _) = audited_query(
            &options.out,
            &mut journal,
            ordinal,
            "probe",
            &item["id"],
            || {
                let u: [f64; 6] = serde_json::from_value(item["latent"].clone())?;
                ensure!(u.iter().all(|v| v.is_finite()), "Nonfinite saved query");
                evaluate(u, "probe", item["id"].clone(), Value::Null, 0.)
            },
        )?;
        serde_json::to_writer(&mut probe_writer, &row)?;
        probe_writer.write_all(b"\n")?;
        probe_writer.flush()?;
        probes += 1;
    }
    probe_writer.flush()?;
    let summary = json!({"complete":true,"samples":options.samples,"probes":probes,"hard_capture_shell_valid":nonzero,
        "conditioned_draws":conditional,"fallback_draws":fallback,"maximum_backmap_error":maximum_backmap,
        "draw_cpu_seconds":draw_cpu,"density_cpu_seconds":density_cpu,"fresh_total_cpu_seconds":fresh_cpu,
        "probe_total_cpu_seconds":cpu_seconds()-probe_start,"manifest":manifest,
        "samples_sha256":hash_file(&options.out.join("samples.jsonl"))?,"probes_sha256":hash_file(&options.out.join("probes.jsonl"))?,
        "attempts_sha256":hash_file(&options.out.join("attempts.jsonl"))?});
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
#[path = "contact_line_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "contact_line_class_tests.rs"]
mod class_tests;
