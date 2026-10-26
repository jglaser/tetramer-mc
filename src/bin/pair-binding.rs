//! Pair and site association constants of contact basins under exact hard-sphere morphometry
//! plus the atom square well and an optional like-charge penalty (the morph-mc energy). A job
//! may give a fixed `environment` (poses of several bodies); K is then the site constant of
//! one more body inserted at `pose` next to that cluster (default: one body at the origin): K(eps) = int_W exp(-F_solv(g) + eps n(g)) dg over
//! relative poses g of body B in the frame of body A within a window W around a contact pose
//! g0 (|dt| <= window_t, rotation angle <= window_r), with translation volume times normalized
//! Haar measure (A^3). A Metropolis walk at eps_walk inside W maps the basin; a 6D Gaussian in
//! local coordinates (t, omega), g = (t, R0 exp(omega)), fitted to the walk and widened by
//! `inflate` is the importance density q; K = E_q[exp(-E) / q] with the exact density of q
//! including the Haar factor theta^2 / (2 (1 - cos theta)) of exponential coordinates.
use anyhow::Result;
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::PathBuf};
use tetramer_mc::{
    geometry::{Placed, Shape, SphereTree},
    math::{Pose, Vec3, add, matmul, matvec, norm, quaternion, rotation, sub, transpose},
    morph_mc::{ChargeGroup, ChemistryModel, LikeCharge, MorphMc, SquareWell},
    solvation::{GaussConvention, SolvationModel, SolvationState},
    spherical::Container,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    /// JSON list of {label, pose}: B relative to A.
    #[arg(long)]
    poses: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "0,0.17,0.19,0.21,0.23")]
    epsilons: Vec<f64>,
    #[arg(long, default_value_t = 0.21)]
    eps_walk: f64,
    #[arg(long, default_value_t = 1.0)]
    width: f64,
    #[arg(long, default_value_t = 8000)]
    walk_steps: usize,
    #[arg(long, default_value_t = 10000)]
    samples: usize,
    #[arg(long, default_value_t = 3.0)]
    window_t: f64,
    #[arg(long, default_value_t = 20.0)]
    window_r_degrees: f64,
    #[arg(long, default_value_t = 1.5)]
    inflate: f64,
    #[arg(long, default_value_t = 32)]
    threads: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Instead of basin constants: B22 from this many uniform relative poses (centre in the
    /// interaction ball, Haar orientation); --poses is then ignored.
    #[arg(long)]
    virial: Option<usize>,
    /// Hard-sphere morphometric solvation off (Baseline 1: hard shape + square well only).
    #[arg(long)]
    no_solvation: bool,
    /// Ideal Asakura-Oosawa depletion baseline (Baseline 2: F_solv = z * dV_excl(r_d) with probe
    /// radius --depletion-probe; overrides White Bear II solvation when given).
    #[arg(long)]
    depletion_z: Option<f64>,
    #[arg(long, default_value_t = 1.4)]
    depletion_probe: f64,
    /// Like-charge penalty (as in morph-mc): charged groups, weight and cutoff.
    #[arg(long)]
    charge_groups: Option<PathBuf>,
    #[arg(long, default_value_t = 0.)]
    like_charge_weight: f64,
    #[arg(long, default_value_t = 5.0)]
    like_charge_cutoff: f64,
    /// Optional SE(3)-invariant interface chemistry residual model JSON.
    #[arg(long)]
    chemistry_model: Option<PathBuf>,
}

fn exp_rot(w: Vec3) -> [f64; 4] {
    let th = norm(w);
    if th < 1e-12 {
        return [1., 0., 0., 0.];
    }
    let s = (th / 2.).sin() / th;
    [(th / 2.).cos(), w[0] * s, w[1] * s, w[2] * s]
}

#[allow(dead_code)]
fn log_rot(q: [f64; 4]) -> Vec3 {
    let q = if q[0] < 0. { q.map(|x| -x) } else { q };
    let v = [q[1], q[2], q[3]];
    let sn = norm(v);
    if sn < 1e-14 {
        return [2. * q[1], 2. * q[2], 2. * q[3]];
    }
    let th = 2. * sn.atan2(q[0]);
    [v[0] / sn * th, v[1] / sn * th, v[2] / sn * th]
}

struct Pair {
    solv: SolvationState,
    mc: MorphMc,
    tree: SphereTree,
    env: Vec<Placed>,
    like_weight: f64,
}

impl Pair {
    /// (F_solv + like-charge + chemistry residual, n_pairs) of the moved body at pose g next to
    /// the fixed environment, or None for a hard overlap.
    fn energy(&self, g: Pose) -> Result<Option<(f64, f64)>> {
        let pg = Placed::new(g);
        if self.env.iter().any(|e| self.tree.overlaps(e, &pg)) {
            return Ok(None);
        }
        let moved = self.env.len();
        let f = self.solv.propose(&[(moved, g)])?.delta;
        let like = if self.like_weight != 0. { self.like_weight * self.mc.like_pairs(moved, g) } else { 0. };
        let (wp, chem) = self.mc.well_and_chem_pairs(moved, g);
        Ok(Some((f + like + chem, wp as f64)))
    }
}

fn pose_of(g0: Pose, t: Vec3, w: Vec3) -> Pose {
    Pose {
        position: t,
        orientation: quaternion(matmul(rotation(g0.orientation), rotation(exp_rot(w)))),
    }
}

fn cholesky(a: &[[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let s = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 0. {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

fn make_pair(a: &Args, tree: &SphereTree, env: &[Pose]) -> Result<Pair> {
    let n = tree.shape.atoms.len();
    let centers: Vec<Vec3> = tree.shape.atoms.iter().map(|x| x.center).collect();
    let radii: Vec<f64> = tree.shape.atoms.iter().map(|x| x.radius).collect();
    let far = Pose { position: [1e5, 0., 0.], orientation: [1., 0., 0., 0.] };
    let model = if a.no_solvation {
        SolvationModel {
            probe: 1.4,
            pressure: 0.,
            coefficients: vec![[0.; 3]],
            atom_types: vec![0; n],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        }
    } else if let Some(z) = a.depletion_z {
        SolvationModel {
            probe: a.depletion_probe,
            pressure: z,
            coefficients: vec![[0.; 3]],
            atom_types: vec![0; n],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        }
    } else {
        SolvationModel {
            probe: 1.4,
            pressure: 0.2090,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0; n],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        }
    };
    let mut bodies = env.to_vec();
    bodies.push(far);
    let solv = SolvationState::new(model.clone(), &centers, &radii, &bodies)?;
    let wall = Container::new(1e6, tree)?;
    let groups: Vec<ChargeGroup> = match &a.charge_groups {
        Some(f) => serde_json::from_slice(&fs::read(f)?)?,
        None => Vec::new(),
    };
    let mut chem: ChemistryModel = match &a.chemistry_model {
        Some(f) => serde_json::from_slice(&fs::read(f)?)?,
        None => ChemistryModel::default(),
    };
    if a.no_solvation {
        chem.solv_weights = [0.0; 5];
        chem.solv_cap = 0.0;
        chem.burial_gate = [0.0; 2];
    }
    let mc = MorphMc::new(tree.clone(), wall, SolvationModel { pressure: 0., coefficients: vec![[0.; 3]], ..model },
        SquareWell { epsilon: 1., width: a.width }, bodies.clone(), vec![true; bodies.len()])?
        .with_like_charge(LikeCharge { weight: 1., cutoff: a.like_charge_cutoff, groups })?
        .with_chemistry_model(chem)?;
    Ok(Pair {
        solv,
        mc,
        tree: tree.clone(),
        env: env.iter().map(|&p| Placed::new(p)).collect(),
        like_weight: a.like_charge_weight,
    })
}

/// B22 = -(V_c / 2) <exp(-E) - 1> over uniform relative poses in the interaction ball of
/// radius R_c (no term acts beyond it): E = F_solv - eps n, hard overlap -> -1.
fn run_virial(a: &Args, tree: &SphereTree, samples: usize) -> Result<Value> {
    let ext = tree.shape.atoms.iter().map(|x| norm(x.center)).fold(0., f64::max);
    let rmax = tree.shape.atoms.iter().map(|x| x.radius).fold(0., f64::max);
    let rc = 2. * (ext + rmax + f64::max(1.4, a.width / 2.)) + 0.5;
    let vc = 4. / 3. * PI * rc * rc * rc;
    let per = samples.div_ceil(a.threads);
    // Per thread: overlaps and, per eps, the list of exp(-E) - 1 of non-overlapping samples
    // that interact (others contribute 0).
    let parts: Vec<Result<(usize, Vec<Vec<f64>>)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..a.threads)
            .map(|k| {
                sc.spawn(move || -> Result<(usize, Vec<Vec<f64>>)> {
                    let id = Pose { position: [0.; 3], orientation: [1., 0., 0., 0.] };
                    let pair = make_pair(a, tree, &[id])?;
                    let mut rng = StdRng::seed_from_u64(a.seed * 7919 + k as u64);
                    let mut overlaps = 0;
                    let mut f: Vec<Vec<f64>> = vec![Vec::new(); a.epsilons.len()];
                    for _ in 0..per {
                        let t = loop {
                            let x: Vec3 = std::array::from_fn(|_| (2. * rng.random::<f64>() - 1.) * rc);
                            if norm(x) <= rc {
                                break x;
                            }
                        };
                        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
                        let qn = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                        let g = Pose { position: t, orientation: q.map(|v| v / qn) };
                        match pair.energy(g)? {
                            None => overlaps += 1,
                            Some((fs, np)) => {
                                if fs != 0. || np != 0. {
                                    for (e, out) in a.epsilons.iter().zip(f.iter_mut()) {
                                        out.push((-(fs - e * np)).exp() - 1.);
                                    }
                                }
                            }
                        }
                    }
                    Ok((overlaps, f))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut overlaps = 0;
    let mut f: Vec<Vec<f64>> = vec![Vec::new(); a.epsilons.len()];
    for p in parts {
        let (o, v) = p?;
        overlaps += o;
        for (dst, src) in f.iter_mut().zip(v) {
            dst.extend(src);
        }
    }
    let total = per * a.threads;
    let mut out = Vec::new();
    for (e, v) in a.epsilons.iter().zip(&f) {
        // Mean and standard error of exp(-E) - 1 over all samples (overlap -1, far 0).
        let s: f64 = v.iter().sum::<f64>() - overlaps as f64;
        let s2: f64 = v.iter().map(|x| x * x).sum::<f64>() + overlaps as f64;
        let mean = s / total as f64;
        let var = (s2 / total as f64 - mean * mean).max(0.);
        let mut top = v.clone();
        top.sort_by(|x, y| y.partial_cmp(x).unwrap());
        let top10: f64 = top.iter().take(10).sum();
        out.push(json!({"epsilon": e, "B22_A3": -vc / 2. * mean,
            "B22_se_A3": vc / 2. * (var / total as f64).sqrt(),
            "attractive_part_A3": -vc / 2. * v.iter().filter(|x| **x > 0.).sum::<f64>() / total as f64,
            "top10_share_of_attraction": top10.max(0.) / v.iter().filter(|x| **x > 0.).sum::<f64>().max(1e-300),
            "max_weight": top.first().copied().unwrap_or(0.) + 1.}));
    }
    Ok(json!({"samples": total, "R_c": rc, "V_c": vc, "overlap_fraction": overlaps as f64 / total as f64,
        "excluded_volume_part_A3": vc / 2. * overlaps as f64 / total as f64,
        "interacting_fraction": f[0].len() as f64 / total as f64, "width": a.width, "B22": out}))
}

fn run_contact(a: &Args, tree: &SphereTree, label: &str, env: &[Pose], g0: Pose, seed: u64) -> Result<Value> {
    let pair = make_pair(a, tree, env)?;
    let wr = a.window_r_degrees.to_radians();
    let inside = |t: Vec3, w: Vec3| norm(sub(t, g0.position)) <= a.window_t && norm(w) <= wr;
    let mut rng = StdRng::seed_from_u64(seed);
    // Walk at eps_walk inside the window.
    let Some(mut cur) = pair.energy(g0)? else {
        return Ok(json!({"label": label, "error": "contact pose overlaps"}));
    };
    let (mut t, mut w) = (g0.position, [0.; 3]);
    let mut e_cur = cur.0 - a.eps_walk * cur.1;
    let mut walk: Vec<[f64; 6]> = Vec::new();
    let mut e_min = e_cur;
    let mut acc = 0;
    // Adaptive step: tuned towards 30% acceptance during the first half of the walk only,
    // then frozen (the fitted Gaussian is only an importance density, so the walk need not
    // be stationary; K is unbiased for any q that covers the basin).
    let (mut st, mut sr) = (0.03, 0.3f64.to_radians());
    let mut recent = 0;
    for s in 0..a.walk_steps {
        if s > 0 && s % 100 == 0 && s < a.walk_steps / 2 {
            let f = (recent as f64 / 100. - 0.3).exp();
            st *= f;
            sr *= f;
            recent = 0;
        }
        let tn: Vec3 = std::array::from_fn(|k| {
            let z: f64 = StandardNormal.sample(&mut rng);
            t[k] + st * z
        });
        let wn: Vec3 = std::array::from_fn(|k| {
            let z: f64 = StandardNormal.sample(&mut rng);
            w[k] + sr * z
        });
        if inside(tn, wn) {
            if let Some(en) = pair.energy(pose_of(g0, tn, wn))? {
                let e = en.0 - a.eps_walk * en.1;
                if e <= e_cur || rng.random::<f64>() < (e_cur - e).exp() {
                    t = tn;
                    w = wn;
                    cur = en;
                    e_cur = e;
                    acc += 1;
                    recent += 1;
                    e_min = e_min.min(e);
                }
            }
        }
        if s >= a.walk_steps / 2 {
            walk.push([t[0], t[1], t[2], w[0], w[1], w[2]]);
        }
    }
    let _ = cur;
    let m = walk.len() as f64;
    let mean: [f64; 6] = std::array::from_fn(|i| walk.iter().map(|x| x[i]).sum::<f64>() / m);
    let mut cov = [[0.; 6]; 6];
    for x in &walk {
        for i in 0..6 {
            for j in 0..6 {
                cov[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]) / m;
            }
        }
    }
    for i in 0..6 {
        for j in 0..6 {
            cov[i][j] *= a.inflate * a.inflate;
        }
        cov[i][i] += if i < 3 { 1e-4 } else { 1e-6 };
    }
    let l = cholesky(&cov).expect("covariance not positive definite");
    let logdet: f64 = (0..6).map(|i| l[i][i].ln()).sum();
    // Importance sampling.
    let eps = &a.epsilons;
    let mut logw: Vec<Vec<f64>> = vec![Vec::new(); eps.len()];
    let mut valid = 0;
    for _ in 0..a.samples {
        let z: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        let x: [f64; 6] = std::array::from_fn(|i| mean[i] + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>());
        let (tt, ww) = ([x[0], x[1], x[2]], [x[3], x[4], x[5]]);
        let th = norm(ww);
        let mut lq = -0.5 * z.iter().map(|v| v * v).sum::<f64>() - 3. * (2. * PI).ln() - logdet;
        // Density over normalized Haar: 8 pi^2 theta^2 / (2 (1 - cos theta)).
        lq += (8. * PI * PI).ln() + if th < 1e-8 { 0. } else { (th * th / (2. * (1. - th.cos()))).ln() };
        let en = if th < PI && inside(tt, ww) { pair.energy(pose_of(g0, tt, ww))? } else { None };
        for (k, &e) in eps.iter().enumerate() {
            logw[k].push(match en {
                Some((f, np)) => -(f - e * np) - lq,
                None => f64::NEG_INFINITY,
            });
        }
        valid += usize::from(en.is_some());
    }
    let mut out = Vec::new();
    for (k, &e) in eps.iter().enumerate() {
        let v = &logw[k];
        let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = v.iter().map(|x| (x - mx).exp()).collect();
        let mu = ws.iter().sum::<f64>() / ws.len() as f64;
        let var = ws.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (ws.len() as f64 * (ws.len() as f64 - 1.));
        let ess = ws.iter().sum::<f64>().powi(2) / ws.iter().map(|x| x * x).sum::<f64>();
        out.push(json!({"epsilon": e, "lnK": mx + mu.ln(), "lnK_se": var.sqrt() / mu, "ess": ess}));
    }
    let _ = (add, matvec, transpose);
    Ok(json!({"label": label, "walk_acceptance": acc as f64 / a.walk_steps as f64,
        "final_step_t": st, "final_step_r_deg": sr.to_degrees(),
        "E_min_walk": e_min, "E_start": pair.energy(g0)?.map(|(f, n)| f - a.eps_walk * n),
        "basin_sd_t": (0..3).map(|i| cov[i][i].sqrt() / a.inflate).collect::<Vec<_>>(),
        "basin_sd_r_deg": (3..6).map(|i| (cov[i][i].sqrt() / a.inflate).to_degrees()).collect::<Vec<_>>(),
        "valid_fraction": valid as f64 / a.samples as f64, "K": out}))
}

fn main() -> Result<()> {
    let a = Args::parse();
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(&a.shape)?)?)?;
    if let Some(n) = a.virial {
        println!("{}", serde_json::to_string_pretty(&run_virial(&a, &tree, n)?)?);
        return Ok(());
    }
    let poses: Vec<Value> = serde_json::from_slice(&fs::read(&a.poses)?)?;
    let id = Pose { position: [0.; 3], orientation: [1., 0., 0., 0.] };
    let jobs: Vec<(String, Vec<Pose>, Pose)> = poses
        .iter()
        .map(|p| {
            let env: Vec<Pose> = match p.get("environment") {
                Some(e) => serde_json::from_value(e.clone())?,
                None => vec![id],
            };
            Ok((p["label"].as_str().unwrap_or("").to_string(), env, serde_json::from_value(p["pose"].clone())?))
        })
        .collect::<Result<_>>()?;
    let mut results: Vec<Option<Value>> = vec![None; jobs.len()];
    for chunk in (0..jobs.len()).collect::<Vec<_>>().chunks(a.threads) {
        let outs: Vec<(usize, Result<Value>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = chunk
                .iter()
                .map(|&i| {
                    let (a, tree, job) = (&a, &tree, &jobs[i]);
                    sc.spawn(move || (i, run_contact(a, tree, &job.0, &job.1, job.2, a.seed * 1000 + i as u64)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (i, r) in outs {
            let v = r?;
            eprintln!("{}", v["label"]);
            results[i] = Some(v);
        }
    }
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
