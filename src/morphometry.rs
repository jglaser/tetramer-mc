//! Exact Minkowski measures of a union of balls: volume V, area A, additive
//! integrated mean curvature M = int (k1 + k2)/2 dA, and integrated Gaussian
//! curvature X = int K dA = 4 pi chi, with a canonical split into per-ball parts.
//!
//! On ball i the exposed region E_i is the unit sphere minus the caps
//! {n . u_j >= a_j} cut by its neighbours. Its boundary consists of exact circle
//! arcs, found by intersecting the cap circles and keeping arc pieces outside
//! every other cap. With E_i on the left of its boundary (Stokes on the unit sphere):
//!   |E_i|       = oint (q . (n x dn)) / (1 + n . q),  q = -(axis of the largest cap)
//!   int_E n dOm = 1/2 oint n x dn
//! and per ball
//!   A_i = r_i^2 |E_i|
//!   V_i = (r_i^3 |E_i| + r_i^2 c_i . int_E n dOm) / 3      (divergence theorem)
//!   M_i = r_i |E_i| - 1/4 sum_arcs psi_ij L                 (reentrant edges, split equally)
//!   X_i = |E_i| - sum_arcs cos(beta) dt + sum_corners (theta - pi/3) / 2
//! where cos psi_ij = (r_i^2 + r_j^2 - d^2) / (2 r_i r_j), beta is the cap's angular
//! radius and theta the turning angle of E_i at a corner (the angle between the
//! tangential projections of the two cap axes). Each corner is the endpoint of two
//! arcs and each union vertex has three corners, so the vertex Gaussian curvature
//! theta_1 + theta_2 + theta_3 - pi is local: no global topology is needed and
//! sum_i X_i = 4 pi chi. The arc integrals use Gauss-Legendre quadrature on smooth
//! integrands. The edge term psi/2 is the additive (Hadwiger) mean curvature; the
//! inflation derivative dA/dr = 2 M_steiner instead weights edges by tan(psi/2).
use crate::math::{Vec3, add, dot, norm, scale, sub};
use anyhow::{Result, ensure};
use std::{collections::HashMap, f64::consts::PI, sync::OnceLock};

const GL_POINTS: usize = 24;
const TWO_PI: f64 = 2. * PI;

fn gauss_legendre() -> &'static ([f64; GL_POINTS], [f64; GL_POINTS]) {
    static NODES: OnceLock<([f64; GL_POINTS], [f64; GL_POINTS])> = OnceLock::new();
    NODES.get_or_init(|| {
        let n = GL_POINTS;
        let mut x = [0.; GL_POINTS];
        let mut w = [0.; GL_POINTS];
        for i in 0..n {
            let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            for _ in 0..100 {
                let (mut p0, mut p1) = (1., z);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                let dp = n as f64 * (z * p1 - p0) / (z * z - 1.);
                let dz = p1 / dp;
                z -= dz;
                if dz.abs() < 1e-16 {
                    let (mut q0, mut q1) = (1., z);
                    for k in 2..=n {
                        let q2 = ((2 * k - 1) as f64 * z * q1 - (k - 1) as f64 * q0) / k as f64;
                        q0 = q1;
                        q1 = q2;
                    }
                    let dq = n as f64 * (z * q1 - q0) / (z * z - 1.);
                    x[i] = z;
                    w[i] = 2. / ((1. - z * z) * dq * dq);
                    break;
                }
            }
        }
        (x, w)
    })
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn frame(u: Vec3) -> (Vec3, Vec3) {
    let a = if u[0].abs() < 0.9 {
        [1., 0., 0.]
    } else {
        [0., 1., 0.]
    };
    let e1 = cross(u, a);
    let e1 = scale(e1, 1. / norm(e1));
    (e1, cross(u, e1))
}

/// Per-ball parts of the union measures. Sums over balls give the union values.
/// `volume` depends on the coordinate origin ball by ball; only its sum does not.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BallMeasures {
    pub volume: f64,
    pub area: f64,
    pub mean: f64,
    pub gauss: f64,
    /// Inflation (Steiner) mean curvature, edges weighted by tan(psi/2); diagnostic.
    pub mean_steiner: f64,
    /// Exposed moment int_E n dOmega on the unit sphere. Under a rigid motion that
    /// keeps the ball's neighbourhood, it rotates and only `volume` changes:
    /// volume = (r^3 |E| + r^2 c . moment) / 3.
    pub moment: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct UnionMeasures {
    pub volume: f64,
    pub area: f64,
    pub mean: f64,
    pub gauss: f64,
    pub mean_steiner: f64,
    pub balls: Vec<BallMeasures>,
}

impl UnionMeasures {
    /// Euler characteristic chi = X / (4 pi), rounded; the raw ratio is in `gauss`.
    pub fn euler(&self) -> i64 {
        (self.gauss / (4. * PI)).round() as i64
    }
}

struct Cap {
    j: usize,
    u: Vec3,
    a: f64,
}

/// Exact measures of ball (c, r) within the union, given every ball that overlaps it
/// as (centre, radius). Returns the ball's parts of V, A, M, X and M_steiner.
pub fn ball_measures(c: Vec3, r: f64, neighbours: &[(Vec3, f64)]) -> BallMeasures {
    let mut caps = Vec::with_capacity(neighbours.len());
    for (j, &(cj, rj)) in neighbours.iter().enumerate() {
        let dv = sub(cj, c);
        let d = norm(dv);
        if d == 0. {
            if rj >= r {
                return BallMeasures::default();
            }
            continue;
        }
        let a = (r * r + d * d - rj * rj) / (2. * r * d);
        if a <= -1. {
            return BallMeasures::default(); // ball inside ball j
        }
        if a >= 1. {
            continue; // no intersection
        }
        caps.push(Cap {
            j,
            u: scale(dv, 1. / d),
            a,
        });
    }
    if caps.is_empty() {
        let e = 4. * PI;
        return BallMeasures {
            volume: (r * r * r * e + r * r * 0.) / 3.,
            area: r * r * e,
            mean: r * e,
            gauss: e,
            mean_steiner: r * e,
            moment: [0.; 3],
        };
    }
    let largest = caps
        .iter()
        .min_by(|x, y| x.a.partial_cmp(&y.a).unwrap())
        .unwrap();
    let q = scale(largest.u, -1.);
    let (gx, gw) = gauss_legendre();
    let mut area = 0.;
    let mut moment = [0.; 3];
    let (mut edge_mean, mut edge_steiner, mut geodesic, mut vertex) = (0., 0., 0., 0.);
    let mut any_arc = false;
    for (k, cap) in caps.iter().enumerate() {
        let (u, a) = (cap.u, cap.a);
        let s = (1. - a * a).sqrt();
        let (e1, e2) = frame(u);
        let point = |t: f64| {
            add(
                scale(u, a),
                scale(add(scale(e1, t.cos()), scale(e2, t.sin())), s),
            )
        };
        let mut ts: Vec<(f64, usize)> = Vec::new();
        for (m, other) in caps.iter().enumerate() {
            if m == k {
                continue;
            }
            let (p1, p2) = (s * dot(e1, other.u), s * dot(e2, other.u));
            let rhs = other.a - a * dot(u, other.u);
            let rr = p1.hypot(p2);
            if rr < 1e-15 || rhs.abs() > rr {
                continue;
            }
            let t0 = p2.atan2(p1);
            let dt = (rhs / rr).clamp(-1., 1.).acos();
            ts.push(((t0 + dt).rem_euclid(TWO_PI), m));
            ts.push(((t0 - dt).rem_euclid(TWO_PI), m));
        }
        ts.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let pieces: Vec<(f64, f64, Option<usize>, Option<usize>)> = if ts.is_empty() {
            vec![(0., TWO_PI, None, None)]
        } else {
            (0..ts.len())
                .map(|i| {
                    let hi = if i + 1 < ts.len() {
                        ts[i + 1].0
                    } else {
                        ts[0].0 + TWO_PI
                    };
                    (ts[i].0, hi, Some(ts[i].1), Some(ts[(i + 1) % ts.len()].1))
                })
                .collect()
        };
        let corner = |tc: f64, m: usize| {
            let nc = point(tc);
            let pk = sub(u, scale(nc, dot(u, nc)));
            let pm = sub(caps[m].u, scale(nc, dot(caps[m].u, nc)));
            let th = (dot(pk, pm) / (norm(pk) * norm(pm))).clamp(-1., 1.).acos();
            0.5 * (th - PI / 3.)
        };
        let (cj, rj) = neighbours[cap.j];
        let dd = norm(sub(cj, c));
        let psi = ((r * r + rj * rj - dd * dd) / (2. * r * rj))
            .clamp(-1., 1.)
            .acos();
        for (t_lo, t_hi, m_lo, m_hi) in pieces {
            if t_hi - t_lo < 1e-14 {
                continue;
            }
            let tm = 0.5 * (t_lo + t_hi);
            let nm = point(tm);
            if caps
                .iter()
                .enumerate()
                .any(|(m, other)| m != k && dot(other.u, nm) > other.a)
            {
                continue; // covered by another cap
            }
            any_arc = true;
            let half = 0.5 * (t_hi - t_lo);
            for (x, w) in gx.iter().zip(gw) {
                let t = half * x + tm;
                let n = point(t);
                let dn = scale(add(scale(e1, -t.sin()), scale(e2, t.cos())), s);
                let cr = cross(n, dn);
                // E lies outside cap k, i.e. on the left when t decreases.
                area -= half * w * dot(cr, q) / (1. + dot(n, q));
                moment = sub(moment, scale(cr, 0.5 * half * w));
            }
            let length = r * s * (t_hi - t_lo);
            edge_mean += 0.25 * psi * length;
            edge_steiner += 0.5 * (psi / 2.).tan() * length;
            geodesic += a * (t_hi - t_lo);
            if let (Some(ml), Some(mh)) = (m_lo, m_hi) {
                vertex += corner(t_lo, ml) + corner(t_hi, mh);
            }
        }
    }
    // Stokes fixes the area modulo the excluded pole cap.
    area = area.rem_euclid(4. * PI);
    if area < 1e-12 && !any_arc {
        area = 0.;
    }
    BallMeasures {
        volume: (r * r * r * area + r * r * dot(c, moment)) / 3.,
        area: r * r * area,
        mean: r * area - edge_mean,
        gauss: area - geodesic + vertex,
        mean_steiner: r * area - edge_steiner,
        moment,
    }
}

/// Spatial hash of ball indices with cell size `cell`.
pub struct BallGrid {
    cell: f64,
    cells: HashMap<[i64; 3], Vec<usize>>,
}

impl BallGrid {
    pub fn new(centers: &[Vec3], cell: f64) -> Self {
        let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (i, c) in centers.iter().enumerate() {
            cells.entry(Self::key(c, cell)).or_default().push(i);
        }
        Self { cell, cells }
    }
    fn key(c: &Vec3, cell: f64) -> [i64; 3] {
        c.map(|x| (x / cell).floor() as i64)
    }
    /// Indices of balls whose cells lie within one cell of `c`.
    pub fn near(&self, c: &Vec3) -> impl Iterator<Item = usize> + '_ {
        let k = Self::key(c, self.cell);
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                (-1..=1).flat_map(move |dz| {
                    self.cells
                        .get(&[k[0] + dx, k[1] + dy, k[2] + dz])
                        .into_iter()
                        .flatten()
                        .copied()
                })
            })
        })
    }
}

/// Exact union measures with per-ball parts.
pub fn union_measures(centers: &[Vec3], radii: &[f64]) -> Result<UnionMeasures> {
    ensure!(
        centers.len() == radii.len(),
        "centre and radius counts differ"
    );
    ensure!(
        radii.iter().all(|r| r.is_finite() && *r > 0.)
            && centers.iter().flatten().all(|x| x.is_finite()),
        "invalid ball"
    );
    let mut out = UnionMeasures::default();
    if centers.is_empty() {
        return Ok(out);
    }
    let rmax = radii.iter().copied().fold(0., f64::max);
    let grid = BallGrid::new(centers, 2. * rmax);
    for (i, (&c, &r)) in centers.iter().zip(radii).enumerate() {
        let nb: Vec<(Vec3, f64)> = grid
            .near(&c)
            .filter(|&j| j != i && norm(sub(centers[j], c)) < r + radii[j])
            .map(|j| (centers[j], radii[j]))
            .collect();
        let b = ball_measures(c, r, &nb);
        out.volume += b.volume;
        out.area += b.area;
        out.mean += b.mean;
        out.gauss += b.gauss;
        out.mean_steiner += b.mean_steiner;
        out.balls.push(b);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1. + b.abs())
    }

    #[test]
    fn single_and_two_balls_are_analytic() {
        let r = 1.7;
        let m = union_measures(&[[0.3, -0.2, 1.0]], &[r]).unwrap();
        assert!(close(m.volume, 4. / 3. * PI * r * r * r, 1e-13));
        assert!(close(m.area, 4. * PI * r * r, 1e-13));
        assert!(close(m.mean, 4. * PI * r, 1e-13));
        assert_eq!(m.euler(), 1);
        let (r1, r2, d): (f64, f64, f64) = (1.5, 1.1, 1.9);
        let a1 = (d * d + r1 * r1 - r2 * r2) / (2. * d);
        let (h1, h2) = (r1 - a1, r2 - (d - a1));
        let cap = |h: f64, rr: f64| PI * h * h * (3. * rr - h) / 3.;
        let v = 4. / 3. * PI * (r1.powi(3) + r2.powi(3)) - cap(h1, r1) - cap(h2, r2);
        let area = 4. * PI * (r1 * r1 + r2 * r2) - 2. * PI * (r1 * h1 + r2 * h2);
        let rho = (r1 * r1 - a1 * a1).sqrt();
        let psi = ((r1 * r1 + r2 * r2 - d * d) / (2. * r1 * r2)).acos();
        let mean = (4. * PI * r1 * r1 - 2. * PI * r1 * h1) / r1
            + (4. * PI * r2 * r2 - 2. * PI * r2 * h2) / r2
            - 0.5 * psi * 2. * PI * rho;
        let m = union_measures(&[[0.1, 0.2, -0.3], [0.1 + d, 0.2, -0.3]], &[r1, r2]).unwrap();
        assert!(close(m.volume, v, 1e-11), "{} {}", m.volume, v);
        assert!(close(m.area, area, 1e-11));
        assert!(close(m.mean, mean, 1e-11));
        assert!(close(m.gauss, 4. * PI, 1e-11));
    }

    #[test]
    fn steiner_identities_and_integer_euler_on_random_clusters() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..4 {
            let c: Vec<Vec3> = (0..25)
                .map(|_| std::array::from_fn(|_| (rng.random::<f64>() - 0.5) * 7.))
                .collect();
            let r: Vec<f64> = (0..25).map(|_| 1.0 + 0.6 * rng.random::<f64>()).collect();
            let h = 1e-5;
            let m0 = union_measures(&c, &r).unwrap();
            let mp = union_measures(&c, &r.iter().map(|x| x + h).collect::<Vec<_>>()).unwrap();
            let mm = union_measures(&c, &r.iter().map(|x| x - h).collect::<Vec<_>>()).unwrap();
            assert!(close((mp.volume - mm.volume) / (2. * h), m0.area, 1e-6));
            assert!(close(
                (mp.area - mm.area) / (2. * h),
                2. * m0.mean_steiner,
                1e-5
            ));
            let chi = m0.gauss / (4. * PI);
            assert!((chi - chi.round()).abs() < 1e-6, "non-integer chi {chi}");
        }
    }

    #[test]
    fn ring_and_shell_topology() {
        // Ring of overlapping balls: a solid torus, chi = 0.
        let n = 16;
        let ring: Vec<Vec3> = (0..n)
            .map(|k| {
                let t = TWO_PI * k as f64 / n as f64;
                [6. * t.cos(), 6. * t.sin(), 0.]
            })
            .collect();
        assert_eq!(union_measures(&ring, &vec![1.6; n]).unwrap().euler(), 0);
        // Closed shell of balls around an empty cavity: chi = 2.
        let golden = PI * (3. - 5f64.sqrt());
        let shell: Vec<Vec3> = (0..200)
            .map(|k| {
                let z = 1. - 2. * (k as f64 + 0.5) / 200.;
                let rho = (1. - z * z).sqrt();
                let t = golden * k as f64;
                scale([rho * t.cos(), rho * t.sin(), z], 8.)
            })
            .collect();
        assert_eq!(union_measures(&shell, &vec![1.6; 200]).unwrap().euler(), 2);
    }
}
