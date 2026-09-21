//! Small FP64 rigid-pose operations. Quaternions are scalar-first.
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];
pub const IDENTITY: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub orientation: [f64; 4],
}

impl Pose {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.position
                .iter()
                .chain(self.orientation.iter())
                .all(|x| x.is_finite()),
            "nonfinite pose"
        );
        let norm: f64 = self.orientation.iter().map(|x| x * x).sum();
        ensure!((norm - 1.).abs() <= 1e-8, "quaternion is not normalized");
        Ok(())
    }
    pub fn apply(&self, x: Vec3) -> Vec3 {
        add(self.position, matvec(rotation(self.orientation), x))
    }
    pub fn inverse(&self, x: Vec3) -> Vec3 {
        matvec(transpose(rotation(self.orientation)), sub(x, self.position))
    }
}

/// Invert a relative rigid transform: I(t, R) = (-R^T t, R^T).
///
/// Unlike `Pose::inverse`, which transforms a point, this returns the inverse
/// pose. It is an involution preserving translation volume times normalized
/// rotational Haar measure, even though its translation depends on rotation.
pub fn invert_relative_pose(pose: Pose) -> Pose {
    let [w, x, y, z] = pose.orientation;
    Pose {
        position: scale(
            matvec(transpose(rotation(pose.orientation)), pose.position),
            -1.,
        ),
        orientation: [w, -x, -y, -z],
    }
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|k| a[k] + b[k])
}
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|k| a[k] - b[k])
}
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    a.map(|x| x * s)
}
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| a[k] * b[k]).sum()
}
pub fn norm(a: Vec3) -> f64 {
    a[0].hypot(a[1]).hypot(a[2])
}
pub fn transpose(a: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
pub fn matvec(a: Mat3, b: Vec3) -> Vec3 {
    a.map(|row| dot(row, b))
}
pub fn matmul(a: Mat3, b: Mat3) -> Mat3 {
    let bt = transpose(b);
    std::array::from_fn(|i| std::array::from_fn(|j| dot(a[i], bt[j])))
}
pub fn wrap(v: Vec3, l: Vec3) -> Vec3 {
    std::array::from_fn(|k| v[k].rem_euclid(l[k]))
}
pub fn minimum_image(v: Vec3, l: Vec3) -> Vec3 {
    std::array::from_fn(|k| v[k] - l[k] * (v[k] / l[k] + 0.5).floor())
}

pub fn rotation(q: [f64; 4]) -> Mat3 {
    // Normalize ingress quaternions before constructing any geometry transform.
    // The accepted input tolerance must not turn a rigid isometry into a shear.
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    let [w, x, y, z] = q.map(|x| x / n);
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
        ],
        [
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
        ],
        [
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
        ],
    ]
}

/// Proper matrix to canonical unit quaternion, stable near half-turns.
pub fn quaternion(r: Mat3) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > 0. {
        let s = 2. * (tr + 1.).sqrt();
        [
            s / 4.,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2. * (1. + r[0][0] - r[1][1] - r[2][2]).sqrt();
        [
            (r[2][1] - r[1][2]) / s,
            s / 4.,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = 2. * (1. + r[1][1] - r[0][0] - r[2][2]).sqrt();
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            s / 4.,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = 2. * (1. + r[2][2] - r[0][0] - r[1][1]).sqrt();
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            s / 4.,
        ]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let sign = if q[0] < 0. { -1. } else { 1. };
    q.map(|v| sign * v / n)
}

pub fn cayley(c: Vec3) -> Mat3 {
    let d = 1_f64.hypot(norm(c));
    rotation([1. / d, c[0] / d, c[1] / d, c[2] / d])
}

pub fn uniform_pose(rng: &mut StdRng, l: Vec3) -> Pose {
    let position = l.map(|length| rng.random::<f64>() * length);
    let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    Pose {
        position,
        orientation: q.map(|x| x / n),
    }
}

pub fn local_pose(rng: &mut StdRng, old: Pose, l: Vec3, dt: f64, dc: f64) -> Pose {
    let noise: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
    let c: Vec3 = std::array::from_fn(|_| {
        let x: f64 = StandardNormal.sample(rng);
        dc * x
    });
    Pose {
        position: wrap(add(old.position, scale(noise, dt)), l),
        orientation: quaternion(matmul(cayley(c), rotation(old.orientation))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_roundtrip() {
        for c in [[0., 0., 0.], [1., 2., -3.], [1e8, 2e8, -1e8]] {
            let r = cayley(c);
            let rr = rotation(quaternion(r));
            for i in 0..3 {
                for j in 0..3 {
                    assert!((r[i][j] - rr[i][j]).abs() < 2e-15);
                }
            }
        }
        assert_eq!(minimum_image([5., -5., 15.], [10.; 3]), [-5.; 3]);
    }
}
