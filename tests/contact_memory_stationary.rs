//! Independent stationarity oracle for the actual contact-memory pair kernel.
//!
//! Every trial starts from iid exact AO pair states; no burn-in or trajectory
//! independence assumption is used. For a fixed partner at the origin, the
//! radial measure is r² dr (not the two-mobile-particle spherical-wall lens).
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;
use tetramer_mc::{
    contact_memory::{MemoryConfig, MemoryState},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, rotation},
};

fn overlap(r: f64) -> f64 {
    // a=1, rd=.5; exclusion radius A=1.5.
    if r >= 3. {
        0.
    } else {
        PI * (6. + r) * (3. - r).powi(2) / 12.
    }
}
fn integrate(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let n = 20000;
    let h = (hi - lo) / n as f64;
    let mut s = f(lo) + f(hi);
    for i in 1..n {
        let coefficient = if i % 2 == 0 { 2. } else { 4. };
        s += coefficient * f(lo + i as f64 * h);
    }
    s * h / 3.
}
fn reference(z: f64) -> [f64; 8] {
    let w = |r: f64| r * r * (z * overlap(r)).exp();
    let partition = integrate(w, 2., 3.) + integrate(w, 3., 4.);
    let moment = |power: i32| {
        (integrate(|r| w(r) * r.powi(power), 2., 3.) + integrate(|r| w(r) * r.powi(power), 3., 4.))
            / partition
    };
    [
        moment(1),
        moment(2),
        integrate(w, 2., 3.) / partition,
        0.,
        0.,
        0.,
        0.,
        0.,
    ]
}
fn draw(rng: &mut StdRng, z: f64) -> Pose {
    let r = loop {
        let r = (8. + 56. * rng.random::<f64>()).cbrt();
        if rng.random::<f64>() <= (z * (overlap(r) - overlap(2.))).exp() {
            break r;
        }
    };
    let d: [f64; 3] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let length = norm(d);
    let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let qnorm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    Pose {
        position: d.map(|x| x * r / length),
        orientation: q.map(|x| x / qnorm),
    }
}
fn features(pose: Pose) -> [f64; 8] {
    let r = norm(pose.position);
    let matrix = rotation(pose.orientation);
    [
        r,
        r * r,
        if r < 3. { 1. } else { 0. },
        pose.position[0],
        pose.position[1],
        pose.position[2],
        matrix[0][0] + matrix[1][1] + matrix[2][2],
        pose.position[2] / r * matrix[0][0],
    ]
}
#[derive(Default)]
struct Moments {
    n: usize,
    sum: [f64; 8],
    square: [f64; 8],
}
impl Moments {
    fn add(&mut self, x: [f64; 8]) {
        self.n += 1;
        for i in 0..8 {
            self.sum[i] += x[i];
            self.square[i] += x[i] * x[i];
        }
    }
    fn check(&self, expected: [f64; 8], name: &str) {
        for i in 0..8 {
            let mean = self.sum[i] / self.n as f64;
            let variance = ((self.square[i] - self.sum[i] * self.sum[i] / self.n as f64)
                / (self.n - 1) as f64)
                .max(0.);
            let se = (variance / self.n as f64).sqrt();
            assert!(
                (mean - expected[i]).abs() < 6. * se + 2e-5,
                "{name} observable{i}: mean{mean}, target{}, SE{se}",
                expected[i]
            );
        }
    }
}

#[test]
fn exact_anchored_ao_equilibrium_survives_memory_local_global_and_mixture() {
    let tree = SphereTree::new(Shape {
        name: "memory oracle sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap();
    assert!(
        (reference(1.5)[2] - reference(0.)[2]).abs() > 0.1,
        "Depletion must be consequential in this oracle"
    );
    for (z_index, z) in [0., 1.5].into_iter().enumerate() {
        for (mode, global_probability) in [0., 1., 0.2].into_iter().enumerate() {
            let mut cfg = MemoryConfig::default();
            cfg.slots = 3;
            cfg.radius = Some(4.);
            cfg.depletant_radius = Some(0.5);
            cfg.depletant_activity = Some(z);
            cfg.global_probability = global_probability;
            cfg.local_translation_std_a = 0.6;
            cfg.local_small_angle_std_degrees = 25.;
            let resolved = cfg.resolve(&tree, 0.5, z).unwrap();
            let mut rng = StdRng::seed_from_u64(913810 + 13 * z_index as u64 + mode as u64);
            let mut before = Moments::default();
            let mut after = Moments::default();
            let mut change = Moments::default();
            let mut accepted = 0;
            let mut slots = [0; 3];
            for _ in 0..4000 {
                let original = (0..3).map(|_| draw(&mut rng, z)).collect::<Vec<_>>();
                let mut state = MemoryState {
                    poses: original.clone(),
                };
                state.validate(&tree, &resolved).unwrap();
                let update = state
                    .update(&tree, &resolved, GateOptions::default(), &mut rng)
                    .unwrap();
                let slot = update.slot;
                slots[slot] += 1;
                accepted += usize::from(update.accepted);
                for j in 0..3 {
                    if j != slot {
                        assert_eq!(state.poses[j], original[j]);
                    }
                }
                assert_eq!(update.old_pose, original[slot]);
                assert_eq!(update.retained_pose, state.poses[slot]);
                let old = features(original[slot]);
                let new = features(state.poses[slot]);
                assert!(new[0] >= 2. - 1e-12 && new[0] <= 4. + 1e-12);
                before.add(old);
                after.add(new);
                change.add(std::array::from_fn(|i| new[i] - old[i]));
            }
            before.check(reference(z), "iid input oracle");
            after.check(reference(z), "actual bank output");
            change.check([0.; 8], "paired memory update difference");
            assert!(
                accepted > 100,
                "A stationary identity kernel would be insufficient validation"
            );
            assert!(slots.iter().all(|n| *n > 1100 && *n < 1550));
            eprintln!(
                "memory oracle z={z}, p_global={global_probability}: 4000 iid starts PASS, accepted={accepted}, slots={slots:?}"
            );
        }
    }
}
