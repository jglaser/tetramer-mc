//! Native Rust GSD output plus lossless body-pose JSONL records.
//! Standard GSD particles are atom spheres for external visualization. Exact
//! FP64 rigid-body poses are also stored in custom log chunks for analysis.
use crate::{geometry::SphereTree, math::*};
use anyhow::Result;
use hoomd_gsd::file_layer::GsdFile;
use std::path::Path;

pub struct Trajectory {
    gsd: GsdFile,
}
impl Trajectory {
    pub fn create(path: &Path) -> Result<Self> {
        Ok(Self {
            gsd: GsdFile::create_new(path, "tetramer-mc", "hoomd", (1, 4))?,
        })
    }
    pub fn append(
        &mut self,
        step: u64,
        poses: &[Pose],
        lengths: Vec3,
        tree: &SphereTree,
    ) -> Result<()> {
        self.append_with_spherical_wall(step, poses, lengths, tree, None)
    }

    /// Spherical atom positions remain in the sphere-centered frame. GSD's
    /// standard box is display metadata; custom boundary flags define the target.
    pub fn append_with_spherical_wall(
        &mut self,
        step: u64,
        poses: &[Pose],
        lengths: Vec3,
        tree: &SphereTree,
        spherical_radius: Option<f64>,
    ) -> Result<()> {
        self.append_with_coordinate_frame(step, poses, lengths, tree, spherical_radius, [0.; 3], 0)
    }

    /// Also retain the accumulated moving-wall coordinate convention in FP64.
    #[allow(clippy::too_many_arguments)]
    pub fn append_with_coordinate_frame(
        &mut self,
        step: u64,
        poses: &[Pose],
        lengths: Vec3,
        tree: &SphereTree,
        spherical_radius: Option<f64>,
        coordinate_wall_center: Vec3,
        coordinate_origin_sweep: u64,
    ) -> Result<()> {
        let n = poses.len() * tree.shape.atoms.len();
        let center = if spherical_radius.is_some() {
            [0.; 3]
        } else {
            scale(lengths, 0.5)
        };
        let mut positions = Vec::with_capacity(n);
        let mut images = Vec::with_capacity(n);
        for pose in poses {
            let r = rotation(pose.orientation);
            for atom in &tree.shape.atoms {
                let raw = sub(add(pose.position, matvec(r, atom.center)), center);
                let image: [i32; 3] = if spherical_radius.is_some() {
                    [0; 3]
                } else {
                    std::array::from_fn(|k| (raw[k] / lengths[k] + 0.5).floor() as i32)
                };
                positions.push(std::array::from_fn::<_, 3, _>(|k| {
                    (raw[k] - f64::from(image[k]) * lengths[k]) as f32
                }));
                images.push(image);
            }
        }
        self.gsd.write_scalars("configuration/step", [step])?;
        self.gsd.write_scalars("configuration/dimensions", [3_u8])?;
        self.gsd.write_scalars(
            "configuration/box",
            [
                lengths[0] as f32,
                lengths[1] as f32,
                lengths[2] as f32,
                0.,
                0.,
                0.,
            ],
        )?;
        self.gsd.write_scalars("particles/N", [u32::try_from(n)?])?;
        self.gsd.write_arrays("particles/position", positions)?;
        self.gsd.write_arrays("particles/image", images)?;
        self.gsd.write_arrays("particles/types", [*b"atom\0"])?;
        self.gsd
            .write_scalars("particles/typeid", (0..n).map(|_| 0_u32))?;
        self.gsd.write_scalars(
            "particles/diameter",
            poses
                .iter()
                .flat_map(|_| tree.shape.atoms.iter().map(|a| (2. * a.radius) as f32)),
        )?;
        // A visualization body label is NOT a HOOMD rigid-body center index.
        self.gsd.write_scalars(
            "log/tetramer_mc/body_id",
            (0..poses.len()).flat_map(|i| std::iter::repeat_n(i as u32, tree.shape.atoms.len())),
        )?;
        self.gsd.write_arrays(
            "log/tetramer_mc/body_position",
            poses.iter().map(|p| p.position),
        )?;
        self.gsd.write_arrays(
            "log/tetramer_mc/body_orientation",
            poses.iter().map(|p| p.orientation),
        )?;
        self.gsd
            .write_scalars("log/tetramer_mc/box_lengths", lengths)?;
        self.gsd.write_scalars(
            "log/tetramer_mc/periodic",
            [u8::from(spherical_radius.is_none()); 3],
        )?;
        if let Some(radius) = spherical_radius {
            self.gsd.write_scalars(
                "log/tetramer_mc/coordinate_wall_center",
                coordinate_wall_center,
            )?;
            self.gsd.write_scalars(
                "log/tetramer_mc/coordinate_origin_sweep",
                [coordinate_origin_sweep],
            )?;
            self.gsd
                .write_scalars("log/tetramer_mc/spherical_wall_radius", [radius])?;
            self.gsd
                .write_scalars("log/tetramer_mc/bath_wall_permeable", [1_u8])?;
        }
        self.gsd.end_frame()?;
        Ok(())
    }
    pub fn sync(&mut self) -> Result<()> {
        self.gsd.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Atom, Shape};
    #[test]
    fn gsd_roundtrip() -> Result<()> {
        let path = std::env::temp_dir().join(format!("tetramer-mc-gsd-{}.gsd", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let tree = SphereTree::new(Shape {
            name: "sphere".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        })?;
        let pose = Pose {
            position: [0.123456789012345, 4., 9.],
            orientation: [1., 0., 0., 0.],
        };
        {
            let mut out = Trajectory::create(&path)?;
            out.append(0, &[pose], [20.; 3], &tree)?;
            out.append(5, &[pose], [20.; 3], &tree)?;
            out.sync()?;
        }
        let gsd = GsdFile::open(&path, hoomd_gsd::file_layer::Mode::Read)?;
        assert_eq!(gsd.n_frames(), 2);
        let position: Vec<[f64; 3]> = gsd
            .iter_arrays(1, "log/tetramer_mc/body_position")?
            .collect();
        assert_eq!(position, vec![pose.position]);
        let lengths: Vec<f64> = gsd
            .iter_scalars(1, "log/tetramer_mc/box_lengths")?
            .collect();
        assert_eq!(lengths, vec![20.; 3]);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
