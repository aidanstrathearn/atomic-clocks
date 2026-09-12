use std::ops::{Add, Mul, Sub};

use super::vec3::Vec3;

/// A real 3×3 matrix stored in row order, acting on column vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    rows: [[f64; 3]; 3],
}

impl Mat3 {
    pub fn from_rows(rows: [[f64; 3]; 3]) -> Self {
        Self { rows }
    }

    pub fn from_columns(columns: [[f64; 3]; 3]) -> Self {
        Self::from_rows(columns).transpose()
    }

    pub fn zero() -> Self {
        Self::from_rows([[0.0; 3]; 3])
    }

    pub fn identity() -> Self {
        Self::diagonal(Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        })
    }

    pub fn diagonal(diagonal: Vec3) -> Self {
        Self::from_rows([
            [diagonal.x, 0.0, 0.0],
            [0.0, diagonal.y, 0.0],
            [0.0, 0.0, diagonal.z],
        ])
    }

    pub fn rows(&self) -> &[[f64; 3]; 3] {
        &self.rows
    }

    pub fn apply_to_vec(&self, v: Vec3) -> Vec3 {
        let [x, y, z] = self
            .rows
            .map(|row| row[0] * v.x + row[1] * v.y + row[2] * v.z);
        Vec3 { x, y, z }
    }

    pub fn transpose(&self) -> Self {
        Self::from_rows(std::array::from_fn(|i| {
            std::array::from_fn(|j| self.rows[j][i])
        }))
    }
}

impl Add for Mat3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_rows(std::array::from_fn(|i| {
            std::array::from_fn(|j| self.rows[i][j] + rhs.rows[i][j])
        }))
    }
}

impl Sub for Mat3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_rows(std::array::from_fn(|i| {
            std::array::from_fn(|j| self.rows[i][j] - rhs.rows[i][j])
        }))
    }
}

impl Mul<f64> for Mat3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self::from_rows(self.rows.map(|row| row.map(|value| value * rhs)))
    }
}

impl Mul<Mat3> for f64 {
    type Output = Mat3;

    fn mul(self, rhs: Mat3) -> Mat3 {
        rhs * self
    }
}

impl Mul for Mat3 {
    type Output = Self;

    /// `later * earlier` applies `earlier` first, then `later`.
    fn mul(self, rhs: Self) -> Self {
        Self::from_rows(std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| self.rows[i][k] * rhs.rows[k][j]).sum())
        }))
    }
}
