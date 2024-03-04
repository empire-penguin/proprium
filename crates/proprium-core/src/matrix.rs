use std::{
    fmt,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

use crate::{Real, Vec2D, Vec3D, Vec4D, Vector};

const MIN_ALLOWED_DET: Real = 1.0e-6;

/// An error type for matrix operations.
#[derive(Debug)]
pub enum MatrixError {
    SingularMatrix,
    DimensionMismatch,
    RowIndexOutOfBounds,
    ColIndexOutOfBounds,
}
impl std::error::Error for MatrixError {}
impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingularMatrix => write!(f, "Singular matrix"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
            Self::RowIndexOutOfBounds => write!(f, "Row index out of bounds"),
            Self::ColIndexOutOfBounds => write!(f, "Column index out of bounds"),
        }
    }
}

/// A trait for matrix operations.
/// Row major order is used for the matrix.
pub trait Matrix:
    Sized + Add<Self, Output = Self> + Sub<Self, Output = Self> + Mul<Self, Output = Self>
{
    fn determinant(&self) -> Real;
    fn transpose(&self) -> Self;
    fn inverse(&self) -> Result<Self, MatrixError>;

    fn get(&self, row: usize, col: usize) -> Result<Real, MatrixError>;
    fn set(&mut self, row: usize, col: usize, value: Real) -> Result<(), MatrixError>;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat2x2 {
    pub data: [Vec2D; 2],
}

impl Add for Mat2x2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: [self.data[0] + rhs.data[0], self.data[1] + rhs.data[1]],
        }
    }
}

impl Sub for Mat2x2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: [self.data[0] - rhs.data[0], self.data[1] - rhs.data[1]],
        }
    }
}

impl Mul for Mat2x2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                Vec2D::new(
                    self.data[0].x() * rhs.data[0].x() + self.data[0].y() * rhs.data[1].x(),
                    self.data[0].x() * rhs.data[0].y() + self.data[0].y() * rhs.data[1].y(),
                ),
                Vec2D::new(
                    self.data[1].x() * rhs.data[0].x() + self.data[1].y() * rhs.data[1].x(),
                    self.data[1].x() * rhs.data[0].y() + self.data[1].y() * rhs.data[1].y(),
                ),
            ],
        }
    }
}

impl Mul<Vec2D> for Mat2x2 {
    type Output = Vec2D;

    fn mul(self, rhs: Vec2D) -> Self::Output {
        Vec2D::new(
            self.data[0].x() * rhs.x() + self.data[0].y() * rhs.y(),
            self.data[1].x() * rhs.x() + self.data[1].y() * rhs.y(),
        )
    }
}

impl Mul<Real> for Mat2x2 {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self::Output {
        Self {
            data: [self.data[0] * rhs, self.data[1] * rhs],
        }
    }
}

impl Mul<Mat2x2> for Real {
    type Output = Mat2x2;

    fn mul(self, rhs: Mat2x2) -> Self::Output {
        rhs * self
    }
}

impl Index<usize> for Mat2x2 {
    type Output = Vec2D;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.data[0],
            1 => &self.data[1],
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Mat2x2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.data[0],
            1 => &mut self.data[1],
            _ => panic!("Index out of bounds"),
        }
    }
}

impl Matrix for Mat2x2 {
    fn determinant(&self) -> Real {
        self.data[0].x() * self.data[1].y() - self.data[0].y() * self.data[1].x()
    }

    fn transpose(&self) -> Self {
        Self {
            data: [
                Vec2D::new(self.data[0].x(), self.data[1].x()),
                Vec2D::new(self.data[0].y(), self.data[1].y()),
            ],
        }
    }

    fn inverse(&self) -> Result<Self, MatrixError> {
        let det = self.determinant();
        println!("det: {}", det);
        if det.abs() < MIN_ALLOWED_DET {
            return Err(MatrixError::SingularMatrix);
        }
        let inv_det = 1.0 / det;
        Ok(Self {
            data: [
                Vec2D::new(self.data[1].y() * inv_det, -self.data[0].y() * inv_det),
                Vec2D::new(-self.data[1].x() * inv_det, self.data[0].x() * inv_det),
            ],
        })
    }

    fn get(&self, row: usize, col: usize) -> Result<Real, MatrixError> {
        match (row, col) {
            (row, col) if row < 2 && col < 2 => Ok(self.data[row][col]),
            (_, col) if col >= 2 => Err(MatrixError::ColIndexOutOfBounds),
            (row, _) if row >= 2 => Err(MatrixError::RowIndexOutOfBounds),
            _ => unreachable!(),
        }
    }

    fn set(&mut self, row: usize, col: usize, value: Real) -> Result<(), MatrixError> {
        match (row, col) {
            (row, col) if row < 2 && col < 2 => {
                self.data[row][col] = value;
                Ok(())
            }
            (_, col) if col >= 2 => Err(MatrixError::ColIndexOutOfBounds),
            (row, _) if row >= 2 => Err(MatrixError::RowIndexOutOfBounds),
            _ => unreachable!(),
        }
    }
}

impl Mat2x2 {
    pub fn new(a: Real, b: Real, c: Real, d: Real) -> Self {
        Self {
            data: [Vec2D::new(a, b), Vec2D::new(c, d)],
        }
    }

    pub fn default() -> Self {
        Self {
            data: [Vec2D::default(), Vec2D::default()],
        }
    }

    pub fn identity() -> Self {
        Self {
            data: [Vec2D::new(1.0, 0.0), Vec2D::new(0.0, 1.0)],
        }
    }

    fn get_row(&self, row: usize) -> Result<Vec2D, MatrixError> {
        if row < 2 {
            Ok(self.data[row])
        } else {
            Err(MatrixError::RowIndexOutOfBounds)
        }
    }

    fn get_col(&self, col: usize) -> Result<Vec2D, MatrixError> {
        if col < 2 {
            Ok(Vec2D::new(self.data[0][col], self.data[1][col]))
        } else {
            Err(MatrixError::ColIndexOutOfBounds)
        }
    }

    fn set_row(&mut self, row: usize, values: Vec2D) -> Result<(), MatrixError> {
        if row < 2 {
            self.data[row] = values;
            Ok(())
        } else {
            Err(MatrixError::RowIndexOutOfBounds)
        }
    }

    fn set_col(&mut self, col: usize, values: Vec2D) -> Result<(), MatrixError> {
        if col < 2 {
            self.data[0][col] = values.x();
            self.data[1][col] = values.y();
            Ok(())
        } else {
            Err(MatrixError::ColIndexOutOfBounds)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat3x3 {
    pub data: [Vec3D; 3],
}

impl Add for Mat3x3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                self.data[0] + rhs.data[0],
                self.data[1] + rhs.data[1],
                self.data[2] + rhs.data[2],
            ],
        }
    }
}

impl Sub for Mat3x3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                self.data[0] - rhs.data[0],
                self.data[1] - rhs.data[1],
                self.data[2] - rhs.data[2],
            ],
        }
    }
}

impl Mul for Mat3x3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                Vec3D::new(
                    self.data[0].x() * rhs.data[0].x()
                        + self.data[0].y() * rhs.data[1].x()
                        + self.data[0].z() * rhs.data[2].x(),
                    self.data[0].x() * rhs.data[0].y()
                        + self.data[0].y() * rhs.data[1].y()
                        + self.data[0].z() * rhs.data[2].y(),
                    self.data[0].x() * rhs.data[0].z()
                        + self.data[0].y() * rhs.data[1].z()
                        + self.data[0].z() * rhs.data[2].z(),
                ),
                Vec3D::new(
                    self.data[1].x() * rhs.data[0].x()
                        + self.data[1].y() * rhs.data[1].x()
                        + self.data[1].z() * rhs.data[2].x(),
                    self.data[1].x() * rhs.data[0].y()
                        + self.data[1].y() * rhs.data[1].y()
                        + self.data[1].z() * rhs.data[2].y(),
                    self.data[1].x() * rhs.data[0].z()
                        + self.data[1].y() * rhs.data[1].z()
                        + self.data[1].z() * rhs.data[2].z(),
                ),
                Vec3D::new(
                    self.data[2].x() * rhs.data[0].x()
                        + self.data[2].y() * rhs.data[1].x()
                        + self.data[2].z() * rhs.data[2].x(),
                    self.data[2].x() * rhs.data[0].y()
                        + self.data[2].y() * rhs.data[1].y()
                        + self.data[2].z() * rhs.data[2].y(),
                    self.data[2].x() * rhs.data[0].z()
                        + self.data[2].y() * rhs.data[1].z()
                        + self.data[2].z() * rhs.data[2].z(),
                ),
            ],
        }
    }
}

impl Mul<Vec3D> for Mat3x3 {
    type Output = Vec3D;

    fn mul(self, rhs: Vec3D) -> Self::Output {
        Vec3D::new(
            self.data[0].x() * rhs.x() + self.data[0].y() * rhs.y() + self.data[0].z() * rhs.z(),
            self.data[1].x() * rhs.x() + self.data[1].y() * rhs.y() + self.data[1].z() * rhs.z(),
            self.data[2].x() * rhs.x() + self.data[2].y() * rhs.y() + self.data[2].z() * rhs.z(),
        )
    }
}

impl Mul<Real> for Mat3x3 {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self::Output {
        Self {
            data: [self.data[0] * rhs, self.data[1] * rhs, self.data[2] * rhs],
        }
    }
}

impl Mul<Mat3x3> for Real {
    type Output = Mat3x3;

    fn mul(self, rhs: Mat3x3) -> Self::Output {
        rhs * self
    }
}

impl Index<usize> for Mat3x3 {
    type Output = Vec3D;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.data[0],
            1 => &self.data[1],
            2 => &self.data[2],
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Mat3x3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.data[0],
            1 => &mut self.data[1],
            2 => &mut self.data[2],
            _ => panic!("Index out of bounds"),
        }
    }
}

impl Matrix for Mat3x3 {
    fn determinant(&self) -> Real {
        self.data[0].x()
            * (self.data[1].y() * self.data[2].z() - self.data[1].z() * self.data[2].y())
            - self.data[0].y()
                * (self.data[1].x() * self.data[2].z() - self.data[1].z() * self.data[2].x())
            + self.data[0].z()
                * (self.data[1].x() * self.data[2].y() - self.data[1].y() * self.data[2].x())
    }

    fn transpose(&self) -> Self {
        Self {
            data: [
                Vec3D::new(self.data[0].x(), self.data[1].x(), self.data[2].x()),
                Vec3D::new(self.data[0].y(), self.data[1].y(), self.data[2].y()),
                Vec3D::new(self.data[0].z(), self.data[1].z(), self.data[2].z()),
            ],
        }
    }

    fn inverse(&self) -> Result<Self, MatrixError> {
        let det = self.determinant();
        if det.abs() < MIN_ALLOWED_DET {
            return Err(MatrixError::SingularMatrix);
        }
        let inv_det = 1.0 / det;
        let a = self.data[0].x();
        let b = self.data[0].y();
        let c = self.data[0].z();
        let d = self.data[1].x();
        let e = self.data[1].y();
        let f = self.data[1].z();
        let g = self.data[2].x();
        let h = self.data[2].y();
        let i = self.data[2].z();

        let inv = Self {
            data: [
                Vec3D::new(e * i - f * h, c * h - b * i, b * f - c * e),
                Vec3D::new(f * g - d * i, a * i - c * g, c * d - a * f),
                Vec3D::new(d * h - e * g, b * g - a * h, a * e - b * d),
            ],
        };

        Ok(inv * inv_det)
    }

    fn get(&self, row: usize, col: usize) -> Result<Real, MatrixError> {
        match (row, col) {
            (row, col) if row < 3 && col < 3 => Ok(self.data[row][col]),
            (_, col) if col >= 3 => Err(MatrixError::ColIndexOutOfBounds),
            (row, _) if row >= 3 => Err(MatrixError::RowIndexOutOfBounds),
            _ => unreachable!(),
        }
    }

    fn set(&mut self, row: usize, col: usize, value: Real) -> Result<(), MatrixError> {
        match (row, col) {
            (row, col) if row < 3 && col < 3 => {
                self.data[row][col] = value;
                Ok(())
            }
            (_, col) if col >= 3 => Err(MatrixError::ColIndexOutOfBounds),
            (row, _) if row >= 3 => Err(MatrixError::RowIndexOutOfBounds),
            _ => unreachable!(),
        }
    }
}

impl Mat3x3 {
    pub fn new(
        a: Real,
        b: Real,
        c: Real,
        d: Real,
        e: Real,
        f: Real,
        g: Real,
        h: Real,
        i: Real,
    ) -> Self {
        Self {
            data: [
                Vec3D::new(a, b, c),
                Vec3D::new(d, e, f),
                Vec3D::new(g, h, i),
            ],
        }
    }

    pub fn default() -> Self {
        Self {
            data: [Vec3D::default(), Vec3D::default(), Vec3D::default()],
        }
    }

    pub fn identity() -> Self {
        Self {
            data: [
                Vec3D::new(1.0, 0.0, 0.0),
                Vec3D::new(0.0, 1.0, 0.0),
                Vec3D::new(0.0, 0.0, 1.0),
            ],
        }
    }

    fn get_row(&self, row: usize) -> Result<Vec3D, MatrixError> {
        if row < 3 {
            Ok(self.data[row])
        } else {
            Err(MatrixError::RowIndexOutOfBounds)
        }
    }

    fn get_col(&self, col: usize) -> Result<Vec3D, MatrixError> {
        if col < 3 {
            Ok(Vec3D::new(
                self.data[0][col],
                self.data[1][col],
                self.data[2][col],
            ))
        } else {
            Err(MatrixError::ColIndexOutOfBounds)
        }
    }

    fn set_row(&mut self, row: usize, values: Vec3D) -> Result<(), MatrixError> {
        if row < 3 {
            self.data[row] = values;
            Ok(())
        } else {
            Err(MatrixError::RowIndexOutOfBounds)
        }
    }

    fn set_col(&mut self, col: usize, values: Vec3D) -> Result<(), MatrixError> {
        if col < 3 {
            self.data[0][col] = values.x();
            self.data[1][col] = values.y();
            self.data[2][col] = values.z();
            Ok(())
        } else {
            Err(MatrixError::ColIndexOutOfBounds)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat4x4 {
    pub data: [Vec4D; 4],
}

impl Add for Mat4x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                self.data[0] + rhs.data[0],
                self.data[1] + rhs.data[1],
                self.data[2] + rhs.data[2],
                self.data[3] + rhs.data[3],
            ],
        }
    }
}

impl Sub for Mat4x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                self.data[0] - rhs.data[0],
                self.data[1] - rhs.data[1],
                self.data[2] - rhs.data[2],
                self.data[3] - rhs.data[3],
            ],
        }
    }
}

impl Mul for Mat4x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: [
                Vec4D::new(
                    self.data[0].x() * rhs.data[0].x()
                        + self.data[0].y() * rhs.data[1].x()
                        + self.data[0].z() * rhs.data[2].x()
                        + self.data[0].w() * rhs.data[3].x(),
                    self.data[0].x() * rhs.data[0].y()
                        + self.data[0].y() * rhs.data[1].y()
                        + self.data[0].z() * rhs.data[2].y()
                        + self.data[0].w() * rhs.data[3].y(),
                    self.data[0].x() * rhs.data[0].z()
                        + self.data[0].y() * rhs.data[1].z()
                        + self.data[0].z() * rhs.data[2].z()
                        + self.data[0].w() * rhs.data[3].z(),
                    self.data[0].x() * rhs.data[0].w()
                        + self.data[0].y() * rhs.data[1].w()
                        + self.data[0].z() * rhs.data[2].w()
                        + self.data[0].w() * rhs.data[3].w(),
                ),
                Vec4D::new(
                    self.data[1].x() * rhs.data[0].x()
                        + self.data[1].y() * rhs.data[1].x()
                        + self.data[1].z() * rhs.data[2].x()
                        + self.data[1].w() * rhs.data[3].x(),
                    self.data[1].x() * rhs.data[0].y()
                        + self.data[1].y() * rhs.data[1].y()
                        + self.data[1].z() * rhs.data[2].y()
                        + self.data[1].w() * rhs.data[3].y(),
                    self.data[1].x() * rhs.data[0].z()
                        + self.data[1].y() * rhs.data[1].z()
                        + self.data[1].z() * rhs.data[2].z()
                        + self.data[1].w() * rhs.data[3].z(),
                    self.data[1].x() * rhs.data[0].w()
                        + self.data[1].y() * rhs.data[1].w()
                        + self.data[1].z() * rhs.data[2].w()
                        + self.data[1].w() * rhs.data[3].w(),
                ),
                Vec4D::new(
                    self.data[2].x() * rhs.data[0].x()
                        + self.data[2].y() * rhs.data[1].x()
                        + self.data[2].z() * rhs.data[2].x()
                        + self.data[2].w() * rhs.data[3].x(),
                    self.data[2].x() * rhs.data[0].y()
                        + self.data[2].y() * rhs.data[1].y()
                        + self.data[2].z() * rhs.data[2].y()
                        + self.data[2].w() * rhs.data[3].y(),
                    self.data[2].x() * rhs.data[0].z()
                        + self.data[2].y() * rhs.data[1].z()
                        + self.data[2].z() * rhs.data[2].z()
                        + self.data[2].w() * rhs.data[3].z(),
                    self.data[2].x() * rhs.data[0].w()
                        + self.data[2].y() * rhs.data[1].w()
                        + self.data[2].z() * rhs.data[2].w()
                        + self.data[2].w() * rhs.data[3].w(),
                ),
                Vec4D::new(
                    self.data[3].x() * rhs.data[0].x()
                        + self.data[3].y() * rhs.data[1].x()
                        + self.data[3].z() * rhs.data[2].x()
                        + self.data[3].w() * rhs.data[3].x(),
                    self.data[3].x() * rhs.data[0].y()
                        + self.data[3].y() * rhs.data[1].y()
                        + self.data[3].z() * rhs.data[2].y()
                        + self.data[3].w() * rhs.data[3].y(),
                    self.data[3].x() * rhs.data[0].z()
                        + self.data[3].y() * rhs.data[1].z()
                        + self.data[3].z() * rhs.data[2].z()
                        + self.data[3].w() * rhs.data[3].z(),
                    self.data[3].x() * rhs.data[0].w()
                        + self.data[3].y() * rhs.data[1].w()
                        + self.data[3].z() * rhs.data[2].w()
                        + self.data[3].w() * rhs.data[3].w(),
                ),
            ],
        }
    }
}

impl Mul<Vec4D> for Mat4x4 {
    type Output = Vec4D;

    fn mul(self, rhs: Vec4D) -> Self::Output {
        Vec4D::new(
            self.data[0].x() * rhs.x()
                + self.data[0].y() * rhs.y()
                + self.data[0].z() * rhs.z()
                + self.data[0].w() * rhs.w(),
            self.data[1].x() * rhs.x()
                + self.data[1].y() * rhs.y()
                + self.data[1].z() * rhs.z()
                + self.data[1].w() * rhs.w(),
            self.data[2].x() * rhs.x()
                + self.data[2].y() * rhs.y()
                + self.data[2].z() * rhs.z()
                + self.data[2].w() * rhs.w(),
            self.data[3].x() * rhs.x()
                + self.data[3].y() * rhs.y()
                + self.data[3].z() * rhs.z()
                + self.data[3].w() * rhs.w(),
        )
    }
}

impl Mul<Real> for Mat4x4 {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self::Output {
        Self {
            data: [
                self.data[0] * rhs,
                self.data[1] * rhs,
                self.data[2] * rhs,
                self.data[3] * rhs,
            ],
        }
    }
}

impl Mul<Mat4x4> for Real {
    type Output = Mat4x4;

    fn mul(self, rhs: Mat4x4) -> Self::Output {
        rhs * self
    }
}

impl Matrix for Mat4x4 {
    fn determinant(&self) -> Real {
        let a = self.data[0].x();
        let b = self.data[0].y();
        let c = self.data[0].z();
        let d = self.data[0].w();
        let e = self.data[1].x();
        let f = self.data[1].y();
        let g = self.data[1].z();
        let h = self.data[1].w();
        let i = self.data[2].x();
        let j = self.data[2].y();
        let k = self.data[2].z();
        let l = self.data[2].w();
        let m = self.data[3].x();
        let n = self.data[3].y();
        let o = self.data[3].z();
        let p = self.data[3].w();

        a * f * k * p - a * f * l * o - a * g * j * p + a * g * l * n + a * h * j * o
            - a * h * k * n
            - b * e * k * p
            + b * e * l * o
            + b * g * i * p
            - b * g * l * m
            - b * h * i * o
            + b * h * k * m
            + c * e * j * p
            - c * e * l * n
            - c * f * i * p
            + c * f * l * m
            + c * h * i * n
            - c * h * j * m
            - d * e * j * o
            + d * e * k * n
            + d * f * i * o
            - d * f * k * m
            - d * g * i * n
            + d * g * j * m
    }

    fn transpose(&self) -> Self {
        Self {
            data: [
                Vec4D::new(
                    self.data[0].x(),
                    self.data[1].x(),
                    self.data[2].x(),
                    self.data[3].x(),
                ),
                Vec4D::new(
                    self.data[0].y(),
                    self.data[1].y(),
                    self.data[2].y(),
                    self.data[3].y(),
                ),
                Vec4D::new(
                    self.data[0].z(),
                    self.data[1].z(),
                    self.data[2].z(),
                    self.data[3].z(),
                ),
                Vec4D::new(
                    self.data[0].w(),
                    self.data[1].w(),
                    self.data[2].w(),
                    self.data[3].w(),
                ),
            ],
        }
    }

    fn inverse(&self) -> Result<Self, MatrixError> {
        let det = self.determinant();
        if det.abs() < MIN_ALLOWED_DET {
            return Err(MatrixError::SingularMatrix);
        }
        let inv_det = 1.0 / det;
        let a = self.data[0].x();
        let b = self.data[0].y();
        let c = self.data[0].z();
        let d = self.data[0].w();
        let e = self.data[1].x();
        let f = self.data[1].y();
        let g = self.data[1].z();
        let h = self.data[1].w();
        let i = self.data[2].x();
        let j = self.data[2].y();
        let k = self.data[2].z();
        let l = self.data[2].w();
        let m = self.data[3].x();
        let n = self.data[3].y();
        let o = self.data[3].z();
        let p = self.data[3].w();

        let inv = Self {
            data: [
                Vec4D::new(
                    f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n),
                    -(b * (k * p - l * o) - c * (j * p - l * n) + d * (j * o - k * n)),
                    b * (g * p - h * o) - c * (f * p - h * n) + d * (f * o - g * n),
                    -(b * (g * l - h * k) - c * (f * l - h * j) + d * (f * k - g * j)),
                ),
                Vec4D::new(
                    -(e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m)),
                    a * (k * p - l * o) - c * (i * p - l * m) + d * (i * o - k * m),
                    -(a * (g * p - h * o) - c * (e * p - h * m) + d * (e * o - g * m)),
                    a * (g * l - h * k) - c * (e * l - h * i) + d * (e * k - g * i),
                ),
                Vec4D::new(
                    e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m),
                    -(a * (j * p - l * n) - b * (i * p - l * m) + d * (i * n - j * m)),
                    a * (f * p - h * m) - b * (e * p - h * m) + d * (e * n - f * m),
                    -(a * (f * l - h * i) - b * (e * l - h * i) + d * (e * j - f * i)),
                ),
                Vec4D::new(
                    -(e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m)),
                    a * (j * o - k * n) - b * (i * o - k * m) + c * (i * n - j * m),
                    -(a * (f * o - g * m) - b * (e * o - g * m) + c * (e * n - f * m)),
                    a * (f * k - g * j) - b * (e * k - g * i) + c * (e * j - f * i),
                ),
            ],
        };

        Ok(inv * inv_det)
    }

    fn get(&self, row: usize, col: usize) -> Result<Real, MatrixError> {
        match (row, col) {
            (row, col) if row < 4 && col < 4 => Ok(self.data[row][col]),
            (_, col) if col >= 4 => Err(MatrixError::ColIndexOutOfBounds),
            (row, _) if row >= 4 => Err(MatrixError::RowIndexOutOfBounds),
            _ => unreachable!(),
        }
    }

    fn set(&mut self, row: usize, col: usize, value: Real) -> Result<(), MatrixError> {
        match (row, col) {
            (row, col) if row < 4 && col < 4 => {
                self.data[row][col] = value;
                Ok(())
            }
            (_, col) if col >= 4 => Err(MatrixError::ColIndexOutOfBounds),
            (row, _) if row >= 4 => Err(MatrixError::RowIndexOutOfBounds),
            _ => unreachable!(),
        }
    }
}

impl Mat4x4 {
    pub fn new(
        a: Real,
        b: Real,
        c: Real,
        d: Real,
        e: Real,
        f: Real,
        g: Real,
        h: Real,
        i: Real,
        j: Real,
        k: Real,
        l: Real,
        m: Real,
        n: Real,
        o: Real,
        p: Real,
    ) -> Self {
        Self {
            data: [
                Vec4D::new(a, b, c, d),
                Vec4D::new(e, f, g, h),
                Vec4D::new(i, j, k, l),
                Vec4D::new(m, n, o, p),
            ],
        }
    }

    pub fn default() -> Self {
        Self {
            data: [
                Vec4D::default(),
                Vec4D::default(),
                Vec4D::default(),
                Vec4D::default(),
            ],
        }
    }

    pub fn identity() -> Self {
        Self {
            data: [
                Vec4D::new(1.0, 0.0, 0.0, 0.0),
                Vec4D::new(0.0, 1.0, 0.0, 0.0),
                Vec4D::new(0.0, 0.0, 1.0, 0.0),
                Vec4D::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    fn get_row(&self, row: usize) -> Result<Vec4D, MatrixError> {
        if row < 4 {
            Ok(self.data[row])
        } else {
            Err(MatrixError::RowIndexOutOfBounds)
        }
    }

    fn get_col(&self, col: usize) -> Result<Vec4D, MatrixError> {
        if col < 4 {
            Ok(Vec4D::new(
                self.data[0][col],
                self.data[1][col],
                self.data[2][col],
                self.data[3][col],
            ))
        } else {
            Err(MatrixError::ColIndexOutOfBounds)
        }
    }

    fn set_row(&mut self, row: usize, values: Vec4D) -> Result<(), MatrixError> {
        if row < 4 {
            self.data[row] = values;
            Ok(())
        } else {
            Err(MatrixError::RowIndexOutOfBounds)
        }
    }

    fn set_col(&mut self, col: usize, values: Vec4D) -> Result<(), MatrixError> {
        if col < 4 {
            self.data[0][col] = values.x();
            self.data[1][col] = values.y();
            self.data[2][col] = values.z();
            self.data[3][col] = values.w();
            Ok(())
        } else {
            Err(MatrixError::ColIndexOutOfBounds)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tests for Mat2x2
    #[test]
    fn test_mat2x2_ops() {
        let m = Mat2x2 {
            data: [Vec2D::new(1.0, 2.0), Vec2D::new(3.0, 4.0)],
        };
        let n = Mat2x2 {
            data: [Vec2D::new(5.0, 6.0), Vec2D::new(7.0, 8.0)],
        };

        let sum = m + n;
        let expected = Mat2x2 {
            data: [Vec2D::new(6.0, 8.0), Vec2D::new(10.0, 12.0)],
        };
        assert_eq!(sum, expected);

        let diff = m - n;
        let expected = Mat2x2 {
            data: [Vec2D::new(-4.0, -4.0), Vec2D::new(-4.0, -4.0)],
        };
        assert_eq!(diff, expected);

        let prod = m * n;
        let expected = Mat2x2 {
            data: [Vec2D::new(19.0, 22.0), Vec2D::new(43.0, 50.0)],
        };
        assert_eq!(prod, expected);

        let v = Vec2D::new(1.0, 2.0);
        let prod = m * v;
        let expected = Vec2D::new(5.0, 11.0);
        assert_eq!(prod, expected);
    }

    #[test]
    fn test_mat2x2_determinant() {
        let m = Mat2x2 {
            data: [Vec2D::new(1.0, 2.0), Vec2D::new(3.0, 4.0)],
        };
        let det = m.determinant();
        assert_eq!(det, -2.0);
    }

    #[test]
    fn test_mat2x2_transpose() {
        let m = Mat2x2 {
            data: [Vec2D::new(1.0, 2.0), Vec2D::new(3.0, 4.0)],
        };
        let transposed = m.transpose();
        let expected = Mat2x2 {
            data: [Vec2D::new(1.0, 3.0), Vec2D::new(2.0, 4.0)],
        };
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_mat2x2_inverse() {
        let m = Mat2x2 {
            data: [Vec2D::new(1.0, 2.0), Vec2D::new(3.0, 4.0)],
        };
        let inv = m.inverse().unwrap();
        let expected = Mat2x2 {
            data: [Vec2D::new(-2.0, 1.0), Vec2D::new(1.5, -0.5)],
        };
        assert_eq!(inv, expected);
    }

    #[test]
    fn test_mat2x2_get() {
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        let value00 = m.get(0, 0).unwrap();
        let value01 = m.get(0, 1).unwrap();
        let value10 = m.get(1, 0).unwrap();
        let value11 = m.get(1, 1).unwrap();
        assert_eq!(value00, 1.0);
        assert_eq!(value01, 2.0);
        assert_eq!(value10, 3.0);
        assert_eq!(value11, 4.0);
    }

    #[test]
    fn test_mat2x2_set() {
        let mut m = Mat2x2::identity();

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 0.0);
        assert_eq!(m[1][0], 0.0);
        assert_eq!(m[1][1], 1.0);

        m.set(0, 0, 2.0).unwrap();
        m.set(0, 1, 3.0).unwrap();
        m.set(1, 0, 4.0).unwrap();
        m.set(1, 1, 5.0).unwrap();
        let expected = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(m, expected);
    }

    /// Tests for Mat3x3
    #[test]
    fn test_mat3x3_ops() {
        let m = Mat3x3 {
            data: [
                Vec3D::new(1.0, 2.0, 3.0),
                Vec3D::new(4.0, 5.0, 6.0),
                Vec3D::new(7.0, 8.0, 9.0),
            ],
        };
        let n = Mat3x3 {
            data: [
                Vec3D::new(9.0, 8.0, 7.0),
                Vec3D::new(6.0, 5.0, 4.0),
                Vec3D::new(3.0, 2.0, 1.0),
            ],
        };

        let sum = m + n;
        let expected = Mat3x3 {
            data: [
                Vec3D::new(10.0, 10.0, 10.0),
                Vec3D::new(10.0, 10.0, 10.0),
                Vec3D::new(10.0, 10.0, 10.0),
            ],
        };
        assert_eq!(sum, expected);

        let diff = m - n;
        let expected = Mat3x3 {
            data: [
                Vec3D::new(-8.0, -6.0, -4.0),
                Vec3D::new(-2.0, 0.0, 2.0),
                Vec3D::new(4.0, 6.0, 8.0),
            ],
        };
        assert_eq!(diff, expected);

        let prod = m * n;
        let expected = Mat3x3 {
            data: [
                Vec3D::new(30.0, 24.0, 18.0),
                Vec3D::new(84.0, 69.0, 54.0),
                Vec3D::new(138.0, 114.0, 90.0),
            ],
        };
        assert_eq!(prod, expected);

        let v = Vec3D::new(1.0, 2.0, 3.0);
        let prod = m * v;
        let expected = Vec3D::new(14.0, 32.0, 50.0);
        assert_eq!(prod, expected);

        let m = Mat3x3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let v = Vec3D::new(1.0, 2.0, 3.0);
        let prod = m * v;
        let expected = Vec3D::new(14.0, 32.0, 50.0);
        assert_eq!(prod, expected);
    }

    #[test]
    fn test_mat3x3_determinant() {
        let m = Mat3x3 {
            data: [
                Vec3D::new(1.0, 2.0, 3.0),
                Vec3D::new(4.0, 5.0, 6.0),
                Vec3D::new(7.0, 8.0, 9.0),
            ],
        };
        let det = m.determinant();
        assert_eq!(det, 0.0);
    }

    #[test]
    fn test_mat3x3_transpose() {
        let m = Mat3x3 {
            data: [
                Vec3D::new(1.0, 2.0, 3.0),
                Vec3D::new(4.0, 5.0, 6.0),
                Vec3D::new(7.0, 8.0, 9.0),
            ],
        };
        let transposed = m.transpose();
        let expected = Mat3x3 {
            data: [
                Vec3D::new(1.0, 4.0, 7.0),
                Vec3D::new(2.0, 5.0, 8.0),
                Vec3D::new(3.0, 6.0, 9.0),
            ],
        };
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_mat3x3_inverse() {
        let m = Mat3x3 {
            data: [
                Vec3D::new(1.0, 2.0, 3.0),
                Vec3D::new(0.0, 1.0, 4.0),
                Vec3D::new(5.0, 6.0, 0.0),
            ],
        };
        let inv = m.inverse().unwrap();
        let expected = Mat3x3 {
            data: [
                Vec3D::new(-24.0, 18.0, 5.0),
                Vec3D::new(20.0, -15.0, -4.0),
                Vec3D::new(-5.0, 4.0, 1.0),
            ],
        };
        assert_eq!(inv, expected);
    }

    /// Tests for Mat4x4
    #[test]
    fn test_mat4x4_ops() {
        let m = Mat4x4 {
            data: [
                Vec4D::new(1.0, 2.0, 3.0, 4.0),
                Vec4D::new(5.0, 6.0, 7.0, 8.0),
                Vec4D::new(9.0, 10.0, 11.0, 12.0),
                Vec4D::new(13.0, 14.0, 15.0, 16.0),
            ],
        };
        let n = Mat4x4 {
            data: [
                Vec4D::new(16.0, 15.0, 14.0, 13.0),
                Vec4D::new(12.0, 11.0, 10.0, 9.0),
                Vec4D::new(8.0, 7.0, 6.0, 5.0),
                Vec4D::new(4.0, 3.0, 2.0, 1.0),
            ],
        };

        let sum = m + n;
        let expected = Mat4x4 {
            data: [
                Vec4D::new(17.0, 17.0, 17.0, 17.0),
                Vec4D::new(17.0, 17.0, 17.0, 17.0),
                Vec4D::new(17.0, 17.0, 17.0, 17.0),
                Vec4D::new(17.0, 17.0, 17.0, 17.0),
            ],
        };
        assert_eq!(sum, expected);

        let diff = m - n;
        let expected = Mat4x4 {
            data: [
                Vec4D::new(-15.0, -13.0, -11.0, -9.0),
                Vec4D::new(-7.0, -5.0, -3.0, -1.0),
                Vec4D::new(1.0, 3.0, 5.0, 7.0),
                Vec4D::new(9.0, 11.0, 13.0, 15.0),
            ],
        };
        assert_eq!(diff, expected);

        let prod = m * n;
        let expected = Mat4x4 {
            data: [
                Vec4D::new(80.0, 70.0, 60.0, 50.0),
                Vec4D::new(240.0, 214.0, 188.0, 162.0),
                Vec4D::new(400.0, 358.0, 316.0, 274.0),
                Vec4D::new(560.0, 502.0, 444.0, 386.0),
            ],
        };
        assert_eq!(prod, expected);

        let v = Vec4D::new(1.0, 2.0, 3.0, 4.0);
        let prod = m * v;
        let expected = Vec4D::new(30.0, 70.0, 110.0, 150.0);
        assert_eq!(prod, expected);
    }

    #[test]
    fn test_mat4x4_determinant() {
        let m = Mat4x4 {
            data: [
                Vec4D::new(1.0, 2.0, 3.0, 4.0),
                Vec4D::new(5.0, 6.0, 7.0, 8.0),
                Vec4D::new(9.0, 10.0, 11.0, 12.0),
                Vec4D::new(13.0, 14.0, 15.0, 16.0),
            ],
        };
        let det = m.determinant();
        assert_eq!(det, 0.0);
    }

    #[test]
    fn test_mat4x4_transpose() {
        let m = Mat4x4 {
            data: [
                Vec4D::new(1.0, 2.0, 3.0, 4.0),
                Vec4D::new(5.0, 6.0, 7.0, 8.0),
                Vec4D::new(9.0, 10.0, 11.0, 12.0),
                Vec4D::new(13.0, 14.0, 15.0, 16.0),
            ],
        };
        let transposed = m.transpose();
        let expected = Mat4x4 {
            data: [
                Vec4D::new(1.0, 5.0, 9.0, 13.0),
                Vec4D::new(2.0, 6.0, 10.0, 14.0),
                Vec4D::new(3.0, 7.0, 11.0, 15.0),
                Vec4D::new(4.0, 8.0, 12.0, 16.0),
            ],
        };
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_mat4x4_inverse() {
        let m = Mat4x4 {
            data: [
                Vec4D::new(0.0, 1.0, 1.0, 1.0),
                Vec4D::new(1.0, 0.0, 1.0, 1.0),
                Vec4D::new(1.0, 1.0, 0.0, 1.0),
                Vec4D::new(1.0, 1.0, 1.0, 0.0),
            ],
        };
        let inv = m.inverse().unwrap();
        let expected = (1.0 / 3.0)
            * Mat4x4 {
                data: [
                    Vec4D::new(-2.0, 1.0, 1.0, 1.0),
                    Vec4D::new(1.0, -2.0, 1.0, 1.0),
                    Vec4D::new(1.0, 1.0, -2.0, 1.0),
                    Vec4D::new(1.0, 1.0, 1.0, -2.0),
                ],
            };
        assert_eq!(inv, expected);
    }
}
