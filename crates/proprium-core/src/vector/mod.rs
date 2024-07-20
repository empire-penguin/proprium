//! A module for 2D, 3D, and 4D vectors and their operations.

mod vector2d;
mod vector3d;
mod vector4d;

pub use vector2d::Vec2D;
pub use vector3d::Vec3D;
pub use vector4d::Vec4D;

use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

pub type Real = f64;

pub trait Vector:
    Sized
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Real, Output = Self>
    + Mul<Real, Output = Self>
    + Neg<Output = Self>
    + Index<usize, Output = Real>
    + IndexMut<usize>
{
    /// Calculates the dot product of two vectors.
    fn dot(&self, other: &Self) -> Real;

    /// Calculates the magnitude squared of a vector
    fn magnitude_sqrd(&self) -> Real;

    /// Calculates the magnitude of a vector.
    fn magnitude(&self) -> Real;

    /// Normalizes a vector in place.
    fn normalize(&mut self) -> ();

    /// Returns a normalized copy of the vector.
    fn normalized(&self) -> Self;

    /// Calculates the angle between two vectors.
    fn angle(&self, other: &Self) -> Real;

    /// Projects this vector onto another vector.
    fn project(&self, other: &Self) -> Self;

    /// Reflects a vector off a surface.
    fn reflect(&self, normal: &Self) -> Self;
}
