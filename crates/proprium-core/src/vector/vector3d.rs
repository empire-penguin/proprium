use crate::{Real, Vec2D, Vector};
use core::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// A 3D vector.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec3D {
    x: Real,
    y: Real,
    z: Real,
}

impl Vec3D {
    pub fn new(x: Real, y: Real, z: Real) -> Self {
        Self { x, y, z }
    }

    pub fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn from_vec2d(v: Vec2D, z: Real) -> Self {
        Self {
            x: v.x(),
            y: v.y(),
            z,
        }
    }

    pub fn x(&self) -> Real {
        self.x
    }

    pub fn y(&self) -> Real {
        self.y
    }

    pub fn z(&self) -> Real {
        self.z
    }

    pub fn set(&mut self, x: Real, y: Real, z: Real) -> () {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    pub fn set_x(&mut self, x: Real) -> () {
        self.x = x;
    }

    pub fn set_y(&mut self, y: Real) -> () {
        self.y = y;
    }

    pub fn set_z(&mut self, z: Real) -> () {
        self.z = z;
    }

    /// Calculates the cross product of two vectors.
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl fmt::Display for Vec3D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Add for Vec3D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<Real> for Vec3D {
    type Output = Self;

    fn mul(self, scalar: Real) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<Vec3D> for Real {
    type Output = Vec3D;

    fn mul(self, rhs: Vec3D) -> Self::Output {
        rhs.mul(self)
    }
}

impl Div<Real> for Vec3D {
    type Output = Self;

    fn div(self, scalar: Real) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl Neg for Vec3D {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl Index<usize> for Vec3D {
    type Output = Real;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec3D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl Vector for Vec3D {
    fn dot(&self, other: &Self) -> Real {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn magnitude(&self) -> Real {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize(&mut self) -> () {
        let mag = self.magnitude();
        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
    }

    fn normalized(&self) -> Self {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    fn angle(&self, other: &Self) -> Real {
        (self.dot(other) / (self.magnitude() * other.magnitude())).acos()
    }

    fn project(&self, other: &Self) -> Self {
        let scalar = self.dot(other) / other.dot(other);
        *other * scalar
    }

    fn reflect(&self, other: &Self) -> Self {
        *self - 2.0 * self.project(other)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_vec3_ops() {
        let a = Vec3D::new(1.0, 2.0, 3.0);
        let b = Vec3D::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3D::new(5.0, 7.0, 9.0));
        assert_eq!(a - b, Vec3D::new(-3.0, -3.0, -3.0));
        assert_eq!(a * 2.0, Vec3D::new(2.0, 4.0, 6.0));
        assert_eq!(a / 2.0, Vec3D::new(0.5, 1.0, 1.5));
        assert_eq!(-a, Vec3D::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_vec3_methods() {
        let a = Vec3D::new(1.0, 2.0, 3.0);
        let b = Vec3D::new(-2.0, -4.0, 6.0);
        assert_eq!(a.dot(&b), 8.0);
        assert_eq!(a.cross(&b), Vec3D::new(24.0, -12.0, 0.0));
        assert_eq!(a.magnitude(), (14.0_f64).sqrt());
        assert_eq!(b.magnitude(), 2.0_f64 * (14.0_f64).sqrt());
        let mut c = Vec3D::new(1.0, 2.0, 3.0);
        c.normalize();
        let mag = (14.0_f64).sqrt();
        assert_eq!(c, Vec3D::new(1.0 / mag, 2.0 / mag, 3.0 / mag));
        assert!(b.angle(&a) - 1.28104463 < EPSILON);
        assert_eq!(a.project(&b), 2.0 / 7.0 * Vec3D::new(-1.0, -2.0, 3.0));
        let reflect = a.reflect(&b);
        assert!((reflect.x - 3.0) < EPSILON);
        assert!((reflect.y - 6.0) < EPSILON);
        assert!((reflect.z - 3.0) < EPSILON);
    }
}
