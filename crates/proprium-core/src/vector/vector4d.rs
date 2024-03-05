use crate::{Real, Vec2D, Vec3D, Vector};
use core::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// A 4D vector.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec4D {
    x: Real,
    y: Real,
    z: Real,
    w: Real,
}

impl Vec4D {
    pub fn new(x: Real, y: Real, z: Real, w: Real) -> Self {
        Self { x, y, z, w }
    }

    pub fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    pub fn from_vec2d(v: &Vec2D, z: Real, w: Real) -> Self {
        Self {
            x: v.x(),
            y: v.y(),
            z,
            w,
        }
    }

    pub fn from_vec3d(v: &Vec3D, w: Real) -> Self {
        Self {
            x: v.x(),
            y: v.y(),
            z: v.z(),
            w,
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

    pub fn w(&self) -> Real {
        self.w
    }

    pub fn set(&mut self, x: Real, y: Real, z: Real, w: Real) -> () {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
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

    pub fn set_w(&mut self, w: Real) -> () {
        self.w = w;
    }

    pub fn homogenize(&mut self) -> () {
        if self.w == 0.0 {
            eprintln!("Cannot homogenize vector with w = 0.0");
            return;
        }
        self.x /= self.w;
        self.y /= self.w;
        self.z /= self.w;
        self.w = 1.0;
    }
}

impl fmt::Display for Vec4D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl Add for Vec4D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl Sub for Vec4D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl Mul<Real> for Vec4D {
    type Output = Self;

    fn mul(self, scalar: Real) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl Mul<Vec4D> for Real {
    type Output = Vec4D;

    fn mul(self, rhs: Vec4D) -> Self::Output {
        rhs.mul(self)
    }
}

impl Div<Real> for Vec4D {
    type Output = Self;

    fn div(self, scalar: Real) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            w: self.w / scalar,
        }
    }
}

impl Neg for Vec4D {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl Index<usize> for Vec4D {
    type Output = Real;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec4D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl Vector for Vec4D {
    fn dot(&self, other: &Self) -> Real {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    fn magnitude(&self) -> Real {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    fn normalize(&mut self) -> () {
        let mag = self.magnitude();
        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
        self.w /= mag;
    }

    fn normalized(&self) -> Self {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
            w: self.w / mag,
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
    fn test_vec4_ops() {
        let a = Vec4D::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4D::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a + b, Vec4D::new(6.0, 8.0, 10.0, 12.0));
        assert_eq!(a - b, Vec4D::new(-4.0, -4.0, -4.0, -4.0));
        assert_eq!(a * 2.0, Vec4D::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(a / 2.0, Vec4D::new(0.5, 1.0, 1.5, 2.0));
        assert_eq!(-a, Vec4D::new(-1.0, -2.0, -3.0, -4.0));
    }

    #[test]
    fn test_vec4_methods() {
        let a = Vec4D::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4D::new(-2.0, -4.0, 6.0, 8.0);
        assert_eq!(a.dot(&b), 40.0);
        assert_eq!(a.magnitude(), (30.0_f64).sqrt());
        assert_eq!(b.magnitude(), 2.0_f64 * (30.0_f64).sqrt());
        let mut c = Vec4D::new(1.0, 2.0, 3.0, 4.0);
        c.normalize();
        let mag = (30.0_f64).sqrt();
        assert_eq!(c, Vec4D::new(1.0 / mag, 2.0 / mag, 3.0 / mag, 4.0 / mag));
        assert!(b.angle(&a) - 1.28104463 < EPSILON);
        assert_eq!(a.project(&b), 1.0 / 3.0 * Vec4D::new(-2.0, -4.0, 6.0, 8.0));
        let reflect = a.reflect(&b);
        assert!((reflect.x - 3.0) < EPSILON);
        assert!((reflect.y - 6.0) < EPSILON);
        assert!((reflect.z - 3.0) < EPSILON);
        assert!((reflect.w - 4.0) < EPSILON);
    }
}
