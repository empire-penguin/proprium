//! A module for 2D, 3D, and 4D vectors and their operations.

use core::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

pub type Real = f64;

pub trait Vector:
    Sized
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Real, Output = Self>
    + Mul<Real, Output = Self>
    + Neg<Output = Self>
{
    /// Calculates the dot product of two vectors.
    fn dot(&self, other: &Self) -> Real;

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

/// A 2D vector.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec2D {
    x: Real,
    y: Real,
}

impl Vec2D {
    pub fn new(x: Real, y: Real) -> Self {
        Self { x, y }
    }

    pub fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn x(&self) -> Real {
        self.x
    }

    pub fn y(&self) -> Real {
        self.y
    }

    pub fn set(&mut self, x: Real, y: Real) -> () {
        self.x = x;
        self.y = y;
    }

    pub fn set_x(&mut self, x: Real) -> () {
        self.x = x;
    }

    pub fn set_y(&mut self, y: Real) -> () {
        self.y = y;
    }
}

impl fmt::Display for Vec2D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Add for Vec2D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Vec2D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<Real> for Vec2D {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vec2D> for Real {
    type Output = Vec2D;

    fn mul(self, rhs: Vec2D) -> Self::Output {
        rhs.mul(self)
    }
}

impl Div<Real> for Vec2D {
    type Output = Self;

    fn div(self, scalar: Real) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }
}

impl Neg for Vec2D {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Index<usize> for Vec2D {
    type Output = Real;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl Vector for Vec2D {
    fn dot(&self, other: &Self) -> Real {
        self.x * other.x + self.y * other.y
    }

    fn magnitude(&self) -> Real {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalize(&mut self) -> () {
        let mag = self.magnitude();
        self.x /= mag;
        self.y /= mag;
    }

    fn normalized(&self) -> Self {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
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
        Self { x: v.x, y: v.y, z }
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
            x: v.x,
            y: v.y,
            z,
            w,
        }
    }

    pub fn from_vec3d(v: &Vec3D, w: Real) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
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
    fn test_vec2_ops() {
        let a = Vec2D::new(1.0, 2.0);
        let b = Vec2D::new(3.0, 4.0);
        assert_eq!(a + b, Vec2D::new(4.0, 6.0));
        assert_eq!(a - b, Vec2D::new(-2.0, -2.0));
        assert_eq!(a * 2.0, Vec2D::new(2.0, 4.0));
        assert_eq!(a / 2.0, Vec2D::new(0.5, 1.0));
        assert_eq!(-a, Vec2D::new(-1.0, -2.0));
    }

    #[test]
    fn test_vec2_methods() {
        let a = Vec2D::new(1.0, 2.0);
        let b = Vec2D::new(3.0, 4.0);
        assert_eq!(a.dot(&b), 11.0);
        assert_eq!(a.magnitude(), (5.0_f64).sqrt());
        let mut c = Vec2D::new(3.0, 4.0);
        c.normalize();
        assert_eq!(c, Vec2D::new(3.0 / 5.0, 4.0 / 5.0));
        assert_eq!(b.angle(&c), 0.0);
        assert_eq!(a.project(&b), Vec2D::new(1.32, 1.76));
        let reflect = a.reflect(&b);
        assert!((reflect.x - -1.64) < EPSILON);
        assert!((reflect.y - -1.52) < EPSILON);
    }

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
