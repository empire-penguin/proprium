use crate::{Real, Vector};
use core::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

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
}
