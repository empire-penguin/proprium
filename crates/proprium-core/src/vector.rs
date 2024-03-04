use std::ops::{Add, Div, Mul, Neg, Sub};

type Real = f64;

trait Vector:
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

    /// Calculates the angle between two vectors.
    fn angle(&self, other: &Self) -> Real;

    /// Projects this vector onto another vector.
    fn project(&self, other: &Self) -> Self;

    /// Reflects a vector off a surface.
    fn reflect(&self, normal: &Self) -> Self;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec2D {
    x: Real,
    y: Real,
}

impl Vec2D {
    pub fn new(x: Real, y: Real) -> Self {
        Self { x, y }
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

    /// Calculates the cross product of two vectors.
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
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
}
