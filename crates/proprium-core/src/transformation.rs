use crate::{Mat4x4, Matrix};
use crate::{Real, Vec3D, Vector};

pub struct Transformation {
    pub mat: Mat4x4,
}

impl Transformation {
    pub fn new(mat: Mat4x4) -> Self {
        Self { mat }
    }

    pub fn identity() -> Self {
        Self {
            mat: Mat4x4::identity(),
        }
    }

    pub fn translate(v: Vec3D) -> Self {
        let mut mat = Mat4x4::identity();
        mat[0][3] = v.x();
        mat[1][3] = v.y();
        mat[2][3] = v.z();
        Self { mat }
    }

    pub fn scale(v: Vec3D) -> Self {
        let mut mat = Mat4x4::identity();
        mat[0][0] = v.x();
        mat[1][1] = v.y();
        mat[2][2] = v.z();
        Self { mat }
    }

    pub fn rotate(axis: Vec3D, angle: Real) -> Self {
        let mut mat = Mat4x4::identity();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        let axis = axis.normalized();
        mat[0][0] = t * axis.x() * axis.x() + c;
        mat[0][1] = t * axis.x() * axis.y() - s * axis.z();
        mat[0][2] = t * axis.x() * axis.z() + s * axis.y();
        mat[1][0] = t * axis.x() * axis.y() + s * axis.z();
        mat[1][1] = t * axis.y() * axis.y() + c;
        mat[1][2] = t * axis.y() * axis.z() - s * axis.x();
        mat[2][0] = t * axis.x() * axis.z() - s * axis.y();
        mat[2][1] = t * axis.y() * axis.z() + s * axis.x();
        mat[2][2] = t * axis.z() * axis.z() + c;
        Self { mat }
    }

    pub fn look_at(eye: Vec3D, center: Vec3D, up: Vec3D) -> Self {
        let f = (center - eye).normalized();
        let s = f.cross(&up).normalized();
        let u = s.cross(&f).normalized();
        let mut mat = Mat4x4::identity();
        mat[0][0] = s.x();
        mat[0][1] = s.y();
        mat[0][2] = s.z();
        mat[1][0] = u.x();
        mat[1][1] = u.y();
        mat[1][2] = u.z();
        mat[2][0] = -f.x();
        mat[2][1] = -f.y();
        mat[2][2] = -f.z();
        mat[0][3] = -s.dot(&eye);
        mat[1][3] = -u.dot(&eye);
        mat[2][3] = f.dot(&eye);
        Self { mat }
    }

    pub fn perspective(fov: Real, aspect: Real, near: Real, far: Real) -> Self {
        let f = 1.0 / (fov / 2.0).tan();
        let mut mat = Mat4x4::identity();
        mat[0][0] = f / aspect;
        mat[1][1] = f;
        mat[2][2] = (far + near) / (near - far);
        mat[2][3] = (2.0 * far * near) / (near - far);
        mat[3][2] = -1.0;
        mat[3][3] = 0.0;
        Self { mat }
    }

    pub fn orthographic(
        left: Real,
        right: Real,
        bottom: Real,
        top: Real,
        near: Real,
        far: Real,
    ) -> Self {
        let mut mat = Mat4x4::identity();
        mat[0][0] = 2.0 / (right - left);
        mat[1][1] = 2.0 / (top - bottom);
        mat[2][2] = -2.0 / (far - near);
        mat[0][3] = -(right + left) / (right - left);
        mat[1][3] = -(top + bottom) / (top - bottom);
        mat[2][3] = -(far + near) / (far - near);
        Self { mat }
    }

    pub fn inverse(&self) -> Self {
        match self.mat.inverse() {
            Ok(mat) => Self { mat },
            Err(e) => {
                eprintln!("{}, Returning identity matrix instead of inverse.", e);
                Self::identity()
            }
        }
    }
}
