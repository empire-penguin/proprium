use crate::{Real, Vec2D, Vec3D, Vec4D};

pub trait Matrix: Sized {
    fn determinant(&self) -> Real;
    fn transpose(&self) -> Self;
    fn inverse(&self) -> Option<Self>;

    fn get(&self, row: usize, col: usize) -> Real;
    fn get_row(&self, row: usize) -> Option<Vec<Real>>;
    fn get_col(&self, col: usize) -> Option<Vec<Real>>;
}
