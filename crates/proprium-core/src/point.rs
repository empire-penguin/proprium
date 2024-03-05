pub struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Default for Point3D {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}
