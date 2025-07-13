use pyo3::{pyclass, pymethods};

use crate::BBox;

#[pyclass(name = "BBox")]
#[derive(Clone)]
pub struct PyBBox {
    pub inner: BBox,
}

#[pymethods]
impl PyBBox {
    #[new]
    pub fn new(x_1: f64, y_1: f64, x_2: f64, y_2: f64) -> Self {
        Self {
            inner: BBox::new(x_1, y_1, x_2, y_2),
        }
    }

    #[getter]
    fn x_1(&self) -> f64 {
        self.inner.x_1
    }

    #[getter]
    fn y_1(&self) -> f64 {
        self.inner.y_1
    }

    #[getter]
    fn x_2(&self) -> f64 {
        self.inner.x_2
    }

    #[getter]
    fn y_2(&self) -> f64 {
        self.inner.y_2
    }

    fn __repr__(&self) -> String {
        format!(
            "BBox(x_1={}, y_1={}, x_2={}, y_2={}",
            self.x_1(),
            self.y_1(),
            self.x_2(),
            self.y_2()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
