use pyo3::{pyclass, pymethods};

use crate::{Detection, python_api::PyBBox};

#[pyclass(name = "Detection")]
pub struct PyDetection {
    pub inner: Detection,
}

#[pymethods]
impl PyDetection {
    #[new]
    pub fn new(bbox: &PyBBox, class_id: u32, score: f64) -> Self {
        Self {
            inner: Detection {
                bbox: bbox.inner,
                class: class_id,
                score: score,
            },
        }
    }

    #[getter]
    fn bbox(&self) -> PyBBox {
        PyBBox {
            inner: self.inner.bbox,
        }
    }

    #[getter]
    fn class_id(&self) -> u32 {
        self.inner.class
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }
}
