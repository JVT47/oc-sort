use pyo3::{pyclass, pymethods};

use crate::{Detection, python_api::PyBBox};

#[pyclass(name = "Detection")]
pub struct PyDetection {
    pub inner: Detection,
}

#[pymethods]
impl PyDetection {
    #[new]
    pub fn new(bbox: &PyBBox, class_id: u32) -> Self {
        Self {
            inner: Detection {
                bbox: bbox.inner,
                class: class_id,
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
}
