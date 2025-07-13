use pyo3::{pyclass, pymethods};

use crate::python_api::PyBBox;

#[pyclass(name = "Track")]
pub struct PyTrack {
    #[pyo3(get)]
    pub id: u32,
    pub bbox: PyBBox,
    #[pyo3(get)]
    pub class_id: u32,
}

#[pymethods]
impl PyTrack {
    #[getter]
    fn bbox(&self) -> PyBBox {
        self.bbox.clone()
    }
}
