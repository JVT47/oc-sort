mod associate;
mod bbox;
mod kalman_box_tracker;
mod oc_sort_tracker;
mod python_api;
pub use bbox::BBox;
pub use oc_sort_tracker::{Detection, OCSort};
use pyo3::{
    Bound, PyResult, pymodule,
    types::{PyModule, PyModuleMethods},
};

use crate::python_api::{PyBBox, PyDetection, PyOCSort};

#[pymodule]
fn oc_sort(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBBox>()?;
    m.add_class::<PyDetection>()?;
    m.add_class::<PyOCSort>()?;

    Ok(())
}
