use pyo3::{PyRef, pyclass, pymethods};

use crate::{
    Detection, OCSort,
    python_api::{PyBBox, PyDetection, PyTrack},
};

#[pyclass(name = "OCSort")]
pub struct PyOCSort {
    inner: OCSort,
}

#[pymethods]
impl PyOCSort {
    #[new]
    pub fn new(max_age: u32, iou_threshold: f64, delta_t: u32, score_threshold: f64) -> PyOCSort {
        Self {
            inner: OCSort::new(max_age, iou_threshold, delta_t, score_threshold),
        }
    }

    pub fn get_trackers(&self) -> Vec<PyTrack> {
        self.inner
            .get_trackers()
            .iter()
            .map(|track| PyTrack {
                id: track.id,
                bbox: PyBBox { inner: track.bbox },
                class_id: track.class,
            })
            .collect()
    }

    pub fn update(&mut self, detections: Vec<PyRef<PyDetection>>) -> Vec<PyTrack> {
        let inner_detections = detections
            .iter()
            .map(|detection| detection.inner)
            .collect::<Vec<Detection>>();
        let tracks = self.inner.update(&inner_detections);

        tracks
            .iter()
            .map(|track| PyTrack {
                id: track.id,
                bbox: PyBBox { inner: track.bbox },
                class_id: track.class,
            })
            .collect()
    }
}
