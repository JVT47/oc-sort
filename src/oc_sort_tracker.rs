use crate::{
    associate::{associate_detections_to_trackers, observation_centric_recovery},
    bbox::BBox,
    kalman_box_tracker::{KalmanBoxTracker, Track},
};

#[derive(Clone, Copy)]
pub struct Detection {
    pub bbox: BBox,
    pub class: u32,
}

impl AsRef<Detection> for Detection {
    fn as_ref(&self) -> &Detection {
        self
    }
}

pub struct OCSort {
    trackers: Vec<KalmanBoxTracker>,
    max_age: u32,
    iou_threshold: f64,
    delta_t: u32,
}

impl OCSort {
    pub fn new(max_age: u32, iou_threshold: f64, delta_t: u32) -> Self {
        Self {
            trackers: Vec::new(),
            max_age,
            iou_threshold,
            delta_t,
        }
    }

    pub fn get_trackers(&self) -> Vec<Track> {
        self.trackers
            .iter()
            .map(|tracker| tracker.get_state())
            .collect()
    }

    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        if self.trackers.is_empty() {
            detections.iter().for_each(|detection| {
                self.trackers.push(KalmanBoxTracker::new(
                    detection.bbox,
                    self.delta_t,
                    detection.class,
                ))
            });
            return self.get_trackers();
        }

        self.trackers.iter_mut().for_each(|tracker| {
            tracker.predict();
        });

        if detections.is_empty() {
            return self.get_trackers();
        }

        let (matched_indices, unmatched_detection_indices, unmatched_tracker_indices) =
            associate_detections_to_trackers(&detections, &self.trackers, self.iou_threshold);

        let unmatched_detections = unmatched_detection_indices
            .iter()
            .map(|i| &detections[*i])
            .collect::<Vec<&Detection>>();

        let unmatched_trackers = unmatched_tracker_indices
            .iter()
            .map(|i| &self.trackers[*i])
            .collect::<Vec<&KalmanBoxTracker>>();

        let (ocr_matched_indices, unmatched_detection_indices, _) = observation_centric_recovery(
            &unmatched_detections,
            &unmatched_trackers,
            self.iou_threshold,
        );

        for (i, j) in matched_indices.iter().chain(ocr_matched_indices.iter()) {
            self.trackers[*j].update(detections[*i].bbox);
        }
        for i in unmatched_detection_indices.iter() {
            let detection = unmatched_detections[*i];
            self.trackers.push(KalmanBoxTracker::new(
                detection.bbox,
                self.delta_t,
                detection.class,
            ));
        }

        self.trackers
            .retain(|tracker| tracker.time_since_update <= self.max_age);

        self.get_trackers()
    }
}
