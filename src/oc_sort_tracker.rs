use crate::{
    associate::{associate_detections_to_trackers, byte_associate, observation_centric_recovery},
    bbox::BBox,
    kalman_box_tracker::{KalmanBoxTracker, Track},
};
use itertools::{Either, Itertools};

/// A detection received from an object detector.
#[derive(Clone, Copy)]
pub struct Detection {
    /// The bounding box of the detection.
    pub bbox: BBox,
    /// The class id of the detection.
    pub class: u32,
    /// The confidence score of the detection.
    pub score: f64,
}

impl AsRef<Detection> for Detection {
    fn as_ref(&self) -> &Detection {
        self
    }
}

/// The OC-SORT tracker.
pub struct OCSort {
    /// Vec of object that are currently being tracked.
    trackers: Vec<KalmanBoxTracker>,
    /// The maximum number of updates a tracker can have without new associations to detections.
    max_age: u32,
    /// The minimum IoU score need for an association.
    iou_threshold: f64,
    /// The time lag used for speed direction calculations.
    delta_t: u32,
    /// Score threshold used to divide detections to high and low sets in BYTE association.
    score_threshold: f64,
    /// The minimum number of consecutive association a track needs to be returned.
    min_hit_streak: u32,
}

impl OCSort {
    /// Creates a new OCSort instance with the given configuration no initial tracked objects
    ///
    /// ## Args
    ///  - max_age: The maximum number of updates a tracker can have without new associations to detections.
    ///  - iou_threshold: The minimum IoU score needed for an association.
    ///  - delta_t: The time lag used for speed direction calculations.
    ///  - score_threshold: Used to divide detections to high and low sets in BYTE association.
    ///  - min_hit_streak: The minimum number of consecutive associations a track needs to be returned.
    pub fn new(
        max_age: u32,
        iou_threshold: f64,
        delta_t: u32,
        score_threshold: f64,
        min_hit_streak: u32,
    ) -> Self {
        Self {
            trackers: Vec::new(),
            max_age,
            iou_threshold,
            delta_t,
            score_threshold,
            min_hit_streak,
        }
    }

    /// Returns the currently tracked objects filtered by min_hit_streak.
    ///
    /// Does not update the state of the tracks.
    pub fn get_trackers(&self) -> Vec<Track> {
        self.trackers
            .iter()
            .filter(|tracker| {
                (tracker.time_since_update < 1) & (tracker.hit_streak >= self.min_hit_streak)
            })
            .map(|tracker| tracker.get_state())
            .collect()
    }

    /// Update the state of the tracked objects and associate them to the detections.
    ///
    /// Creates new tracks for the detections which are not associated and that have score equal or above
    /// to the score_threshold.
    ///
    /// Uses the OC-SORT algorithm with BYTE association.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.trackers.iter_mut().for_each(|tracker| {
            tracker.predict();
        });

        self.trackers
            .retain(|tracker| tracker.time_since_update <= self.max_age);

        let (high_score_indices, low_score_indices): (Vec<usize>, Vec<usize>) = detections
            .iter()
            .enumerate()
            .partition_map(|(i, detection)| {
                if detection.score >= self.score_threshold {
                    Either::Left(i)
                } else {
                    Either::Right(i)
                }
            });

        if self.trackers.is_empty() {
            for detection_index in high_score_indices {
                let detection = detections[detection_index];
                self.trackers.push(KalmanBoxTracker::new(
                    detection.bbox,
                    self.delta_t,
                    detection.class,
                ));
            }
            return self.get_trackers();
        }

        if detections.is_empty() {
            return self.get_trackers();
        }

        let unmatched_tracker_indices: Vec<usize> = (0..self.trackers.len()).into_iter().collect();
        let (matched_indices, unmatched_detection_indices, unmatched_tracker_indices) =
            associate_detections_to_trackers(
                &detections,
                &high_score_indices,
                &self.trackers,
                &unmatched_tracker_indices,
                self.iou_threshold,
            );

        let (byte_matched_indices, _, unmatched_tracker_indices) = byte_associate(
            &detections,
            &low_score_indices,
            &self.trackers,
            &unmatched_tracker_indices,
            self.iou_threshold,
        );

        let (ocr_matched_indices, unmatched_detection_indices, _) = observation_centric_recovery(
            &detections,
            &unmatched_detection_indices,
            &self.trackers,
            &unmatched_tracker_indices,
            self.iou_threshold,
        );

        for &(detection_index, tracker_index) in matched_indices
            .iter()
            .chain(byte_matched_indices.iter())
            .chain(ocr_matched_indices.iter())
        {
            self.trackers[tracker_index].update(detections[detection_index].bbox);
        }

        for detection_index in unmatched_detection_indices {
            let detection = detections[detection_index];
            self.trackers.push(KalmanBoxTracker::new(
                detection.bbox,
                detection.class,
                self.delta_t,
            ));
        }

        self.get_trackers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_reassociates_lost_object() {
        let mut oc_sort_tracker = OCSort::new(5, 0.3, 3, 0.5, 1);
        let detections = vec![Detection {
            bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
            class: 1,
            score: 0.7,
        }];

        oc_sort_tracker.update(&detections);
        let detections = vec![Detection {
            bbox: BBox::new(0.5, 0.0, 1.5, 1.0),
            class: 1,
            score: 0.6,
        }];
        oc_sort_tracker.update(&detections);

        oc_sort_tracker.update(&Vec::new());

        let detections = vec![Detection {
            bbox: BBox::new(1.5, 0.0, 2.5, 1.0),
            class: 1,
            score: 0.8,
        }];
        let tracks = oc_sort_tracker.update(&detections);

        assert_eq!(tracks.len(), 1);

        let tolerance = 0.1;
        let track = &tracks[0];
        assert_eq!(track.class, 1);
        assert!((track.bbox.x_1 - 1.5).abs() <= tolerance);
        assert!((track.bbox.x_2 - 2.5).abs() <= tolerance);
        assert!((track.bbox.y_1 - 0.0).abs() <= tolerance);
        assert!((track.bbox.y_2 - 1.0).abs() <= tolerance);
    }

    #[test]
    fn test_update_keeps_track_of_objects() {
        let motorcycle_bboxes = vec![
            BBox::new(187.0, 324.0, 303.0, 422.0),
            BBox::new(183.0, 321.0, 302.0, 426.0),
            BBox::new(180.0, 324.0, 303.0, 429.0),
            BBox::new(179.0, 324.0, 303.0, 433.0),
            BBox::new(168.0, 327.0, 305.0, 438.0),
        ];

        let person_bboxes = vec![
            BBox::new(213.0, 280.0, 266.0, 402.0),
            BBox::new(211.0, 278.0, 265.0, 403.0),
            BBox::new(211.0, 278.0, 269.0, 406.0),
            BBox::new(210.0, 276.0, 268.0, 405.0),
            BBox::new(206.0, 277.0, 269.0, 408.0),
        ];

        let mut oc_sort_tracker = OCSort::new(5, 0.3, 3, 0.5, 1);

        for i in 0..motorcycle_bboxes.len() {
            let detections = vec![
                Detection {
                    bbox: motorcycle_bboxes[i],
                    class: 3,
                    score: 0.9,
                },
                Detection {
                    bbox: person_bboxes[i],
                    class: 0,
                    score: 0.8,
                },
            ];
            let tracks = oc_sort_tracker.update(&detections);
            assert_eq!(tracks.len(), 2);
        }
    }
}
