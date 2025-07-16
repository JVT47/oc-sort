use crate::{
    associate::{associate_detections_to_trackers, byte_associate, observation_centric_recovery},
    bbox::BBox,
    kalman_box_tracker::{KalmanBoxTracker, Track},
};
use itertools::{Either, Itertools};

#[derive(Clone, Copy)]
pub struct Detection {
    pub bbox: BBox,
    pub class: u32,
    pub score: f64,
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
    score_threshold: f64,
    min_hit_streak: u32,
}

impl OCSort {
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

    pub fn get_trackers(&self) -> Vec<Track> {
        self.trackers
            .iter()
            .filter(|tracker| {
                (tracker.time_since_update < 1) & (tracker.hit_streak >= self.min_hit_streak)
            })
            .map(|tracker| tracker.get_state())
            .collect()
    }

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
                self.delta_t,
                detection.class,
            ));
        }

        self.get_trackers()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_update_reassociates_lost_object() {
//         let mut oc_sort_tracker = OCSort::new(5, 0.3, 3);
//         let detections = vec![Detection {
//             bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
//             class: 1,
//             score: 0.7,
//         }];

//         oc_sort_tracker.update(&detections);
//         let detections = vec![Detection {
//             bbox: BBox::new(0.5, 0.0, 1.5, 1.0),
//             class: 1,
//             score: 0.6,
//         }];
//         oc_sort_tracker.update(&detections);

//         oc_sort_tracker.update(&Vec::new());

//         let detections = vec![Detection {
//             bbox: BBox::new(1.5, 0.0, 2.5, 1.0),
//             class: 1,
//             score: 0.8,
//         }];
//         let tracks = oc_sort_tracker.update(&detections);

//         assert_eq!(tracks.len(), 1);

//         let tolerance = 0.1;
//         let track = &tracks[0];
//         assert_eq!(track.id, 0);
//         assert_eq!(track.class, 1);
//         assert!((track.bbox.x_1 - 1.5).abs() <= tolerance);
//     }

//     #[test]
//     fn test_update_keeps_track_of_objects() {
//         let motorcycle_bboxes = vec![
//             BBox::new(187.0, 324.0, 303.0, 422.0),
//             BBox::new(183.0, 321.0, 302.0, 426.0),
//             BBox::new(180.0, 324.0, 303.0, 429.0),
//             BBox::new(179.0, 324.0, 303.0, 433.0),
//             BBox::new(168.0, 327.0, 305.0, 438.0),
//         ];

//         let person_bboxes = vec![
//             BBox::new(213.0, 280.0, 266.0, 402.0),
//             BBox::new(211.0, 278.0, 265.0, 403.0),
//             BBox::new(211.0, 278.0, 269.0, 406.0),
//             BBox::new(210.0, 276.0, 268.0, 405.0),
//             BBox::new(206.0, 277.0, 269.0, 408.0),
//         ];

//         let mut oc_sort_tracker = OCSort::new(5, 0.3, 3);

//         for i in 0..motorcycle_bboxes.len() {
//             let detections = vec![
//                 Detection {
//                     bbox: motorcycle_bboxes[i],
//                     class: 3,
//                     score: 0.9,
//                 },
//                 Detection {
//                     bbox: person_bboxes[i],
//                     class: 0,
//                     score: 0.8,
//                 },
//             ];
//             let tracks = oc_sort_tracker.update(&detections);
//             assert_eq!(tracks.len(), 2);
//         }
//     }
// }
