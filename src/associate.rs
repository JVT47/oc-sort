use std::{collections::HashSet, f64::consts::PI};

use crate::{bbox::BBox, kalman_box_tracker::KalmanBoxTracker, oc_sort_tracker::Detection};
use pathfinding::prelude::{Matrix, kuhn_munkres_min};

const IOU_MULTIPLIER: f64 = 10000.0;

pub fn associate_detections_to_trackers(
    detections: &[Detection],
    trackers: &[KalmanBoxTracker],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let detection_bboxes = detections
        .iter()
        .map(|detection| detection.bbox)
        .collect::<Vec<BBox>>();

    let tracker_bboxes = trackers
        .iter()
        .map(|tracker| tracker.get_bbox())
        .collect::<Vec<BBox>>();

    let iou_matrix = calc_iou_cost_matrix(&detection_bboxes, &tracker_bboxes);
    let mut cost_matrix = iou_matrix.clone();
    add_speed_cost_matrix(&detection_bboxes, &trackers, &mut cost_matrix);
    add_class_cost_matrix(detections, trackers, &mut cost_matrix);

    calculate_matching(
        detections,
        trackers,
        &cost_matrix,
        &iou_matrix,
        iou_threshold,
    )
}

pub fn observation_centric_recovery(
    unmatched_detections: &[&Detection],
    unmatched_trackers: &[&KalmanBoxTracker],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if unmatched_detections.is_empty() || unmatched_trackers.is_empty() {
        return (
            Vec::new(),
            (0..unmatched_detections.len()).into_iter().collect(),
            (0..unmatched_trackers.len()).into_iter().collect(),
        );
    }
    let detection_bboxes = unmatched_detections
        .iter()
        .map(|detection| detection.bbox)
        .collect::<Vec<BBox>>();
    let tracker_observations = unmatched_trackers
        .iter()
        .map(|tracker| *tracker.get_last_observation())
        .collect::<Vec<BBox>>();

    let iou_matrix = calc_iou_cost_matrix(&detection_bboxes, &tracker_observations);
    let mut cost_matrix = iou_matrix.clone();
    add_class_cost_matrix(&unmatched_detections, unmatched_trackers, &mut cost_matrix);

    calculate_matching(
        unmatched_detections,
        unmatched_trackers,
        &cost_matrix,
        &iou_matrix,
        iou_threshold,
    )
}
fn calculate_matching<D: AsRef<Detection>, K: AsRef<KalmanBoxTracker>>(
    detections: &[D],
    trackers: &[K],
    cost_matrix: &Matrix<i64>,
    iou_matrix: &Matrix<i64>,
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let transpose = cost_matrix.rows > cost_matrix.columns;
    let weights = if transpose {
        &cost_matrix.transposed()
    } else {
        cost_matrix
    };
    let assignment_vector = kuhn_munkres_min(weights).1;
    let assigned: HashSet<usize> = assignment_vector.iter().cloned().collect();

    let mut unmatched_detections = if transpose {
        (0..weights.columns)
            .filter(|col| !assigned.contains(col))
            .collect()
    } else {
        Vec::new()
    };
    let mut unmatched_trackers = if transpose {
        Vec::new()
    } else {
        (0..weights.columns)
            .filter(|col| !assigned.contains(col))
            .collect()
    };

    let mut matched = Vec::new();

    for (i, j) in assignment_vector.iter().enumerate() {
        let (detection_index, tracker_index) = if transpose { (*j, i) } else { (i, *j) };
        let detection = &detections[detection_index];
        let tracker = &trackers[tracker_index];

        let invalid_iou =
            -iou_matrix[(detection_index, tracker_index)] < (iou_threshold * IOU_MULTIPLIER) as i64;
        let invalid_class = detection.as_ref().class != tracker.as_ref().class;
        if invalid_iou || invalid_class {
            unmatched_detections.push(detection_index);
            unmatched_trackers.push(tracker_index);
            continue;
        }
        matched.push((detection_index, tracker_index));
    }

    (matched, unmatched_detections, unmatched_trackers)
}

fn calc_iou_cost_matrix(bboxes_1: &[BBox], bboxes_2: &[BBox]) -> Matrix<i64> {
    let rows = bboxes_1.len();
    let columns = bboxes_2.len();

    let mut matrix = Matrix::new(rows, columns, 0);

    for (i, bbox_1) in bboxes_1.iter().enumerate() {
        for (j, bbox_2) in bboxes_2.iter().enumerate() {
            matrix[(i, j)] = -(bbox_1.iou(bbox_2) * IOU_MULTIPLIER) as i64;
        }
    }

    matrix
}

fn add_class_cost_matrix<D: AsRef<Detection>, K: AsRef<KalmanBoxTracker>>(
    detections: &[D],
    trackers: &[K],
    cost_matrix: &mut Matrix<i64>,
) {
    for (i, detection) in detections.iter().enumerate() {
        for (j, tracker) in trackers.iter().enumerate() {
            let cost = if detection.as_ref().class == tracker.as_ref().class {
                0
            } else {
                (100.0 * IOU_MULTIPLIER) as i64
            };
            cost_matrix[(i, j)] += cost;
        }
    }
}

fn add_speed_cost_matrix(
    detection_bboxes: &[BBox],
    trackers: &[KalmanBoxTracker],
    cost_matrix: &mut Matrix<i64>,
) {
    for (i, bbox_1) in detection_bboxes.iter().enumerate() {
        for (j, tracker) in trackers.iter().enumerate() {
            let inertia = tracker.speed_direction;
            let bbox_2 = tracker.get_observation_dt_time_steps_away();
            let speed_direction = bbox_1.speed_direction(bbox_2);

            let diff_angle = inertia.dot(&speed_direction).acos();
            let diff_angle_cost = (diff_angle - PI) / PI;

            cost_matrix[(i, j)] += (diff_angle_cost * 0.2 * IOU_MULTIPLIER) as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_associate_detections_to_trackers_returns_correct_matching() {
        let detections = vec![
            Detection {
                bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
                class: 0,
            },
            Detection {
                bbox: BBox::new(2.0, 3.0, 4.0, 4.0),
                class: 0,
            },
        ];

        let trackers = vec![KalmanBoxTracker::new(BBox::new(0.5, 0.0, 1.5, 1.0), 3, 0)];

        let iou_threshold = 0.3;

        let (matched_indices, unmatched_detection_indices, unmatched_tracker_indices) =
            associate_detections_to_trackers(&detections, &trackers, iou_threshold);

        assert_eq!(matched_indices, vec![(0, 0)]);
        assert_eq!(unmatched_detection_indices, vec![1]);
        assert_eq!(unmatched_tracker_indices, Vec::<usize>::new());
    }
}
