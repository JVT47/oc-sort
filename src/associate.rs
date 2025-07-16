use std::{collections::HashSet, f64::consts::PI};

use crate::{bbox::BBox, kalman_box_tracker::KalmanBoxTracker, oc_sort_tracker::Detection};
use pathfinding::prelude::{Matrix, kuhn_munkres_min};

// used to convert small float to some large integer since
// the weight matrix of the hungarian algorithm only
// accepts integers.
const IOU_MULTIPLIER: f64 = 10000.0;

/// Associates the given detections to the given trackers.
///
/// ## Args
///  - detections: Reference to all detections
///  - detection_indices: The indices of the detections available for association.
///  - trackers: Reference to all trackers.
///  - tracker_indices: The indices of the trackers available for association.
///  - iou_threshold: The minimum iou score needed for a valid association.
///
/// Takes into account iou scores, observation centric momentum
/// and class similarity.
pub fn associate_detections_to_trackers(
    detections: &[Detection],
    detection_indices: &[usize],
    trackers: &[KalmanBoxTracker],
    tracker_indices: &[usize],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let (detection_bboxes, tracker_bboxes) =
        get_bboxes(detections, detection_indices, trackers, tracker_indices);

    let iou_matrix = calc_iou_cost_matrix(&detection_bboxes, &tracker_bboxes);
    let mut cost_matrix = iou_matrix.clone();
    add_speed_cost_matrix(&detection_bboxes, &trackers, &mut cost_matrix);
    add_class_cost_matrix(
        detections,
        detection_indices,
        trackers,
        tracker_indices,
        &mut cost_matrix,
    );

    calculate_matching(
        detections,
        detection_indices,
        trackers,
        tracker_indices,
        &cost_matrix,
        &iou_matrix,
        iou_threshold,
    )
}

/// Runs BYTE association, i.e, associates the low score detections to the current
/// position of the trackers by only considering iou and class similarity.
///
/// ## Args
///  - detections: Reference to all detections.
///  - detection_indices: The indices of detections with a low score.
///  - trackers: Reference to all trackers.
///  - tracker_indices: The indices of trackers available for association.
///  - iou_threshold: The minimum iou score needed for a valid association.
pub fn byte_associate(
    detections: &[Detection],
    detection_indices: &[usize],
    trackers: &[KalmanBoxTracker],
    tracker_indices: &[usize],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if detection_indices.is_empty() || tracker_indices.is_empty() {
        return (
            Vec::new(),
            Vec::from(detection_indices),
            Vec::from(tracker_indices),
        );
    }
    let (detection_bboxes, tracker_bboxes) =
        get_bboxes(detections, detection_indices, trackers, tracker_indices);

    let iou_matrix = calc_iou_cost_matrix(&detection_bboxes, &tracker_bboxes);
    let mut cost_matrix = iou_matrix.clone();
    add_class_cost_matrix(
        detections,
        detection_indices,
        trackers,
        tracker_indices,
        &mut cost_matrix,
    );

    calculate_matching(
        detections,
        detection_indices,
        trackers,
        tracker_indices,
        &cost_matrix,
        &iou_matrix,
        iou_threshold,
    )
}

/// Runs Observation Centric Recovery (OCR) association, i.e, associates
/// detections to the last associations made by the trackers by only
/// considering iou and class similarity.
///
/// ## Args
///  - detections: Reference to all detections.
///  - detection_indices: The indices of detections available for association.
///  - trackers: Reference to all trackers.
///  - tracker_indices: The indices of trackers available for association.
///  - iou_threshold: The minimum iou score needed for a valid association.
pub fn observation_centric_recovery(
    detections: &[Detection],
    detection_indices: &[usize],
    trackers: &[KalmanBoxTracker],
    tracker_indices: &[usize],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if detection_indices.is_empty() || tracker_indices.is_empty() {
        return (
            Vec::new(),
            Vec::from(detection_indices),
            Vec::from(tracker_indices),
        );
    }
    let detection_bboxes: Vec<BBox> = detection_indices
        .iter()
        .map(|&detection_index| detections[detection_index].bbox)
        .collect();

    let tracker_observations: Vec<BBox> = tracker_indices
        .iter()
        .map(|&tracker_index| *trackers[tracker_index].get_last_observation())
        .collect();

    let iou_matrix = calc_iou_cost_matrix(&detection_bboxes, &tracker_observations);
    let mut cost_matrix = iou_matrix.clone();
    add_class_cost_matrix(
        detections,
        detection_indices,
        trackers,
        tracker_indices,
        &mut cost_matrix,
    );

    calculate_matching(
        detections,
        detection_indices,
        trackers,
        tracker_indices,
        &cost_matrix,
        &iou_matrix,
        iou_threshold,
    )
}

fn get_bboxes(
    detections: &[Detection],
    detection_indices: &[usize],
    trackers: &[KalmanBoxTracker],
    tracker_indices: &[usize],
) -> (Vec<BBox>, Vec<BBox>) {
    let detection_bboxes: Vec<BBox> = detection_indices
        .iter()
        .map(|&detection_index| detections[detection_index].bbox)
        .collect();

    let tracker_bboxes: Vec<BBox> = tracker_indices
        .iter()
        .map(|&tracker_index| trackers[tracker_index].get_bbox())
        .collect();

    (detection_bboxes, tracker_bboxes)
}

fn calculate_matching(
    detections: &[Detection],
    detection_indices: &[usize],
    trackers: &[KalmanBoxTracker],
    tracker_indices: &[usize],
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

    for (i, &j) in assignment_vector.iter().enumerate() {
        let (detection_indices_index, tracker_indices_index) =
            if transpose { (j, i) } else { (i, j) };

        let detection_index = detection_indices[detection_indices_index];
        let tracker_index = tracker_indices[tracker_indices_index];
        let detection = &detections[detection_index];
        let tracker = &trackers[tracker_index];

        let invalid_iou = -iou_matrix[(detection_indices_index, tracker_indices_index)]
            < (iou_threshold * IOU_MULTIPLIER) as i64;
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

fn add_class_cost_matrix(
    detections: &[Detection],
    detection_indices: &[usize],
    trackers: &[KalmanBoxTracker],
    tracker_indices: &[usize],
    cost_matrix: &mut Matrix<i64>,
) {
    for (i, &detection_index) in detection_indices.iter().enumerate() {
        for (j, &tracker_index) in tracker_indices.iter().enumerate() {
            let cost = if detections[detection_index].class == trackers[tracker_index].class {
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
                score: 0.7,
            },
            Detection {
                bbox: BBox::new(2.0, 3.0, 4.0, 4.0),
                class: 0,
                score: 0.8,
            },
        ];
        let detection_indices = vec![0, 1];

        let trackers = vec![KalmanBoxTracker::new(BBox::new(0.5, 0.0, 1.5, 1.0), 0, 3)];
        let tracker_indices = vec![0];

        let iou_threshold = 0.3;

        let (matched_indices, unmatched_detection_indices, unmatched_tracker_indices) =
            associate_detections_to_trackers(
                &detections,
                &detection_indices,
                &trackers,
                &tracker_indices,
                iou_threshold,
            );

        assert_eq!(matched_indices, vec![(0, 0)]);
        assert_eq!(unmatched_detection_indices, vec![1]);
        assert_eq!(unmatched_tracker_indices, Vec::<usize>::new());
    }
}
