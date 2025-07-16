use std::{
    collections::VecDeque,
    sync::atomic::{AtomicU32, Ordering},
};

use kfilter::{
    Kalman1M, KalmanFilter, KalmanPredict, measurement::LinearMeasurement,
    system::LinearNoInputSystem,
};
use nalgebra::{SMatrix, SVector};

use crate::bbox::BBox;

struct Observation {
    time_step: u32,
    bbox: BBox,
}

/// Represents a tracked object.
#[derive(Debug)]
pub struct Track {
    /// Unique id of the object.
    pub id: u32,
    /// The bounding box of the object.
    pub bbox: BBox,
    /// The class id of the object.
    pub class: u32,
}

/// Struct that keeps track of an object with the use of a Kalman Filter.
pub struct KalmanBoxTracker {
    /// The age of the tracked object in time steps.
    age: u32,
    /// The class id of the object.
    pub class: u32,
    /// The time lag used for speed direction calculations.
    delta_t: u32,
    /// The number of consecutive associations.
    pub hit_streak: u32,
    /// The id of the tracker.
    id: u32,
    /// The Kalman Filter used to track the object.
    kalman_filter:
        Kalman1M<f64, 7, 0, 4, LinearNoInputSystem<f64, 7>, LinearMeasurement<f64, 7, 4>>,
    /// The previous associations made.
    prev_observations: VecDeque<Observation>,
    /// The direction the object is going to.
    pub speed_direction: SVector<f64, 2>,
    /// Time since last association.
    pub time_since_update: u32,
}

static ID_COUNTER: AtomicU32 = AtomicU32::new(0);

impl KalmanBoxTracker {
    /// Creates a new tracker for a given bounding box.
    ///
    /// ## Args:
    ///  - bbox: The bounding box of the object.
    ///  - class: The class id of the object.
    ///  - delta_t: The time lag used for speed direction calculations.
    #[allow(non_snake_case)]
    pub fn new(bbox: BBox, class: u32, delta_t: u32) -> Self {
        let mut F = SMatrix::<f64, 7, 7>::identity();
        F[(0, 4)] = 1.0;
        F[(1, 5)] = 1.0;
        F[(2, 6)] = 1.0;
        let Q_diag = SVector::<f64, 7>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]);
        let Q = SMatrix::<f64, 7, 7>::from_diagonal(&Q_diag);
        let mut x_initial = SVector::<f64, 7>::zeros();
        x_initial
            .fixed_rows_mut::<4>(0)
            .copy_from(&bbox.to_observation_vector());
        let system = LinearNoInputSystem::new(F, Q, x_initial);

        let P_diag =
            SVector::<f64, 7>::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0]);
        let P = SMatrix::<f64, 7, 7>::from_diagonal(&P_diag);

        let H = SMatrix::<f64, 4, 7>::identity();
        let R_diag = SVector::<f64, 4>::new(1.0, 1.0, 10.0, 10.0);
        let R = SMatrix::from_diagonal(&R_diag);
        let measurement = LinearMeasurement::new(H, R, bbox.to_observation_vector());

        let kalman_filter = Kalman1M::new_custom(system, P, measurement);

        let id = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let age: u32 = 0;

        Self {
            kalman_filter,
            id,
            prev_observations: VecDeque::from([Observation {
                time_step: age,
                bbox,
            }]),
            age,
            hit_streak: 1,
            delta_t,
            speed_direction: SVector::<f64, 2>::zeros(),
            class,
            time_since_update: 0,
        }
    }

    /// Returns the bounding box of the last association made to a detection.
    pub fn get_last_observation(&self) -> &BBox {
        self.prev_observations.back().map(|obs| &obs.bbox).unwrap()
    }

    /// Returns the observation bounding box of the tracker that is closest to delta_t
    /// time steps away.
    pub fn get_observation_dt_time_steps_away(&self) -> &BBox {
        self.prev_observations
            .iter()
            .min_by_key(|obs| {
                obs.time_step
                    .abs_diff(self.age.saturating_sub(self.delta_t))
            })
            .map(|obs| &obs.bbox)
            .unwrap()
    }

    /// Returns the tracker's current bounding box.
    pub fn get_bbox(&self) -> BBox {
        BBox::from_state_vector(*self.kalman_filter.state())
    }

    /// Returns the Track representation of the currently tracked object.
    pub fn get_state(&self) -> Track {
        let bbox = BBox::from_state_vector(*self.kalman_filter.state());
        Track {
            id: self.id,
            bbox,
            class: self.class,
        }
    }

    /// Updates the state estimation of the tracked object with the bounding box from a detection.
    pub fn update(&mut self, bbox: BBox) {
        self.update_speed_direction(&bbox);
        self.update_kalman_filter(&bbox.to_observation_vector());
        self.add_bbox_to_observations(bbox);
        self.time_since_update = 0;
        self.hit_streak += 1;
    }

    /// Predicts the next state of the object. Returns the predicted bounding box.
    pub fn predict(&mut self) -> BBox {
        self.age += 1;
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        let state_vector = self.kalman_filter.predict();

        BBox::from_state_vector(*state_vector)
    }

    fn update_speed_direction(&mut self, bbox: &BBox) {
        let prev_obs = self.get_observation_dt_time_steps_away();
        self.speed_direction = bbox.speed_direction(&prev_obs);
    }

    fn update_kalman_filter(&mut self, z: &SVector<f64, 4>) {
        let last_observation = self.prev_observations.back().unwrap();
        let steps_between = self.age - last_observation.time_step;
        for t in 1..=steps_between {
            let z_interpolated = (steps_between - t) as f64 / steps_between as f64
                * last_observation.bbox.to_observation_vector()
                + t as f64 / steps_between as f64 * z;
            self.kalman_filter.update(z_interpolated);
            if t < steps_between {
                self.kalman_filter.predict();
            }
        }
    }

    fn add_bbox_to_observations(&mut self, bbox: BBox) {
        if self.prev_observations.len() >= self.delta_t as usize {
            self.prev_observations.pop_front();
        }
        self.prev_observations.push_back(Observation {
            time_step: self.age,
            bbox,
        });
    }
}

impl AsRef<KalmanBoxTracker> for KalmanBoxTracker {
    fn as_ref(&self) -> &KalmanBoxTracker {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_succeeds() {
        let bbox = BBox::new(1.0, 1.0, 2.0, 2.0);
        KalmanBoxTracker::new(bbox, 3, 0);
    }

    #[test]
    fn test_predict_advances_bbox() {
        let bbox_1 = BBox::new(0.0, 0.0, 1.0, 1.0);
        let bbox_2 = BBox::new(0.5, 0.0, 1.5, 1.0);

        let mut tracker = KalmanBoxTracker::new(bbox_1, 1, 1);
        tracker.predict();
        tracker.update(bbox_2);

        let bbox_3 = tracker.predict();
        let tolerance = 0.01;

        assert!((bbox_3.x_1 - 1.0).abs() < tolerance);
        assert!((bbox_3.y_1 - 0.0).abs() < tolerance);
        assert!((bbox_3.x_2 - 2.0).abs() < tolerance);
        assert!((bbox_3.y_2 - 1.0).abs() < tolerance);
    }
}
