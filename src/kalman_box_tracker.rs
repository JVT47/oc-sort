use std::{
    collections::VecDeque,
    sync::atomic::{AtomicU32, Ordering},
};

use kfilter::{
    Kalman1M, KalmanPredict, measurement::LinearMeasurement, system::LinearNoInputSystem,
};
use nalgebra::{SMatrix, SVector};

struct Observation {
    time_step: u32,
    z: SVector<f64, 4>,
}

pub struct KalmanBoxTracker {
    kalman_filter:
        Kalman1M<f64, 7, 0, 4, LinearNoInputSystem<f64, 7>, LinearMeasurement<f64, 7, 4>>,
    last_observation: SVector<f64, 4>,
    id: u32,
    age: u32,
    prev_observations: VecDeque<Observation>,
    delta_t: u32,
    speed_direction: SVector<f64, 2>,
}

static ID_COUNTER: AtomicU32 = AtomicU32::new(0);

impl KalmanBoxTracker {
    pub fn new(z: SVector<f64, 4>, delta_t: u32) -> Self {
        let mut F = SMatrix::<f64, 7, 7>::identity();
        F[(0, 4)] = 1.0;
        F[(1, 5)] = 1.0;
        F[(2, 6)] = 1.0;
        let Q_diag = SVector::<f64, 7>::from_vec(vec![1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]);
        let Q = SMatrix::<f64, 7, 7>::from_diagonal(&Q_diag);
        let mut x_initial = SVector::<f64, 7>::zeros();
        x_initial.fixed_rows_mut::<4>(0).copy_from(&z);
        let system = LinearNoInputSystem::new(F, Q, x_initial);

        let P_diag =
            SVector::<f64, 7>::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0]);
        let P = SMatrix::<f64, 7, 7>::from_diagonal(&P_diag);

        let H = SMatrix::<f64, 4, 7>::identity();
        let R_diag = SVector::<f64, 4>::new(1.0, 1.0, 10.0, 10.0);
        let R = SMatrix::from_diagonal(&R_diag);
        let measurement = LinearMeasurement::new(H, R, z);

        let kalman_filter = Kalman1M::new_custom(system, P, measurement);

        let id = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let age: u32 = 0;

        Self {
            kalman_filter,
            last_observation: z,
            id,
            prev_observations: VecDeque::from([Observation { time_step: age, z }]),
            age,
            delta_t,
            speed_direction: SVector::<f64, 2>::zeros(),
        }
    }

    pub fn get_observation_dt_time_steps_away(&self) -> SVector<f64, 4> {
        self.prev_observations
            .iter()
            .min_by_key(|obs| obs.time_step.abs_diff(self.age - self.delta_t))
            .map(|obs| obs.z.clone())
            .unwrap()
    }

    pub fn update(&mut self, z: SVector<f64, 4>) {
        self.update_speed_direction(&z);
        self.update_kalman_filter(&z);
        self.add_observation_to_observations(z);
    }

    pub fn predict(&mut self) -> SVector<f64, 4> {
        self.age += 1;
        self.kalman_filter
            .predict()
            .fixed_rows::<4>(0)
            .clone_owned()
    }

    fn update_speed_direction(&mut self, z: &SVector<f64, 4>) {
        let prev_obs = self.get_observation_dt_time_steps_away();
        self.speed_direction = Self::calc_speed_direction(z, &prev_obs)
    }

    fn update_kalman_filter(&mut self, z: &SVector<f64, 4>) {
        let last_observation = self.prev_observations.back().unwrap();
        let steps_between = self.age - last_observation.time_step;
        for t in 1..=steps_between {
            let z_interpolated = (steps_between - t) as f64 / steps_between as f64
                * last_observation.z
                + t as f64 / steps_between as f64 * z;
            self.kalman_filter.update(z_interpolated);
        }
    }

    fn add_observation_to_observations(&mut self, z: SVector<f64, 4>) {
        if self.prev_observations.len() >= self.delta_t as usize {
            self.prev_observations.pop_front();
        }
        self.prev_observations.push_back(Observation {
            time_step: self.age,
            z,
        });
    }

    fn calc_speed_direction(z1: &SVector<f64, 4>, z2: &SVector<f64, 4>) -> SVector<f64, 2> {
        let diff = z2.fixed_rows::<2>(0) - z1.fixed_rows::<2>(0);
        let norm = diff.norm();

        if norm != 0.0 {
            return diff / norm;
        }
        SVector::<f64, 2>::zeros()
    }
}
