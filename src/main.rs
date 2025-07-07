use kfilter::{Kalman1M, KalmanPredict};
use nalgebra::{SMatrix, SVector};

fn main() {
    let mut f: SMatrix<f64, 7, 7> = SMatrix::identity();
    f[(0, 4)] = 1.0;
    f[(1, 5)] = 1.0;
    f[(2, 6)] = 1.0;
    let H: SMatrix<f64, 4, 7> = SMatrix::identity();
    let Q_diag = SVector::<f64, 7>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]);
    let Q = SMatrix::from_diagonal(&Q_diag);
    let R_diag = SVector::<f64, 4>::new(1.0, 1.0, 10.0, 10.0);
    let R= SMatrix::from_diagonal(&R_diag);
    let x = SVector::<f64, 7>::from_vec(vec![1.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0]);

    let mut k = Kalman1M::new(f, Q, H, R, x);

    for i in 0..10 {
        let x_hat = k.predict();
        println!("i: {}, prediction: {:?}", i, x_hat);
        let new_measurement = SVector::<f64, 4>::new(1.0, i as f64, 4.0, 1.0);
        let x_update = k.update(new_measurement);
        println!("i: {}, update: {:?}", i, x_update);


    }
}
