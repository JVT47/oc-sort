use std::f64::EPSILON;

use nalgebra::SVector;

#[derive(Clone, Copy, Default, Debug)]
pub struct BBox {
    pub x_1: f64,
    pub y_1: f64,
    pub x_2: f64,
    pub y_2: f64,
}

impl BBox {
    pub fn new(x_1: f64, y_1: f64, x_2: f64, y_2: f64) -> Self {
        BBox { x_1, y_1, x_2, y_2 }
    }

    pub fn from_state_vector(state_vector: SVector<f64, 7>) -> Self {
        let w = (state_vector[2] * state_vector[3]).sqrt();
        if w == 0.0 {
            return BBox {
                x_1: 0.0,
                y_1: 0.0,
                x_2: 0.0,
                y_2: 0.0,
            };
        }
        let h = state_vector[2] / w;

        Self {
            x_1: state_vector[0] - w / 2.0,
            y_1: state_vector[1] - h / 2.0,
            x_2: state_vector[0] + w / 2.0,
            y_2: state_vector[1] + h / 2.0,
        }
    }

    pub fn to_observation_vector(&self) -> SVector<f64, 4> {
        let w = (self.x_2 - self.x_1).max(0.0);
        let h = (self.y_2 - self.y_1).max(0.0);

        let cx = self.x_1 + w / 2.0;
        let cy = self.y_1 + h / 2.0;
        let area = w * h;
        let r = w / (h + EPSILON);

        SVector::<f64, 4>::new(cx, cy, area, r)
    }

    pub fn iou(&self, other: &Self) -> f64 {
        let iwidth = (self.x_2.min(other.x_2) - self.x_1.max(other.x_1)).max(0.0);
        let iheight = (self.y_2.min(other.y_2) - self.y_1.max(other.y_1)).max(0.0);
        let iarea = iwidth * iheight;

        let union = self.area() + other.area() - iarea;

        if union == 0.0 {
            return 0.0;
        }

        iarea / union
    }

    pub fn area(&self) -> f64 {
        ((self.x_2 - self.x_1) * (self.y_2 - self.y_1)).max(0.0)
    }

    pub fn speed_direction(&self, other: &Self) -> SVector<f64, 2> {
        let diff = self.to_observation_vector().fixed_rows::<2>(0)
            - other.to_observation_vector().fixed_rows::<2>(0);
        let norm = diff.norm();

        if norm > 0.0 {
            return diff / norm;
        }

        SVector::<f64, 2>::zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_returns_correct_value() {
        let bbox_1 = BBox::new(1.0, 1.0, 2.0, 2.0);
        let bbox_2 = BBox::new(1.0, 1.0, 1.5, 1.5);

        assert_eq!(bbox_1.iou(&bbox_2), 0.25)
    }
}
