use std::f64::EPSILON;

use nalgebra::SVector;

/// Struct that represents a bounding box.
#[derive(Clone, Copy, Default, Debug)]
pub struct BBox {
    /// The top left x coordinate.
    pub x_1: f64,
    /// The top left y coordinate.
    pub y_1: f64,
    /// The bottom right x coordinate.
    pub x_2: f64,
    /// The bottom right y coordinate.
    pub y_2: f64,
}

impl BBox {
    /// Creates a new bounding box from the top left (x_1, y_1) and
    /// bottom right (x_2, y_2) coordinates.
    pub fn new(x_1: f64, y_1: f64, x_2: f64, y_2: f64) -> Self {
        if x_1 > x_2 || y_1 > y_2 {
            return BBox {
                x_1: 0.0,
                y_1: 0.0,
                x_2: 0.0,
                y_2: 0.0,
            };
        };
        BBox { x_1, y_1, x_2, y_2 }
    }

    /// Creates a bounding box from a Kalman Filter state vector.
    pub fn from_state_vector(state_vector: SVector<f64, 7>) -> Self {
        if state_vector[2] < 0.0 || state_vector[3] < 0.0 {
            return BBox::new(0.0, 0.0, 0.0, 0.0);
        }
        let w = (state_vector[2] * state_vector[3]).sqrt();
        let h = state_vector[2] / w;

        Self::new(
            state_vector[0] - w / 2.0,
            state_vector[1] - h / 2.0,
            state_vector[0] + w / 2.0,
            state_vector[1] + h / 2.0,
        )
    }

    /// Creates a Kalman Filter observation vector out of this bounding box.
    pub fn to_observation_vector(&self) -> SVector<f64, 4> {
        let w = (self.x_2 - self.x_1).max(0.0);
        let h = (self.y_2 - self.y_1).max(0.0);

        let cx = self.x_1 + w / 2.0;
        let cy = self.y_1 + h / 2.0;
        let area = w * h;
        let r = w / (h + EPSILON);

        SVector::<f64, 4>::new(cx, cy, area, r)
    }

    /// Calculates the intersection over union value between this bounding box and a given bbox.
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

    /// Calculates the area of this bounding box. Returns 0 if negative.
    pub fn area(&self) -> f64 {
        ((self.x_2 - self.x_1) * (self.y_2 - self.y_1)).max(0.0)
    }

    /// Returns a unit vector pointing to the direction of the other bounding box.
    /// Returns a zero vector if the speed direction cannot be determined.
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
    fn test_invalid_new_bbox_returns_zero_bbox() {
        let bbox = BBox::new(3.0, 4.0, 2.0, 5.0);

        assert_eq!(bbox.x_1, 0.0);
        assert_eq!(bbox.x_2, 0.0);
        assert_eq!(bbox.y_1, 0.0);
        assert_eq!(bbox.y_2, 0.0);
    }

    #[test]
    fn test_from_state_vector_returns_zero_bbox_for_invalid_state() {
        let state_vector = SVector::<f64, 7>::from_vec(vec![1.0, 1.0, 4.0, -1.0, 0.0, 0.0, 0.0]);
        let bbox = BBox::from_state_vector(state_vector);

        assert_eq!(bbox.x_1, 0.0);
        assert_eq!(bbox.x_2, 0.0);
        assert_eq!(bbox.y_1, 0.0);
        assert_eq!(bbox.y_2, 0.0);
    }

    #[test]
    fn test_iou_returns_correct_value_1() {
        let bbox_1 = BBox::new(1.0, 1.0, 2.0, 2.0);
        let bbox_2 = BBox::new(1.0, 1.0, 1.5, 1.5);

        assert_eq!(bbox_1.iou(&bbox_2), 0.25)
    }

    #[test]
    fn test_iou_returns_correct_value_2() {
        let bbox_1 = BBox::new(0.0, 0.0, 1.0, 2.0);
        let bbox_2 = BBox::new(1.0, 2.0, 3.0, 3.0);

        assert_eq!(bbox_1.iou(&bbox_2), 0.0)
    }

    #[test]
    fn test_iou_returns_correct_value_3() {
        let bbox_1 = BBox::new(0.0, 0.0, 3.0, 3.0);
        let bbox_2 = BBox::new(1.0, 1.0, 2.0, 2.0);

        assert_eq!(bbox_1.iou(&bbox_2), 1.0 / 9.0)
    }
}
