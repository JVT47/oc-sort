use oc_sort::{BBox, Detection, OCSort};

fn main() {
    let mut sort = OCSort::new(5, 0.1, 3);

    let detections = vec![Detection {
        bbox: BBox::new(1.0, 1.0, 2.0, 2.0),
        class: 1,
    }];
    let tracks = sort.update(detections);
    println!("number of tracks {:?}, Tracks: {:?}", tracks.len(), tracks);

    let detections = vec![
        Detection {
            bbox: BBox::new(1.1, 1.0, 2.1, 2.0),
            class: 1,
        },
        Detection {
            bbox: BBox::new(1.0, 1.0, 2.0, 2.0),
            class: 0,
        },
    ];
    let tracks = sort.update(detections);
    println!("number of tracks {:?}, Tracks: {:?}", tracks.len(), tracks);
}
