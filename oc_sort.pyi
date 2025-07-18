class BBox:
    """Class representing the properties of a bounding box."""
    def __new__(cls, x_1: float, y_1: float, x_2: float, y_2: float) -> BBox:
        """Construct a new BBox out of the top left (x_1, y_1) and bottom right (x_2, y_2) coordinates."""

    @property
    def x_1(self) -> float:
        """The min x coordinate of the bbox."""

    @property
    def y_1(self) -> float:
        """The min y coordinate of the bbox."""

    @property
    def x_2(self) -> float:
        """The max x coordinate of the bbox."""

    @property
    def y_2(self) -> float:
        """The max y coordinate of the bbox."""

class Detection:
    """Class representing the properties of a valid object detection."""
    def __new__(cls, bbox: BBox, class_id: int, score: float) -> Detection:
        """Construct a new detection out of the given bbox and class_id."""

    @property
    def bbox(self) -> BBox:
        """The bbox of the detection."""

    @property
    def class_id(self) -> int:
        """The class id of the detection."""

    @property
    def score(self) -> int:
        """The confidence score of the detection."""

class Track:
    """Class representing a tracked object."""

    @property
    def id(self) -> int:
        """The id of the tracked object."""

    @property
    def bbox(self) -> BBox:
        """The bbox of the tracked object."""

    @property
    def class_id(self) -> int:
        """The class id of the tracked object."""

class OCSort:
    """The oc sort object tracker."""

    def __new__(
        cls,
        max_age: int,
        iou_threshold: float,
        delta_t: int,
        score_threshold: float,
        min_hit_streak: int,
    ) -> OCSort:
        """Construct a new tracker.

        ## Args:
            - max_age: the maximum number of updates a tracked bbox can have without new detections.
            - iou_threshold: the minimum iou score needed for a valid association between a tracked bbox and new observation.
            - delta_t: time difference used in velocity calculations.
            - score_threshold: the score threshold used for byte association.
            - min_hit_streak: the minimum number of consecutive associations a track needs to be returned.
        """

    def get_trackers(self) -> list[Track]:
        """Return currently tracked objects."""

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the state of the object tracker.

        Returns a list of the tracked objects after the update process.

        Note: if no detections are made pass an empty list to advance the internal state.
        """
