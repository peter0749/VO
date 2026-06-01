"""Custom exceptions for slam_dnn."""


class TrackingLostError(Exception):
    """Raised when tracking is lost (insufficient matches or pose estimation fails).

    This typically triggers a frame skip in visual odometry.
    """
    pass
