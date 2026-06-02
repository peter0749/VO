"""Constant-velocity motion model for relative pose prediction."""
import numpy as np


class MotionModel:
    """Predicts relative camera poses using constant-velocity assumption."""

    def __init__(self, ema_alpha: float = 0.5):
        """
        Args:
            ema_alpha: Smoothing factor for exponential moving average (0 = ignore new, 1 = use only last).
        """
        self.ema_alpha = ema_alpha
        self._R_vel = np.eye(3, dtype=np.float64)
        self._t_vel = np.zeros(3, dtype=np.float64)
        self._initialized = False

    def update(self, R_rel: np.ndarray, t_rel: np.ndarray) -> None:
        """Update motion velocity estimate with a new successful relative pose.

        Args:
            R_rel: Relative rotation matrix.
            t_rel: Relative translation vector.
        """
        R_rel = np.asarray(R_rel, dtype=np.float64)
        t_rel = np.asarray(t_rel, dtype=np.float64).ravel()

        if not self._initialized:
            self._R_vel = R_rel.copy()
            self._t_vel = t_rel.copy()
            self._initialized = True
        else:
            # EMA translation
            self._t_vel = (1.0 - self.ema_alpha) * self._t_vel + self.ema_alpha * t_rel

            # EMA rotation via linear blending + SVD orthogonalization (very fast & robust)
            R_blend = (1.0 - self.ema_alpha) * self._R_vel + self.ema_alpha * R_rel
            U, _, Vt = np.linalg.svd(R_blend)
            self._R_vel = U @ Vt

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next relative pose based on current velocity.

        Returns:
            Tuple (R_pred, t_pred) of predicted relative rotation and translation.
        """
        return self._R_vel.copy(), self._t_vel.copy()

    def reset(self) -> None:
        """Reset motion model state."""
        self._R_vel = np.eye(3, dtype=np.float64)
        self._t_vel = np.zeros(3, dtype=np.float64)
        self._initialized = False
