# src/validation.py

import numpy as np


def validate_shape(window: np.ndarray, expected_shape) -> bool:
    """
    Validate input window shape.

    expected_shape is typically the model input tensor shape:
        [batch_dim, window_length, num_channels]

    We accept either:
        - (window_length, num_channels)   --> 2D
        - (batch_dim, window_length, num_channels) --> 3D
    """

    # Expected format: [B, W, C]
    if len(expected_shape) != 3:
        return False

    batch_dim, exp_W, exp_C = expected_shape

    # 2D input: (W, C)
    if window.ndim == 2:
        return window.shape == (exp_W, exp_C)

    # 3D input: (B, W, C)
    if window.ndim == 3:
        # Check the time and channel dimensions
        if window.shape[1:] != (exp_W, exp_C):
            return False

        # If batch_dim is an integer, we can also check it; if it's symbolic, ignore it
        if isinstance(batch_dim, int) and window.shape[0] != batch_dim:
            return False

        return True

    # Any other rank is invalid
    return False


def validate_dtype(window: np.ndarray, expected_dtype) -> bool:
    """Validate input window dtype."""
    return window.dtype == expected_dtype


def validate_range(
    window: np.ndarray,
    min_value: float = -4.0,
    max_value: float = 4.0,
) -> bool:
    """
    Ensure values fall within the expected IMU range.

    Reject NaNs and infinities as well.
    """
    if not np.isfinite(window).all():
        return False

    return (window >= min_value).all() and (window <= max_value).all()


def validate_input(
    window: np.ndarray,
    expected_shape,
    expected_dtype,
) -> bool:
    """Master validator: combine all checks."""
    if not validate_shape(window, expected_shape):
        return False
    if not validate_dtype(window, expected_dtype):
        return False
    if not validate_range(window):
        return False
    return True
