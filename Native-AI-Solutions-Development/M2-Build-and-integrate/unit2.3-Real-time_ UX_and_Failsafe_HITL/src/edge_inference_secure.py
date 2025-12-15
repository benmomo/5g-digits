# src/edge_inference_secure.py

import numpy as np
from typing import Any, Dict, Optional, Tuple

from .validation import validate_input  # assumes src/validation.py exists and works
from .load_model import load_model


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def _infer_expected_w_c(session, input_name: str) -> Tuple[int, int]:
    """
    Infer (W, C) from ONNX input shape [B, W, C] where B may be symbolic.
    """
    inp = None
    for i in session.get_inputs():
        if i.name == input_name:
            inp = i
            break
    if inp is None:
        raise ValueError(f"Input tensor '{input_name}' not found in session inputs.")

    shape = inp.shape  # e.g. ['unk__120', 100, 7]
    if len(shape) != 3:
        raise ValueError(f"Expected 3D input shape [B,W,C], got: {shape}")

    _, W, C = shape
    return int(W), int(C)


def secure_predict(
    session,
    input_name: str,
    output_name: str,
    window: np.ndarray,
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
    expected_dtype: Any = np.float32,
    min_value: float = -4.0,
    max_value: float = 4.0,
) -> Dict[str, Any]:
    """
    Secure inference wrapper:
    - validates input
    - runs inference only if valid
    - returns structured result for downstream logic

    Returns:
      {"ok": bool, "error": str|None, "prediction": int|None, "confidence": float|None}
    """
    if expected_shape is None:
        W, C = _infer_expected_w_c(session, input_name)
        expected_shape = (W, C)

    # validate_input() in your src/validation.py should check:
    # shape, dtype, range (and ideally NaN/Inf)
    # If your validate_input does not include range/min/max, keep range validation here.
    ok, reason = validate_input(window, expected_shape, expected_dtype)
    if not ok:
        return {"ok": False, "error": reason, "prediction": None, "confidence": None}

    # Additional safety (range + finiteness) to avoid silent issues
    if not np.isfinite(window).all():
        return {"ok": False, "error": "contains_nan_or_inf", "prediction": None, "confidence": None}
    if (window < min_value).any() or (window > max_value).any():
        return {"ok": False, "error": "out_of_range", "prediction": None, "confidence": None}

    x = window[None, :, :].astype(np.float32)  # add batch dim
    out = session.run([output_name], {input_name: x})[0][0]  # (num_classes,)

    pred = int(np.argmax(out))
    probs = _softmax(out)
    conf = float(probs[pred])

    return {"ok": True, "error": None, "prediction": pred, "confidence": conf}

