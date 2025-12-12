"""
Utility functions for evaluating HAR models in the
NAIS Smart-Glasses project.

This module is intentionally lightweight so it can be
used both locally and in Google Colab.
"""



"""
Utility functions for simple model metrics used in the labs.
"""

from typing import Optional
import onnx


def estimate_macs(onnx_model_path: str) -> Optional[int]:
    """
    Very rough MACs estimate for a feed-forward ONNX model.

    This function walks through the ONNX graph and sums up
    MACs for the most common layers (Conv and Gemm).
    It is intentionally simplified for teaching purposes.

    Args:
        onnx_model_path: Path to the ONNX model file.

    Returns:
        Integer with an approximate MAC count, or None if the
        model cannot be parsed.
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # Build a map from tensor name -> (C, H, W...) shape info (when available)
    shape_map = {}

    # Read shapes from value_info / inputs / outputs
    def _register_shape(value_info):
        name = value_info.name
        if not value_info.type.tensor_type.shape.dim:
            return
        dims = []
        for d in value_info.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                # Unknown dimension (e.g. batch size) – treat as 1
                dims.append(1)
        shape_map[name] = dims

    for vi in graph.value_info:
        _register_shape(vi)
    for vi in graph.input:
        _register_shape(vi)
    for vi in graph.output:
        _register_shape(vi)

    total_macs = 0

    for node in graph.node:
        if node.op_type == "Conv":
            # Conv: MACs ≈ Cout * Hout * Wout * (Cin * Kh * Kw)
            # We try to read output and weight shapes; if missing, skip.
            if not node.output:
                continue
            out_name = node.output[0]
            if out_name not in shape_map:
                continue

            out_shape = shape_map[out_name]  # [N, C_out, H_out, W_out] or [N, C_out, L_out]
            if len(out_shape) == 4:
                _, C_out, H_out, W_out = out_shape
                spatial = H_out * W_out
            elif len(out_shape) == 3:
                # 1D conv case: [N, C_out, L_out]
                _, C_out, L_out = out_shape
                spatial = L_out
            else:
                continue

            # Find weight tensor shape
            W = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    W = init
                    break
            if W is None:
                continue

            # Weight: [C_out, C_in, Kh, Kw] or [C_out, C_in, K]
            w_dims = list(W.dims)
            if len(w_dims) == 4:
                _, C_in, Kh, Kw = w_dims
                kernel_elems = C_in * Kh * Kw
            elif len(w_dims) == 3:
                _, C_in, K = w_dims
                kernel_elems = C_in * K
            else:
                continue

            total_macs += C_out * spatial * kernel_elems

        elif node.op_type == "Gemm":
            # Fully connected: MACs ≈ M * N
            # We look at the weight matrix (input 1)
            W = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    W = init
                    break
            if W is None:
                continue
            # Weight dims: [M, K] or [K, N]; we take product of the two largest dims
            dims = sorted(list(W.dims))
            if len(dims) >= 2:
                total_macs += dims[-1] * dims[-2]

    return int(total_macs)







from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def measure_latency(
    session,
    input_name: str,
    sample: np.ndarray,
    runs: int = 100,
) -> float:
    """
    Measure average inference latency (in milliseconds)
    for a single sample using an ONNX Runtime session.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        Loaded ONNX model session (CPUExecutionProvider).
    input_name : str
        Name of the model input.
    sample : np.ndarray
        Single input sample with the correct shape.
        Shape is usually (1, window_length, num_features).
    runs : int, optional
        Number of repeated inferences to average over.

    Returns
    -------
    float
        Average latency per inference in milliseconds.
    """
    # Ensure sample has a batch dimension
    if sample.ndim == 2:
        sample = sample[None, ...]

    # Warm-up run (avoid including initial overhead)
    _ = session.run(None, {input_name: sample})

    start = time.time()
    for _ in range(runs):
        _ = session.run(None, {input_name: sample})
    end = time.time()

    avg_ms = (end - start) * 1000.0 / runs
    return avg_ms


def run_inference(
    session,
    input_name: str,
    output_name: str,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run inference on a batch of samples and return
    predicted class indices.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        Loaded ONNX model session.
    input_name : str
        Name of the model input.
    output_name : str
        Name of the model output.
    X : np.ndarray
        Input windows, shape (N, window_length, num_features).

    Returns
    -------
    np.ndarray
        Predicted class indices, shape (N,).
    """
    preds = []
    for sample in X:
        # Add batch dimension: (window_length, num_features) -> (1, ...)
        sample_batched = sample[None, ...]
        output = session.run([output_name], {input_name: sample_batched})[0]
        pred_idx = int(np.argmax(output, axis=-1))
        preds.append(pred_idx)
    return np.array(preds, dtype=int)


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute classification accuracy in percent.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels, shape (N,).
    y_pred : np.ndarray
        Predicted class labels, shape (N,).

    Returns
    -------
    float
        Accuracy in percent (0.0–100.0).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."

    correct = (y_true == y_pred).sum()
    total = y_true.size
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """
    Compute a simple confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels, shape (N,).
    y_pred : np.ndarray
        Predicted class labels, shape (N,).
    num_classes : int, optional
        Number of classes. If None, inferred from data.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (num_classes, num_classes),
        where rows = true labels, columns = predicted labels.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm
