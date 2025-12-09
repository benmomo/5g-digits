# utils/metrics.py

import onnx

def estimate_macs(onnx_model_path: str) -> int:
    """
    Approximate MACs for an ONNX model.
    Includes:
      - Conv (1D/2D)
      - Gemm
      - MatMul (fallback for dense layers)
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # Collect shape information
    shape_map = {}

    def register_shape(vi):
        if vi.type.tensor_type.shape.dim:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            shape_map[vi.name] = dims

    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        register_shape(vi)

    macs = 0

    for node in graph.node:

        if node.op_type == "Conv":
            # Detect Conv1d or Conv2d
            out_name = node.output[0]
            if out_name not in shape_map:
                continue
            out_shape = shape_map[out_name]  # e.g. [N, C_out, L] or [N, C_out, H, W]

            if len(out_shape) == 3:    # Conv1d
                _, Cout, Lout = out_shape
                spatial = Lout
            elif len(out_shape) == 4:  # Conv2d
                _, Cout, Hout, Wout = out_shape
                spatial = Hout * Wout
            else:
                continue

            # Read weight tensor (kernel)
            W = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    W = init
                    break
            if W is None:
                continue

            w_dims = list(W.dims)  # e.g. [C_out, C_in, K] or [C_out, C_in, Kh, Kw]
            if len(w_dims) == 3:
                _, Cin, K = w_dims
                kernel_elems = Cin * K
            elif len(w_dims) == 4:
                _, Cin, Kh, Kw = w_dims
                kernel_elems = Cin * Kh * Kw
            else:
                continue

            macs += Cout * spatial * kernel_elems

        elif node.op_type in ["Gemm", "MatMul"]:
            # Dense layer approximation: MACs ≈ M*N
            W = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    W = init
                    break
            if W is None:
                continue

            dims = list(W.dims)
            if len(dims) >= 2:
                # product of last two dims
                macs += dims[-2] * dims[-1]

    return int(macs)
