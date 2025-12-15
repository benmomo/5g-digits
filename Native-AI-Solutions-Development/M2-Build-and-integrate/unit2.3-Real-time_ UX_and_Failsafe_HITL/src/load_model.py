import onnxruntime as ort

def load_model(path: str, providers=None):
    """
    Load an ONNX model using ONNX Runtime.
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(path, providers=providers)

    input_tensor = session.get_inputs()[0]
    output_tensor = session.get_outputs()[0]

    return session, input_tensor, output_tensor
