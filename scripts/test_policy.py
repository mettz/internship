import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("policies/isaacflie.onnx")

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type

# Print input info
print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")
print(f"Input type: {input_dtype}")

while True:
    # Create dummy input (adapt to your real input shape and type)
    # Example: input_shape = [1, 4] → replace with your actual input
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Run inference
    outputs = session.run(None, {input_name: dummy_input})

    # Print outputs with 18 decimal digits
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        flat_output = output.flatten()
        for j, val in enumerate(flat_output):
            print(f"  [{j}] = {val:.18f}")
