import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("policies/isaacflie-noise.onnx")

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type

# Print input info
print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")
print(f"Input type: {input_dtype}")

obs = np.zeros((1, 146), dtype=np.float32)

# Position at origin
obs[0, 0:3] = 0.0

# Rotation matrix = identity (flat hover)
obs[0, 3:12] = np.eye(3).flatten()

# Linear velocity ≈ 0
obs[0, 12:15] = 0.0

# Angular velocity ≈ 0
obs[0, 15:18] = 0.0
obs[0, 15] = 1.0

# Action history = small RPM-normalized values (~0.5)
obs[0, 18:] = 0.5

outputs = session.run(None, {input_name: obs})

# Print outputs with 18 decimal digits
for i, output in enumerate(outputs):
    print(f"Output {i}:")
    flat_output = output.flatten()
    for j, val in enumerate(flat_output):
        print(f"  [{j}] = {val:.18f}")

# while True:
#     # Create dummy input (adapt to your real input shape and type)
#     # Example: input_shape = [1, 4] → replace with your actual input
#     dummy_input = np.random.uniform(-1, 1, size=(1, 146)).astype(np.float32)

#     # Run inference
#     outputs = session.run(None, {input_name: dummy_input})

#     # Print outputs with 18 decimal digits
#     for i, output in enumerate(outputs):
#         print(f"Output {i}:")
#         flat_output = output.flatten()
#         for j, val in enumerate(flat_output):
#             print(f"  [{j}] = {val:.18f}")
