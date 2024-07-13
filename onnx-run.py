import torch
import numpy as np
import onnxruntime as ort

# Create a random input tensor
test_input = torch.randint(3, size=(1, 11), dtype=torch.long)
test_input = test_input.detach().cpu().numpy()

# Load the ONNX model and run the inference
ort_sess = ort.InferenceSession("model.onnx")
outputs = ort_sess.run(None, {"input": test_input})

# Get the predictions
preds: np.ndarray = outputs[0].squeeze()
preds = preds.argmax(axis=1)

# Print the input and output, see if they match
print(f"Input:  {test_input[0][:6]}")
print(f"Output: {preds[-6:]}")
