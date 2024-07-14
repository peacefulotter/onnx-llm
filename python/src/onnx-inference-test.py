import torch
import numpy as np
import onnxruntime as ort

import config
from model import get_checkpoint_path

args = config.get_args()

# Create a random input tensor
test_input = torch.randint(
    args.vocab_size, size=(args.batch_size, args.block_size), dtype=torch.long
)
test_input = test_input.detach().cpu().numpy()

# Load the ONNX model and run the inference
cp_path = get_checkpoint_path(args, onnx=True)
ort_sess = ort.InferenceSession(cp_path)
outputs = ort_sess.run(None, {"input": test_input})

# Get the predictions
preds: np.ndarray = outputs[0].squeeze()
preds = preds.argmax(axis=1)

# Print the input and output, see if they match
print(f"Input:  {test_input[0][:6]}")
print(f"Output: {preds[-6:]}")
