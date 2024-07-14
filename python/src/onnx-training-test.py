import torch
from onnxruntime.training.api import CheckpointState, Module, Optimizer

from dataset import SortDataset


state = CheckpointState.load_checkpoint("checkpoint")
module = Module("training_model.onnx", state, "eval_model.onnx", device="cpu")
optimizer = Optimizer("optimizer_model.onnx", module)

data = SortDataset(split="train")
x, y = data[0]

print("Original", x.shape, x.dtype, y.shape, y.dtype)

# Simulate batch_size of 2
x = torch.stack((x, x)).int().numpy()
y = torch.stack((y, y)).flatten().numpy()
print("Adapted to ONNX", x.shape, x.dtype, y.shape, y.dtype)

module.train()
training_loss = module(x, y)
optimizer.step()
module.lazy_reset_grad()

print(training_loss)
print(training_loss[0].shape, training_loss[1].shape)

# Eval
data = SortDataset(split="test")
x, y = data[1]

x = torch.stack((x, x)).int().numpy()
y = torch.stack((y, y)).flatten().numpy()

module.eval()
eval_loss = module(x, y)
print(eval_loss)
