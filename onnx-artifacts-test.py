import onnx
from onnxruntime.training.api import CheckpointState, Module, Optimizer

from dataset import SortDataset

state = CheckpointState.load_checkpoint("checkpoint")
module = Module("training_model.onnx", state, "eval_model.onnx", device="cpu")
optimizer = Optimizer("optimizer_model.onnx", module)

data = SortDataset(split="train")
x, y = data[0]
print(x)

module.train()
training_loss = module(x)
optimizer.step()
module.lazy_reset_grad()

# Eval
data = SortDataset(split="eval")
x, y = data[0]

module.eval()
eval_loss = module(x)
