import onnx
import torch
import torch.nn.functional as F
from onnxruntime.training.experimental import export_gradient_graph

from train import get_model

torch._dynamo.config.dynamic_shapes = True


def log_softmax(logits: torch.Tensor, dim) -> torch.Tensor:
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    stable_logits = logits - max_logits
    exp_logits = torch.exp(stable_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    log_probs = stable_logits - torch.log(sum_exp_logits)
    return log_probs


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    log_probs = log_softmax(logits, dim=1)
    selected_log_probs = log_probs[range(log_probs.shape[0]), targets]
    loss = -selected_log_probs.mean()
    return loss


loss_fn = cross_entropy_loss

bs = 8  # any batch size works
vocab_size = 3
block_size = 11

print(
    loss_fn(
        torch.randn(bs, block_size, vocab_size),
        torch.randint(vocab_size, (bs, block_size)),
    )
)


model = get_model(vocab_size=vocab_size, block_size=block_size)

example_input = torch.randint(
    high=vocab_size,
    size=(bs, block_size),
    dtype=torch.long,
)

print(model(example_input).shape)

gradient_graph_path = "gradient_graph.onnx"

print(f'Writing gradient graph to "{gradient_graph_path}".')
export_gradient_graph(
    model, loss_fn, example_input, example_input, gradient_graph_path, opset_version=17
)
print(f'Done writing gradient graph to "{gradient_graph_path}".')

"""print("Checking gradient graph...")
onnx_model = onnx.load(gradient_graph_path)
onnx.checker.check_model(onnx_model)
print("✅ Gradient graph should be okay.")"""

"""
print("Creating Adam optimizer...")
optimizer = AdamOnnxGraphBuilder(model.named_parameters())
onnx_optimizer = optimizer.export()
optimizer_graph_path = "mnist_optimizer_graph.onnx"
print(f'Writing optimizer graph to "{optimizer_graph_path}".')
onnx.save(onnx_optimizer, optimizer_graph_path)

print("Checking optimizer graph...")
onnx_optimizer = onnx.load(optimizer_graph_path)
onnx.checker.check_model(onnx_optimizer)
print("✅ Optimizer graph should be okay.")
"""

print("✅ Done.")
