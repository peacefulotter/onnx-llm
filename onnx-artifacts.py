import io
import onnx
import torch
import numpy as np
import torch._dynamo.config
import onnx.onnx_operators_pb
from onnxruntime.training import artifacts

from model import get_model, load_model, get_checkpoint_path

torch._dynamo.config.dynamic_shapes = True

import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training import artifacts


# Define a custom loss block that takes in two inputs
# and performs a weighted average of the losses from these
# two inputs.
class Softmax(onnxblock.Block):
    def __init__(self):
        onnxblock.loss.L1Loss
        self._exp = onnxblock.blocks.Pow(exponent=np.e)
        self._sum = onnxblock.blocks.ReduceSum(keepdims=1)

    def build(self, input):
        return self._exp(input) / self._sum(self._exp(input), axis=1)


class CrossEntropyLoss(onnxblock.Block):
    def __init__(self):
        self._softmax = Softmax()
        self._log = onnxblock.blocks.Log()
        self._neg = onnxblock.blocks.Neg()
        self._gather = onnxblock.blocks.ReduceSum()

    def build(self, loss_input_name1, loss_input_name2):
        # print(loss_input_name1, loss_input_name2)
        # print(loss_input_name1)
        # input1 = self._softmax(loss_input_name1)
        # input2 = self._gather("labels")
        return loss_input_name2  # self._neg(self._log(input1)) * input2


if __name__ == "__main__":

    model_type = "gpt-nano"
    vocab_size = 3
    block_size = 11

    model = get_model(
        model_type=model_type, vocab_size=vocab_size, block_size=block_size
    )

    # Load checkpoint
    load_model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    bs = 8  # any batch size works
    example_input = torch.randint(
        high=vocab_size,
        size=(bs, block_size),
        dtype=torch.int32,
    )

    model = torch.jit.script(model)

    f = io.BytesIO()
    input_names = ["input"]
    # output_names = ["output", "loss"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model=model,
        args=example_input,
        f=f,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        dynamic_axes=dynamic_axes,
        opset_version=15,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.TRAINING,
        keep_initializers_as_inputs=False,
    )

    onnx_model = onnx.load_model_from_string(f.getvalue())

    requires_grad = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    frozen_params = [
        name for name, param in model.named_parameters() if not param.requires_grad
    ]

    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.AdamW,
        loss=artifacts.LossType.CrossEntropyLoss,  # (),  #
        # loss=CrossEntropyLoss(),
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        additional_output_names=output_names,
    )
