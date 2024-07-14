import io
import onnx
import torch
import numpy as np
import torch._dynamo.config
import onnx.onnx_operators_pb

from onnxruntime.training import artifacts

import config
from model import get_model, load_model

torch._dynamo.config.dynamic_shapes = True


if __name__ == "__main__":

    args = config.get_args()

    model = get_model(args)

    if args.pretrained:
        load_model(model)

    example_input = torch.randint(
        high=args.vocab_size,
        size=(args.batch_size, args.block_size),
        dtype=torch.int32,
    )

    model = torch.jit.script(model)

    f = io.BytesIO()
    input_names = ["input"]
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
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        additional_output_names=output_names,
    )
