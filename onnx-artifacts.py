import io
import onnx
import torch
import torch._dynamo.config
from onnxruntime.training import artifacts

from model import get_model, load_model, get_checkpoint_path

torch._dynamo.config.dynamic_shapes = True

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
