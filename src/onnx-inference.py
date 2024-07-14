import torch
import torch._dynamo.config

import config
from model import get_model, load_model, get_checkpoint_path

torch._dynamo.config.dynamic_shapes = True


if __name__ == "__main__":

    args = config.get_args()
    model = get_model(**args)

    # Load checkpoint and move model to device
    if args.pretrained:
        load_model(model)

    bs = args.batch_size
    example_input = torch.randint(
        high=args.vocab_size,
        size=(bs, args.block_size),
        dtype=torch.int32,
    )

    f = get_checkpoint_path(model)
    f = f.replace(".pt", ".onnx")
    print(f"Writing ONNX model to {f}")

    model = torch.jit.script(model)

    torch.onnx.export(
        model=model,
        args=example_input,
        f=f,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=15,
    )
