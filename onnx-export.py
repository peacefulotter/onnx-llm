import torch
import torch._dynamo.config

from model import get_model, load_model, get_checkpoint_path

torch._dynamo.config.dynamic_shapes = True


if __name__ == "__main__":

    model_type = "gpt-nano"
    vocab_size = 3
    block_size = 11

    model = get_model(
        model_type=model_type, vocab_size=vocab_size, block_size=block_size
    )

    # Load checkpoint and move model to device
    load_model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    bs = 8  # any batch size works
    example_input = torch.randint(
        high=vocab_size,
        size=(bs, block_size),
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
