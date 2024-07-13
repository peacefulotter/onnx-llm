import torch
import torch._dynamo.config

from train import get_model

torch._dynamo.config.dynamic_shapes = True


if __name__ == "__main__":
    model = get_model()

    # Load checkpoint and move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load("model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    input_length = 11
    dummy_batch_size = 16
    dummy_input = torch.randint(
        high=3, size=(dummy_batch_size, input_length), dtype=torch.long, device=device
    )

    model = torch.jit.script(model)

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f="model.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=15,
    )
