import torch
import torch._dynamo.config

from mingpt.model import GPT

# from dataset import SortDataset

torch._dynamo.config.dynamic_shapes = True


if __name__ == "__main__":
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = 50257
    model_config.block_size = 1024
    model = GPT(model_config)

    # Load checkpoint and move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load("model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # model.train()
    # dataset = SortDataset("train")
    # dummy_input_train = SortDataset("train")[:32]

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
