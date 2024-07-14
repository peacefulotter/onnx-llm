import torch
from mingpt.model import GPT as _GPT


class GPT(_GPT):
    def __init__(self, config):
        super().__init__(config)
        self.config = config


def get_model(args) -> GPT:
    config = GPT.get_default_config()
    config.model_type = args.model_type
    config.vocab_size = args.vocab_size
    config.block_size = args.block_size
    config.args = args
    model = GPT(config)
    return model


def get_checkpoint_path(config, onnx=False):
    c = config
    path = f"{c.model_type}_vs={c.vocab_size}_bs={c.block_size}.pt"
    if onnx:
        path = path.replace(".pt", ".onnx")
    return path


def save_model(model: GPT):
    path = get_checkpoint_path(model.config)
    torch.save(model.state_dict(), path)


def load_model(model: GPT):
    path = get_checkpoint_path(model.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
