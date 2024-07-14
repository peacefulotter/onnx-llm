import torch
from mingpt.model import GPT as _GPT


class GPT(_GPT):
    def __init__(self, config):
        super().__init__(config)
        self.config = config


def get_model(
    model_type: str = "gpt-nano", vocab_size: int = 3, block_size: int = 12
) -> GPT:
    config = GPT.get_default_config()
    config.model_type = model_type
    config.vocab_size = vocab_size
    config.block_size = block_size
    model = GPT(config)
    return model


def get_checkpoint_path(model: GPT):
    c = model.config
    path = f"{c.model_type}_vs={c.vocab_size}_bs={c.block_size}.pt"
    return path


def save_model(model: GPT):
    path = get_checkpoint_path(model)
    torch.save(model.state_dict(), path)


def load_model(model: GPT):
    path = get_checkpoint_path(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
