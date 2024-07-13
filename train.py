import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer


from dataset import SortDataset

# %%
import torch


class Net(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(C, eps=1e-8)

    def forward(self, x):
        return self.layer_norm(x)


N, C = 8, 4
model = Net(C).half()
x = torch.randn(N, C).half()

torch.onnx.export(model, x, "test_layernorm_export_fp16.onnx", opset_version=12)
# %%


def print_model_state_dict(model: nn.Module):
    print("Model's state_dict:")
    for param in model.state_dict():
        print(param, "\t", model.state_dict()[param].size())


def batch_end_callback(trainer: Trainer):
    if trainer.iter_num % 25 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )


def train_model(trainer: Trainer, model: nn.Module):
    print_model_state_dict(model)

    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()

    print_model_state_dict(model)

    torch.save(model.state_dict(), "model.pt")


def get_model(vocab_size=3, block_size=12) -> nn.Module:
    config = GPT.get_default_config()
    config.model_type = "gpt-nano"
    config.vocab_size = vocab_size
    config.block_size = block_size
    model = GPT(config)
    return model


def get_trainer(model: nn.Module, dataset: Dataset):
    config = Trainer.get_default_config()
    config.learning_rate = 5e-4
    config.max_iters = 1000
    config.batch_size = 32
    trainer = Trainer(config, model, dataset)
    return trainer


if __name__ == "__main__":

    model = get_model()
    dataset = SortDataset("train")
    trainer = get_trainer(model, dataset)

    train_model(trainer, model)
