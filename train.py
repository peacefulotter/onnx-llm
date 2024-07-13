import torch
import torch.nn as nn

from mingpt.model import GPT
from mingpt.trainer import Trainer


from dataset import SortDataset


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


if __name__ == "__main__":
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = 50257
    model_config.block_size = 1024
    model = GPT(model_config)

    train_dataset = SortDataset("train")

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4  # many possible options, see the file
    train_config.max_iters = 1000
    train_config.batch_size = 32
    trainer = Trainer(train_config, model, train_dataset)

    train_model(trainer, model)
