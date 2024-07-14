import torch.nn as nn
from torch.utils.data import Dataset

from mingpt.trainer import Trainer

from model import get_model, save_model
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
        save_model(trainer.model)
        raise Exception("stop training")


def train_model(trainer: Trainer, model: nn.Module):
    print_model_state_dict(model)

    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()

    print_model_state_dict(model)


def get_trainer(model: nn.Module, dataset: Dataset):
    config = Trainer.get_default_config()
    config.learning_rate = 5e-4
    config.max_iters = 1000
    config.batch_size = 32
    trainer = Trainer(config, model, dataset)
    return trainer


if __name__ == "__main__":

    dataset = SortDataset("train")
    model = get_model(model_type="gpt-nano", vocab_size=3, block_size=11)

    trainer = get_trainer(model, dataset)
    train_model(trainer, model)

    save_model(model)
