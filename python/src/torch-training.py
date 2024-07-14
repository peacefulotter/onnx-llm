import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mingpt.trainer import Trainer

from model import get_model, save_model
from dataset import SortDataset


def print_model_state_dict(model: nn.Module):
    print("Model's state_dict:")
    for param in model.state_dict():
        print(param, "\t", model.state_dict()[param].size())


def batch_end_callback(trainer: Trainer, **kwargs):

    if trainer.model.config.args.use_wandb:
        logits, targets = kwargs["logits"], kwargs["targets"]
        logits = logits.argmax(dim=-1)
        logits = logits.reshape(targets.shape)

        bs = trainer.model.block_size
        mid = bs // 2
        logits = logits[:, mid:]
        targets = targets[:, mid:]

        acc = (logits == targets).float().mean().item()

        data = dict(
            loss=trainer.loss.item(), iter=trainer.iter_num, dt=trainer.iter_dt, acc=acc
        )
        wandb.log(data)

    if trainer.iter_num % 25 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )


def get_trainer(model: nn.Module, dataset: Dataset, args):
    config = Trainer.get_default_config()
    config.learning_rate = args.learning_rate
    config.max_iters = args.max_iters
    config.batch_size = args.batch_size
    trainer = Trainer(config, model, dataset)
    return trainer


def get_run_name(args):
    return f"torch_{args.model_type}_iters={args.max_iters}_batch={args.batch_size}_lr={args.learning_rate}_voc={args.vocab_size}_block={args.block_size}"


if __name__ == "__main__":

    import config

    args = config.get_training_args()
    model = get_model(args)

    dataset = SortDataset("train")
    trainer = get_trainer(model, dataset, args)

    if args.use_wandb:
        wandb.init(project="onnx-llm", name=get_run_name(args), config=args)

    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()

    if args.use_wandb:
        wandb.finish()

    save_model(model)
