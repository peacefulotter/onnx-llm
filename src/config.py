import torch
import argparse


def none_or_str(value):
    if value == "None":
        return None
    return value


def default_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model_type",
        default="gpt-nano",
        type=str,
        choices=[
            "openai-gpt",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "gopher-44m",
            "gpt-mini",
            "gpt-micro",
            "gpt-nano",
        ],
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--block_size", default=11, type=int)
    parser.add_argument("--vocab_size", default=3, type=int)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def get_args():
    parser = default_parser()
    args = parser.parse_args()
    return args


def get_training_args():
    parser = default_parser()
    parser.add_argument("--max_iters", default=1000, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    print(args)
