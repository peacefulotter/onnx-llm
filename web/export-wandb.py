import json
import wandb

wandb.init(project="onnx-llm")

with open("training.json") as f:
    data = json.load(f)

for d in data:
    wandb.log(d)
wandb.finish()
