import json
import wandb


with open("training.json") as f:
    js = json.load(f)

wandb.init(project="onnx-llm", config=js["config"])

data = js["data"]

for d in data:
    wandb.log(d)

wandb.finish()
