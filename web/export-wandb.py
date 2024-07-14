import json
import wandb


with open("training.json") as f:
    js = json.load(f)

c = js["config"]
run_name = f'onnx_gpt-nano_iters={c['maxIters']}_batch={c['batchSize']}_lr=unknown_voc={c["vocabSize"]}_block={c["blockSize"]}'

wandb.init(project="onnx-llm", name=run_name, config=c)

data = js["data"]

for d in data:
    wandb.log(d)

wandb.finish()
