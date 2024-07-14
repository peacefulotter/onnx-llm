# ONNX-LLM

> Allows training and evaluation of a Large Language Model (LLM) on ONNX.


# Requirements
- Python 3 (Tested on 3.10.12)
- Pip (Tested on 22.0.2)
- Node.js <b>v16</b> (Tested on 16.20.0), use [nvm](https://github.com/nvm-sh/nvm) to install and manage Node.js versions
- NPM (Tested on 8.19.4)
- <it>Optional</it> If you use Bun, it should work too (Tested on 1.1.20)

# Guide

#### Offline setup
Start with the offline setup under `python/`. The goal is to convert a torch model to a ONNX model and artifacts. The ONNX model can then be used for inference and training on device. See `python/README.md` for more details.

#### Web setup
The web setup under `web/` is a NextJS app that uses `onnxruntime-web` to run the model in the browser. See `web/README.md` for more details.