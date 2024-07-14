# Requirements
- Python 3 (Tested on 3.10.12)
- Pip (Tested on 22.0.2)

# Installation

### Setup

```sh
# 1. Create and activate the environment
python -m venv venv
source venv/bin/activate

# 2. Install the required packages
pip install -r requirements.txt

# (Optional) Train the model for a few iterations and save a checkpoint file
python torch-training.py
```

### Inference Mode

```sh
# 3.1 Convert the model to ONNX, this step creates a .onnx file
python src/onnx-inference.py

# 3.2. Run the model on ONNX
python src/onnx-inference-test.py
```

### Training Mode

```sh
# 4.1 Conver the model to ONNX artifacts, this step creates 3 .onnx files and a "checkpoint" file
python src/onnx-training.py

# 4.2. Run training / eval step on ONNX
python src/onnx-training-test.py
```
