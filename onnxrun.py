import onnx
import torch
import onnxruntime as ort
import numpy as np
dummy_input = torch.randint(3, size=(1,11), dtype=torch.long,device="cuda:0")
onnx_model = onnx.load("test.onnx.pb")
ort_sess = ort.InferenceSession('test.onnx.pb', )
outputs = ort_sess.run(None, {'input': dummy_input.detach().cpu().numpy()})
# Print Result
result = outputs[0]
print("input")
print(dummy_input[0])
print("output ---")
firstr = result[0]
print(len(firstr))
max_ind = [ max( (v, i) for i, v in enumerate(a) )[1] for a in firstr]
print(max_ind)



