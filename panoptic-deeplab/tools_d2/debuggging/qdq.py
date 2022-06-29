import onnx
from onnx import numpy_helper
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort

fp32_path = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic.onnx'
quant_path = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal50-test-2output.onnx'

sess = ort.InferenceSession(quant_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

inputs = Image.open('/root/ljh726/PanopticDeepLab/data/cityscape_berlin.png')
inputs = np.array(inputs).astype(np.float32)
print(inputs.shape)
inputs = np.transpose(inputs, (2,0,1))
outputs = sess.run(None, {'image':inputs})

x = outputs[3]
y = outputs[4]

print(x)
print(y)

print(x*0.23025)
print(np.mean(np.abs(y-x*0.23025)))