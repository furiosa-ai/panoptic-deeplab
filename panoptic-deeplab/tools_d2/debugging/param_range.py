import onnx
from onnx import numpy_helper
import numpy as np

fp32_path = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic.onnx'
quant_path = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal50.onnx'

model_fp32 = onnx.load(fp32_path)
model_quant = onnx.load(quant_path)

def weight_range(model):
    weights = model.graph.initializer
    shape_list = []
    min_list = []
    max_list = []
    #histogram
    for weight in weights:
        weight = numpy_helper.to_array(weight)
        #w_flat = np.flatten(weight)
        try:
            min_list.append(weight.min())
            max_list.append(weight.max())
        except:
            print(weight.shape)
            assert weight.shape == (0,)
            min_list.append(weight)
            max_list.append(weight)
            
    
    for idx in range(len(min_list)):
        min_elt = min_list[idx]
        max_elt = max_list[idx]

        if min_elt < -10:
            print("%dth min:"%idx, min_elt)
        if min_elt > 10:
            print("%dth max:"%idx, max_elt)

def q_param(model):
    weights = model.graph.initializer
    min_max_list = []
    #histogram
    for weight in weights:
        #print(weight.name)
        if 'scale' in weight.name:
            weight = numpy_helper.to_array(weight)
            print(weight)

#        weight = numpy_helper.to_array(weight)


#q_param(model_quant)
weight_range(model_quant)