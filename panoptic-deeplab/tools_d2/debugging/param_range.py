from venv import create
import onnx
from onnx import numpy_helper
import numpy as np

fp32_path = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic.onnx'
quant_path = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal100_seed1000.onnx'

model_fp32 = onnx.load(fp32_path)
model_quant = onnx.load(quant_path)


def print_hist(scale_list,tab=""):
    tab += "\t"
    if type(scale_list) is dict:
        for key in scale_list.keys():
            print("%s%s:"%(tab,key))
            print_hist(scale_list[key], tab)
    else:
        for i, (scale,zero_pt,q_max) in enumerate(scale_list):
            print("%s%dth scale:%f zero_pt:%d max:%f"%(tab,i,scale, zero_pt, q_max))

def create_hist_dict(scales):
    hist = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}
    for i in range(12):
        if i<=5:
            hist['semantic']['res5'].append(scales[i])
        elif i<=8:
            hist['semantic']['res3'].append(scales[i])
        else:
            hist['semantic']['res2'].append(scales[i])
    for i in range(12,24):
        if i-12<=5:
            hist['instance']['res5'].append(scales[i])
        elif i-12<=8:
            hist['instance']['res3'].append(scales[i])
        else:
            hist['instance']['res2'].append(scales[i])
    return hist


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

def q_param(model, word):
    weights = model.graph.initializer
    scale_list = []
    #histogram
    cnt = 0
    for weight in weights:
        #print(weight.name)       
        if word in weight.name and 'Concat' in weight.name:
            name = weight.name
            weight = numpy_helper.to_array(weight)
            scale_list.append(weight)
            cnt +=1
            print("%dth %s: "%(cnt,name), weight)

    return scale_list
#        weight = numpy_helper.to_array(weight)


def get_q_max(model_quant):
    model = onnx.load(model_quant)
    weights = model.graph.initializer
    q_params = []
    for weight in weights:
        if 'Concat' in weight.name:
            q_params.append(numpy_helper.to_array(weight))

    q_max_list = []
    temp = 0
    for i in range(0,len(q_params),2):
        scale = q_params[i+1]
        zero_pt = q_params[i]
        q_max = (256-zero_pt)*scale
        q_max_list.append((scale, zero_pt, q_max))
        if i//2 in [4,6,8,13,15,17]:
            q_max_list.append(max(q_max_list[temp-i//2:]))
            temp = i//2
    
    return q_max_list

#word = scale
q_max_list = get_q_max(quant_path)
hist = create_hist_dict(q_max_list)
print_hist(hist)
#weight_range(model_quant)