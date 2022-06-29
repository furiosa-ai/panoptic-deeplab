import glob
from PIL import Image
import numpy as np
import onnxruntime as ort
from detectron2.modeling import build_model
from demo import setup_cfg, get_parser
import torch
from random import sample
import onnx
from onnx import numpy_helper
from scipy.stats import percentileofscore

torch.manual_seed(1000)
max_list = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}

def print_hist(max_list,tab=""):
    tab += "\t"
    if type(max_list) is dict:
        for key in max_list.keys():
            print("%s%s:"%(tab,key))
            print_hist(max_list[key], tab)
    else:
        for i, max_hist in enumerate(max_list):
            print("%s%dth max:"%(tab,i),max_hist)

def create_hist_dict(histogram):
    hist = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}
    for i in range(12):
        if i<=5:
            hist['semantic']['res5'].append(histogram[i])
        elif i<=8:
            hist['semantic']['res3'].append(histogram[i])
        else:
            hist['semantic']['res2'].append(histogram[i])
    for i in range(12,24):
        if i-12<=5:
            hist['instance']['res5'].append(histogram[i])
        elif i-12<=8:
            hist['instance']['res3'].append(histogram[i])
        else:
            hist['instance']['res2'].append(histogram[i])
    return hist

def init_list(max_list):
    for key in max_list.keys():
        for res in max_list[key]:
            if res=='res5':
                max_list[key][res] = [[] for _ in range(6)]
            else:
                max_list[key][res] = [[] for _ in range(3)]

def get_max(histogram):
    for key in histogram.keys():
        for res in histogram[key]:
            for i, arr in enumerate(histogram[key][res]):
                if torch.is_tensor(arr):
                    max_list[key][res][i].append(torch.max(arr).item())
                else:
                    max_list[key][res][i].append(arr.max())

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
        q_max_list.append(q_max)
        if i//2 in [4,6,8,13,15,17]:
            q_max_list.append(max(q_max_list[temp-i//2:]))
            temp = i//2
    
    return q_max_list

#get percentile of q-scale
def get_percentile(q_max_dict, max_list):
    percentile = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}
    for key in max_list:
        for res in max_list[key]:
            for idx, max_dist in enumerate(max_list[key][res]):
                p = percentileofscore(max_dist, q_max_dict[key][res][idx])
                if p >= 100:
                    percentile[key][res].append((p,q_max_dict[key][res][idx]/max(max_dist)))
                else:
                    percentile[key][res].append(p)
    return percentile

mean = np.repeat(np.array([[[128]]]),3,axis=0)
std = np.repeat(np.array([[[128]]]),3,axis=0)
sess = ort.InferenceSession("/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic_hist.onnx")
model_quant = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal30_seed1000.onnx"


args = get_parser().parse_args()
cfg = setup_cfg(args)
model = build_model(cfg)
model.eval()
dataset = glob.glob("datasets/cityscapes/leftImg8bit/val/*/*.png")

#initialize
init_list(max_list)


for idx, file in enumerate(dataset):
    print("%dth file: "%idx,file)

    img = np.array(Image.open(file))
    #preprocess
    img = img[:,:,::-1]
    img = img.transpose(2,0,1)
    img = (img - mean) / std
    img = img.astype(np.float32)

    inputs = {"image": torch.as_tensor(img), "height": 1024, "width": 2048}
    #get network outputs: sem_seg,center,offset,histograms
    model.network_only=True
    model.onnx = True
#    outputs1 = model([torch.as_tensor(img).to(model.device)])

    outputs2 = sess.run(None, {'image':img})
#    print(torch.max(outputs1[0]))
#    print(outputs2[0].max())
    histogram = create_hist_dict(outputs2[3:])
    get_max(histogram)

#print_hist(max_list)

q_max_list = get_q_max(model_quant)
q_max_dict = create_hist_dict(q_max_list)
percentile = get_percentile(q_max_dict, max_list)
print_hist(max_list)
print_hist(q_max_dict)
print_hist(percentile)
