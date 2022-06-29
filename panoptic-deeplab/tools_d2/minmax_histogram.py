import glob
from PIL import Image
import numpy as np
import onnxruntime as ort
from detectron2.modeling import build_model
from demo import setup_cfg, get_parser
import torch
from random import sample

torch.manual_seed(100)

mean = np.repeat(np.array([[[128]]]),3,axis=0)
std = np.repeat(np.array([[[128]]]),3,axis=0)
sess = ort.InferenceSession("/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic.onnx")


args = get_parser().parse_args()
cfg = setup_cfg(args)
model = build_model(cfg)
model.eval()
dataset = glob.glob("datasets/cityscapes_origin/leftImg8bit/train/*/*.png")

max_list = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}

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
            for i, tensor in enumerate(histogram[key][res]):
                max_list[key][res][i].append(torch.max(tensor).item())

#initialize
init_list(max_list)


for idx, file in enumerate(sample(dataset,1)):
    print(file)
    if idx > 20:
        break
    img = np.array(Image.open(file))
    #preprocess
    img = img[:,:,::-1]
    img = img.transpose(2,0,1)
    img = (img - mean) / std
    img = img.astype(np.float32)
    inputs = {"image": torch.as_tensor(img), "height": 1024, "width": 2048}
    #get network outputs: sem_seg,center,offset,histograms
    model.network_only=True
    outputs = model([inputs])
    histogram = outputs[3]
    get_max(histogram)

def print_hist(max_list,tab=""):
    tab += "\t"
    if type(max_list) is dict:
        for key in max_list.keys():
            print("%s%s:"%(tab,key))
            print_hist(max_list[key], tab)
    else:
        for i, max_hist in enumerate(max_list):
            print("%s%dth max:"%(tab,i),max_hist)

print_hist(max_list)

