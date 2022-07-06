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
from detectron2.checkpoint import DetectionCheckpointer
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt

def print_hist(max_list,tab=""):
    tab += "\t"
    if type(max_list) is dict:
        for key in max_list.keys():
            print("%s%s:"%(tab,key))
            print_hist(max_list[key], tab)
    else:
        for i, minmax_hist in enumerate(max_list):
            min_elt = min(minmax_hist, key=lambda x:x[0])[0]
            max_elt = max(minmax_hist, key=lambda x:x[1])[1]
            print("%s%dth min:%f max:%f"%(tab,i,min_elt, max_elt))

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

def init_list():
    dist = {'semantic':{'res5':[],'res3':[],'res2':[]},'instance':{'res5':[],'res3':[],'res2':[]}}
    for task in dist:
        for res in dist[task]:
            if res=='res5':
                dist[task][res] = [[] for _ in range(6)]
            else:
                dist[task][res] = [[] for _ in range(3)]
    return dist

def get_minmax(histogram):
    for key in histogram:
        for res in histogram[key]:
            for i, arr in enumerate(histogram[key][res]):
                if torch.is_tensor(arr):
                    max_elt = torch.max(arr).item()
                    min_elt = torch.min(arr).item()
                else:
                    max_elt = arr.max()
                    min_elt = arr.min()
                minmax_list[key][res][i].append((min_elt,max_elt))


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
                    percentile[key][res].append((p, min(max_dist), max(max_dist)))
    return percentile

def save_percentile(dist, p):
    from matplotlib import mlab
    import matplotlib.pyplot as plt

    perc = mlab.prctile(dist, p=p)

    plt.plot(dist)
    # Place red dots on the percentiles
    plt.plot((len(dist)-1) * p/100., perc, 'ro')
    plt.savefig("./debugging/max_dist.png")

    import sys
    sys.exit()

# accumulate each histogram over validatio dataset
# we divide histogram into two parts. values less than 10, larger than 10
# since almost values are close to zero, normalize histogram using log.
def accumulate_histogram(histogram):
    for key in histogram.keys():
        for res in histogram[key]:
            for i, arr in enumerate(histogram[key][res]):
                if not torch.is_tensor(arr):
                    arr = torch.from_numpy(arr)
                #ignore zero elements.
                #and focus on elts < 10
                arr = arr[arr.nonzero(as_tuple=True)]
                arr = arr[arr > 10]
                hist,_ = torch.histogram(arr, bins=600, range=(10,610))
                #normalize histogram using log
                hist = torch.log(torch.add(hist,1))

                hist = hist.detach().cpu().numpy()
                if acc_dist[key][res][i]==[]:
                    acc_dist[key][res][i] = hist
                else:
                    acc_dist[key][res][i] += hist

def plot_hist(hist):
    for task in hist:
        for res in hist[task]:
            n = len(hist[task][res])
            # plot 
            fig,ax = plt.subplots(n,1)
            for idx in range(n):
                ax[idx].bar(np.arange(10,610),hist[task][res][idx], align='edge', width=0.1)
            plt.savefig("debugging/histogram/test/%s:%s_tail"%(task,res))

# get histogram for semantic segmentation results, center results, offset results
# center results has pixel value representing probability of the pixel being center
# since we only consider center with probability > threshold = 0.1
# ignore values less than threshold
def get_output(outputs1, outputs2):
    fp32_sem,_ = torch.histogram(outputs1[0].detach().cpu(), bins=120, range=(-30, 30))
    c0 = outputs1[1].detach().cpu()
    fp32_center,_ = torch.histogram(c0[c0>0.1], bins=100, range=(0,1))
    fp32_offset,_ = torch.histogram(outputs1[2].detach().cpu(), bins=600, range=(-300,300))

    uint8_sem,_ = torch.histogram(torch.from_numpy(outputs2[0]), bins=120, range=(-30, 30))
    c1 = torch.from_numpy(outputs2[1])
    uint8_center,_ = torch.histogram(c1[c1>0.1], bins=100, range=(0,1))
    uint8_offset,_ = torch.histogram(torch.from_numpy(outputs2[2]), bins=600, range=(-300,300))

    if idx == 0:
        fp32 = [fp32_sem,fp32_center,fp32_offset]
        uint8 = [uint8_sem,uint8_center,uint8_offset]
    else:
        fp32[0] += fp32_sem
        fp32[1] += fp32_center
        fp32[2] += fp32_offset

        uint8[0] += uint8_sem
        uint8[1] += uint8_center
        uint8[2] += uint8_offset
    return fp32, uint8

def plot_output(fp32, uint8):
    bins = [
        np.arange(-60,60)*0.5,
        np.arange(1,100)*0.01,
        np.arange(-300,300)
    ]

    fig,ax = plt.subplots(3,1)
    for idx in range(3):
        ax[idx].bar(bins[idx], fp32[idx].numpy())
    plt.savefig("debugging/histogram/outputs/fp32")

    fig,ax = plt.subplots(3,1)
    for idx in range(3):
        ax[idx].bar(bins[idx], uint8[idx].numpy())
    plt.savefig("debugging/histogram/outputs/uint8")

#dataset
dataset = glob.glob("datasets/cityscapes/leftImg8bit/val/*/*.png")

#load model, session
'''
args = get_parser().parse_args()
cfg = setup_cfg(args)
model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
model.eval()
'''
sess = ort.InferenceSession("/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic_hist.onnx")
#sess = ort.InferenceSession("/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/debugging/test.onnx")
model_quant = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal30_seed1000.onnx"

#initialize
torch.manual_seed(1000)
#max_list = init_list()
minmax_list = init_list()
acc_dist = init_list()

mean = np.repeat(np.array([[[128]]]),3,axis=0)
std = np.repeat(np.array([[[128]]]),3,axis=0)

for idx, file in enumerate(dataset):
    print("%dth file: "%idx,file)
    img = np.array(Image.open(file))

    #preprocess
    img = img[:,:,::-1]
    img = img.transpose(2,0,1)
    img1 = img
    img = (img - mean) / std
    img = img.astype(np.float32)
    inputs = {"image": torch.as_tensor(img), "height": 1024, "width": 2048}
    
    #get network outputs: sem_seg,center,offset,histograms
    #model.network_only=True
    #model.onnx = True
    #outputs1 = model([torch.as_tensor(img).to(model.device)])
    outputs2 = sess.run(None, {'image':img})

    histogram = create_hist_dict(outputs2[3:])
    accumulate_histogram(histogram)
    #get_minmax(histogram)

plot_hist(acc_dist)
#print_hist(minmax_list)

#q_max_list = get_q_max(model_quant)
#q_max_dict = create_hist_dict(q_max_list)
#percentile = get_percentile(q_max_dict, max_list)
#print_hist(max_list)
#print_hist(q_max_dict)
#print_hist(percentile)
#dist = q_max_dict['semantic']['res5'][0]
#p = percentile['semantic']['res5'][0]
#save_percentile(dist, p)