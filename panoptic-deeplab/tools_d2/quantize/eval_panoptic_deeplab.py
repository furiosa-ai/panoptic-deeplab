from pickle import FALSE
from detectron2.modeling.meta_arch.build import build_model
import tensorrt as trt

import helper
from helper import infer_helper
import glob
from PIL import Image
import numpy as np
import argparse

import onnxruntime as ort
from train_panoptic_deeplab import Trainer, setup
import torch

import argparse
from typing import List, Union

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)

from detectron2.checkpoint import DetectionCheckpointer
import matplotlib.pyplot as plt
from panopticapi.utils import id2rgb


# ----- NOTE : you can modify here ------------------------ #
engine_path = '/root/ljh726/PanopticDeepLab/warboy/trt_model/panoptic.engine'
DYNAMIC_SHAPE = False
test_files = glob.glob( '/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/datasets/cityscapes/leftImg8bit/val/*/*.png')

def preprocessor( inputs ) :
    img = inputs[0]['image']

    #torch to numpy [C,H,W]
    img = np.array(img).astype(dtype=np.float32)
    #RGB to BGR
    #img = img[::-1,:,:]
    #normalize img
    mean = np.repeat(np.array([[[128]]]),3,axis=0)
    std = np.repeat(np.array([[[128]]]),3,axis=0)
    img = (img - mean) / std
    img = np.array(img).astype(dtype=np.float32)
    img = img[np.newaxis]

    return img

def get_imagelist(model, inputs):
    size_divisibility = (
    model.size_divisibility
    if model.size_divisibility > 0
    else model.backbone.size_divisibility
    )
    from detectron2.structures import BitMasks, ImageList, Instances
    images = ImageList.from_tensors([inputs[0]['image']], size_divisibility)

    return images

def postprocessor( outputs, model, inputs) :
    sem_seg_results, center_results, offset_results = outputs

    #TensorRT output has flatten format
    #reshape outputs
    sem_seg_results = sem_seg_results.reshape(1,19,1024,2048).copy()
    center_results = center_results.reshape(1,1,1024,2048).copy()
    offset_results = offset_results.reshape(1,2,1024,2048).copy()

    sem_seg_results = torch.from_numpy(sem_seg_results).to('cuda')
    center_results = torch.from_numpy(center_results).to('cuda')
    offset_results = torch.from_numpy(offset_results).to('cuda')

    imagelist = get_imagelist(model, inputs)
    outputs = model.post_process(sem_seg_results,
                                center_results,
                                offset_results, 
                                [{'height':inputs[0]['height'], 'width':inputs[0]['width']}],
                                imagelist
                                )
    return outputs


class Evaluator(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, dataset_name):
        evaluator_list = []
        evaluator_list.append(COCOPanopticEvaluator(dataset_name))
        evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def inference(cls, model, infer_helper, dataloader, evaluator: DatasetEvaluators):
        #Initialize evaluator
        evaluator.reset()

        #evaluate
        for idx, inputs in enumerate(dataloader):
            print("idx:%d"%idx)
            #TensorRT inference
            image = preprocessor(inputs)
            outputs = infer_helper.infer(image)
            #Pytorch fp32 model inferene
            #model.network_only = True
            #sem, cent, off = model(inputs)

            #plot semantic segmentation results
            '''
            r_trt = outputs[0].reshape(19,1024,2048).argmax(axis=0)
            r_trt = (np.arange(19) == r_trt[...,None]-1).astype(int)
            r_trt = np.matmul(r_trt, np.array(labels))
            im_trt = Image.fromarray(r_trt.astype(np.uint8))
            im_trt.save("/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/quantize/EDSR-PyTorch/tensorrt_example/sem_seg/int8.png")

            r = sem[0].argmax(dim=0).detach().cpu().numpy()
            r = (np.arange(19) == r[...,None]-1).astype(int)
            r = np.matmul(r, np.array(labels))
            im = Image.fromarray(r.astype(np.uint8))
            im.save('/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/quantize/EDSR-PyTorch/tensorrt_example/sem_seg/fp32.png')
            break
            '''

            #plot histogram
            '''
            fig, ax = plt.subplots(2,1)
            bins = list(range(-30,30))
            ax[0].hist(outputs, bins=bins)
            ax[1].hist([sem.detach().cpu().numpy().flatten(),
                        cent.detach().cpu().numpy().flatten(),
                        off.detach().cpu().numpy().flatten()], bins=bins)
            plt.savefig("/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/quantize/EDSR-PyTorch/tensorrt_example/histogram/hist_entropy_bgr.png")
            break
            '''

            outputs = postprocessor(outputs, model, inputs)


            evaluator.process(inputs, outputs)
        results = evaluator.evaluate()
        return results


    @classmethod
    def test(cls, cfg, model, infer_helper, dataset_name):
        print(type(model))
        dataloader = super().build_test_loader(cfg, dataset_name)
        evaluator = cls.build_evaluator(dataset_name)
        results = cls.inference(model, infer_helper, dataloader, evaluator)
        return results

# --------------------------------------------------------- #


def main() :
    logger = trt.Logger(trt.Logger.WARNING)
    cfg = setup(args)
    dataset_name = 'cityscapes_fine_panoptic_val'
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)    
    model.eval()
    infer_helper = helper.infer_helper( engine_path, logger, dynamic_shape=DYNAMIC_SHAPE )

    #infer_helper.engine.run( test_files[0] )   # The first batch run takes longer because of loading engine. we need to intialize before measuring timing. ( QuickStart Guide 4.5 )
    
    with torch.no_grad():
        res = Evaluator.test(cfg, model, infer_helper, dataset_name)

    infer_helper = None

## id to rgb color
labels = [[111, 74,  0],[81,  0, 81],[128, 64,128],[244, 35,232],[250,170,160],[230,150,140],[ 70, 70, 70],[102,102,156],[190,153,153],[180,165,180],
[150,100,100],[150,120, 90],[153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],[70,130,180],[220, 20, 60] ]


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main()