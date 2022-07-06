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
import glob
import numpy as np
from detectron2.modeling.meta_arch.build import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os

def path_manager(name):
    extn = ".onnx"
    dir = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/"
    return dir+name+extn

def load_model(cfg=None, model_path=None):
    if model_path is not None:
        #open session for onnx model
        return ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        #load pytorch model
        return Trainer.build_model(cfg)
        


# ----- NOTE : you can modify here ------------------------ #
engine_path = '/root/ljh726/PanopticDeepLab/warboy/trt_model/panoptic.engine'
DYNAMIC_SHAPE = False
test_files = glob.glob( '/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/datasets/cityscapes/leftImg8bit/val/*/*.png')

def preprocessor( inputs ) :
    img = inputs[0]['image']

    #torch to numpy [C,H,W]
    img = np.array(img).astype(dtype=np.float32)
    #RGB to BGR
    img = img[::-1,:,:]
    #normalize img
    mean = np.repeat(np.array([[[128]]]),3,axis=0)
    std = np.repeat(np.array([[[128]]]),3,axis=0)
    img = (img - mean) / std
    img = np.array(img).astype(dtype=np.float32)

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

def postprocessor( sem_seg_results, center_results, offset_results, model, inputs) :
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
    def build_evaluator(cls, cfg, dataset_name):
        output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_dir))
        evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def inference(cls, model, datasize,dataloader, evaluator: DatasetEvaluators, sess=None):
        evaluator.reset()
        for idx, inputs in enumerate(dataloader):
            print("idx:%d"%idx)
            if sess is not None:
                image = preprocessor(inputs)
                sem_seg_results, center_results, offset_results = sess.run(None, {'image':image})
                outputs = postprocessor(sem_seg_results, center_results, offset_results, model, inputs)
            else:
                outputs = model(inputs)

            evaluator.process(inputs, outputs)
        #unless len(gt_json)==len(dataset), no image id in annotations error occured
        results = evaluator.evaluate()
        return results

    @classmethod
    def test(cls, cfg, model, dataset_name, datasize,sess=None):
        print(type(model))
        print(type(dataset_name))
        print(type(datasize))
        dataloader = super().build_test_loader(cfg, dataset_name)
        evaluator = cls.build_evaluator(cfg, dataset_name)
        results = cls.inference(model, datasize, dataloader, evaluator, sess=sess)
        return results

# --------------------------------------------------------- #


def main() :
    cfg = setup(args)
    dataset_name = 'cityscapes_fine_panoptic_val'
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)    
    model.eval()
    

    model_path = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic.onnx"
    sess = None
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    with torch.no_grad():
        res = Evaluator.test(cfg, model, dataset_name, 3, sess=sess)

    return res

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main()

