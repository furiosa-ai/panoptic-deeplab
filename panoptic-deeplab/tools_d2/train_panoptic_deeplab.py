#!/usr/bin/env python3
#
# Modified by Bowen Cheng
#
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import _init_paths
import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

'''
ADD modules
'''
import d2
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, CalibrationMethod

from random import sample
from itertools import islice
from furiosa.quantizer.frontend.onnx import optimize_model, quantize, post_training_quantize
import sys
import glob
from PIL import Image

import numpy as np
import cv2
import sys

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        #mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=[T.RandomFlip()])
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        
        sess = None
        run_onnx = True
        patching = False
        cali = True
        qdq = False 

        #get pixel mean and std
        mean = np.repeat(np.array([[[128]]]),3,axis=0)
        std = np.repeat(np.array([[[128]]]),3,axis=0)

        if run_onnx == True:
            model_path = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic.onnx"
            #model_path = "/root/ljh726/PanopticDeepLab/warboy/q_concat/panoptic-int8-cal100-FAKE.onnx"
            print(ort.get_device())
            sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
        if cali:
            dataloader = glob.glob("datasets/cityscapes/leftImg8bit/val/*/*.png")
            sample_dataloader = sample(dataloader, k=100)
            model = onnx.load_model(model_path)
           
            onnx_model_quantized = post_training_quantize(
                model,
                ({"image":preprocess_image(Image.open(path), mean, std).astype(np.float32)} for path in sample_dataloader),
            )

            onnx.save_model(onnx_model_quantized, "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812/panoptic-sdk-test.onnx")
            sys.exit()
        if qdq:
            model_fp32 = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic_add_name.onnx"
            model_quant = "/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic13-ort-entropy-test.onnx"
            data_path = "datasets/cityscapes/leftImg8bit/train/*/*.png"
            cali_data_reader = CityscapesDataReader(data_path, mean, std)
            quantize_static(model_fp32, model_quant, cali_data_reader, calibrate_method=CalibrationMethod.Entropy, per_channel=True)
            print("quantize done")
            sys.exit()

        #iter_l1loss(model0, sess, dataloader)

        model = Trainer.build_model(cfg)
        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
        )
        with torch.no_grad():
            res = Trainer.test(cfg, model, sess=sess, patching = patching)

        return res
        

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def preprocess_image(image, mean, std):
    #resize image
    image = np.array(image).astype(dtype=np.float32)
    #image = cv2.resize(image, dsize=(1024,512), interpolation=cv2.INTER_AREA)
    #RGB-> BGR
    image = image[:,:,::-1]
    image = image.transpose(2,0,1)
    #normalize
    image = (image - mean) / std
    #expand dim
    #image = np.expand_dims(image, axis=0)
    return image

class CityscapesDataReader(CalibrationDataReader):
    def __init__(self, data_path, mean, std) -> None:
        super().__init__()
        self.preprocess_flag = True
        self.datalist = glob.glob(data_path)
        self.sample_datalist = sample(self.datalist, k=3)
        self.mean = mean
        self.std = std
        #self.datasize = 50
        self.data_dicts = iter([{'image': preprocess_image(Image.open(img_path), self.mean, self.std).astype(np.float32)} for img_path in self.sample_datalist])
        self.counter = 0
    def get_next(self) -> dict:
        self.counter += 1
        print(self.counter,"th sample")
        return next(self.data_dicts, None)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )





################################################################################################


def iter_l1loss(model, sess, dataloader):
    for idx, data in enumerate(dataloader):
        if idx > 10:
            break
        data = resize_input(data)
        print(idx," th loss")
        compute_l1loss(model, sess, data)
    sys.exit()

def compute_l1loss(model, sess, data):
    import torch.nn as nn
    l1_loss = nn.L1Loss()
    model.eval()
    model.network_only = True
    sem_seg1, center1, offset1 = model(data)
    sem_seg2, center2, offset2 = sess.run(None, {'image':data[0]['image'].numpy()})

    sem_seg2 = torch.from_numpy(sem_seg2).to('cuda')
    center2 = torch.from_numpy(center2).to('cuda')
    offset2 = torch.from_numpy(offset2).to('cuda')
    loss1 = l1_loss(sem_seg1, sem_seg2)
    loss2 = l1_loss(center1, center2)
    loss3 = l1_loss(offset1, offset2)

    print("SEM SEG norms", torch.mean(torch.abs(sem_seg1)).item(), torch.mean(torch.abs(sem_seg2)).item())
    print("center norms", torch.mean(torch.abs(center1)).item(), torch.mean(torch.abs(center2)).item())
    print("offset norms", torch.mean(torch.abs(offset1)).item(), torch.mean(torch.abs(offset2)).item())

    print("SEM SEG error", loss1)
    print("CENTER error", loss2)
    print("OFFSET error", loss3)

def preprocess_input(inputs, mean, std):
    inputs = resize_input(inputs)
    inputs[0]['image'] = (inputs[0]['image'] - mean) / std
    return inputs

def resize_input(inputs):
    #resize input images into 512*1024
    image = inputs[0]['image']
    image = np.array(image).astype(dtype=np.float32)
    image = image.transpose(1,2,0)
    image = cv2.resize(image, dsize=(1024,512), interpolation=cv2.INTER_AREA)
    #BGR-> RGB
    image = image[:,:,::-1]
    image = image.transpose(2,0,1)
    inputs[0]['image'] = torch.from_numpy(image.copy())
    return inputs