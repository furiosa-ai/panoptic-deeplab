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

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.model is None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

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

class FastTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, dataset_name, output_folder):
        evaluator_list = []
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def inference(cls, model, datasize,dataloader, evaluator: DatasetEvaluators):
        for idx, inputs in enumerate(dataloader):
            if idx < datasize:
                outputs = model(inputs)
                evaluator.process(outputs, inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        results = evaluator.evaluate()
        return results

    @classmethod
    def test(cls, model, dataset_name, datasize):
        print(type(model))
        print(type(dataset_name))
        print(type(datasize))
        dataloader = super().build_test_loader(cfg, dataset_name)
        evaluator = super().build_evaluator(cfg, dataset_name)
        results = cls.inference(model, datasize, dataloader, evaluator)
        return results



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Fast Evaluation for fp32 or quantized model")
    parser.add_argument(
        "--config-file",
        default=None,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="A file name to evaluate model",
    )
    args = parser.parse_args()
    cfg = setup(args)

    model_path = path_manager(args.model)
    dataset_name = "cityscapes_fine_panoptic_val"

    model = load_model(cfg, model_path)
    print(cfg)
    print(model)
    
    with torch.no_grad():
        res = FastTrainer.test(cfg, model, dataset_name, 30)

#########
# we need small train, val dataset and prediction.json