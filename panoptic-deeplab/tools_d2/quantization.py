from statistics import mode
from PIL import Image
import numpy as np
import glob 
import argparse

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, CalibrationMethod

from random import sample
from itertools import islice
from furiosa.quantizer.frontend.onnx import optimize_model, quantize, post_training_quantize
import sys

from detectron2.modeling import build_model
from .demo import setup_cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--quant",
        help="A task you want to do",
        choices=['sdk', 'ort']
    )
    parser.add_argument(
        "--dir",
        default="/root/ljh726/PanopticDeepLab/warboy/",
        help="A directory to save or load model.",
    )
    parser.add_argument(
        "--data",
        default="datasets/cityscapes/leftImg8bit/train/*/*.png",
        help="A file or directory to load calibration data.",
    )
    parser.add_argument(
        "--cal",
        type=int,
        default=50,
        help="A number of calibration data used to quantize model.",
    )
    return parser

def sdk_ptq(model_fp32, model_quant, data_path, cal_num):
    dataloader = glob.glob(data_path)
    sample_dataloader = sample(dataloader, k=cal_num)
    print("Loading..")
    model = onnx.load_model(model_fp32)
    print("FURISOA post-training-quantize...")
    onnx_model_quantized = post_training_quantize(
        model,
        ({"image":preprocess_image(Image.open(path), mean, std).astype(np.float32)} for path in sample_dataloader),
    )
    print("end ptq")
    onnx.save_model(onnx_model_quantized, model_quant)

def ort_ptq(model_fp32, model_quant, data_path, mean, std, cal_num):
    cali_data_reader = CityscapesDataReader(data_path, mean, std, cal_num)
    print("onnxruntime post-training-quantize...")
    quantize_static(model_fp32, model_quant, cali_data_reader, calibrate_method=CalibrationMethod.MinMax, per_channel=True)
    print("end ptq")

def preprocess_image(image, mean, std):
    image = np.array(image).astype(dtype=np.float32)
    #resize image
    #image = cv2.resize(image, dsize=(2048,1024), interpolation=cv2.INTER_AREA)
    #RGB-> BGR
    image = image[:,:,::-1]
    image = image.transpose(2,0,1)
    #normalize
    image = (image - mean.numpy()) / std.numpy()
    #expand dim
    #image = np.expand_dims(image, axis=0)
    return image

class CityscapesDataReader(CalibrationDataReader):
    def __init__(self, data_path, mean, std, cal_num) -> None:
        super().__init__()
        self.preprocess_flag = True
        self.datalist = glob.glob(data_path)
        self.sample_datalist = sample(self.datalist, k=cal_num)
        self.mean = mean
        self.std = std
        #self.datasize = 50
        self.data_dicts = iter([{'image': preprocess_image(Image.open(img_path), self.mean, self.std).astype(np.float32)} for img_path in self.sample_datalist])
        self.counter = 0
    def get_next(self) -> dict:
        self.counter += 1
        print(self.counter,"th sample")
        return next(self.data_dicts, None)

if __name__=='__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    model = build_model(cfg)

    #mean, std
    mean = model.pixel_mean.cpu().detach()
    std = model.pixel_std.cpu().detach()

    model_dir = args.dir + "xception65_dsconv_4812_1024_2048/"
    model_fp32 = model_dir + "panoptic.onnx"

    if args.quant == 'sdk':
        model_quant = model_fp32[:-5] + "-int8-cal" + args.cal + ".onnx"
        sdk_ptq(model_fp32, model_quant, args.data, args.cal)
    else:
        model_quant = model_fp32[:-5] + "-qdq-cal" + args.cal + ".onnx"
        ort_ptq(model_fp32, model_quant, args.data, args.cal)
