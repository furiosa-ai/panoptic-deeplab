from detectron2.modeling import build_model
from demo import setup_cfg
import torch
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    
    parser.add_argument(
        "--output",
        help="A file or directory to save onnx model. ",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__=="__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    model = build_model(cfg)

    print("Start to build onnx")
    model.onnx = True
    dir_path = "/root/ljh726/PanopticDeepLab/warboy/"
    model_ver = "xception65_dsconv_4812_1024_2048/"

    #INPUT
    inputs = torch.randn(1, 3, 512, 1024, device= 'cuda')
    #EXPORT
    torch.onnx.export(model, args=inputs, f=dir_path + model_ver + "panoptic.onnx", export_params=True,opset_version=12, input_names=['image'],output_names = ['output'])
    print("Onnx model has been built")