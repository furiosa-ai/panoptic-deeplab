from detectron2.modeling import build_model
from demo import setup_cfg
import torch
import argparse
from detectron2.checkpoint import DetectionCheckpointer

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
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
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
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    print("Start to build onnx")
    model.onnx = True
    dir_path = "/root/ljh726/PanopticDeepLab/warboy/"
    model_ver = "xception65_dsconv_4812_1024_2048/"

    #INPUT
    inputs = torch.randn(3, 1024, 2048, device= 'cuda')
    #EXPORT
    torch.onnx.export(model, args=[inputs], f=dir_path + model_ver + "panoptic_add_name.onnx", export_params=True, opset_version=13, input_names=['image'],output_names = ['sem_seg','center','offset'])
    print("Onnx model has been built")