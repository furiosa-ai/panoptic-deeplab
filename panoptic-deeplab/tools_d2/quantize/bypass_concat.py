import torch
from detectron2.modeling import build_model
from demo import setup_cfg, get_parser
from detectron2.checkpoint import DetectionCheckpointer

model_path = '/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/experiments/lr_1e-3_batch32_4812/model_0059999.pth'

args = get_parser().parse_args()
cfg = setup_cfg(args)
model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
model.eval()

name_list = ['ins_embed_head.decoder.res2.project_conv.weight',
            'ins_embed_head.decoder.res3.fuse_conv.pointwise.weight',
            'ins_embed_head.decoder.res3.project_conv.weight',
            'ins_embed_head.decoder.res5.project_conv.convs.0.weight',
            'ins_embed_head.decoder.res5.project_conv.convs.1.pointwise.weight',
            'ins_embed_head.decoder.res5.project_conv.convs.2.pointwise.weight',
            'ins_embed_head.decoder.res5.project_conv.convs.3.pointwise.weight',
            'ins_embed_head.decoder.res5.project_conv.convs.4.1.weight',

            'sem_seg_head.decoder.res2.project_conv.weight',
            'sem_seg_head.decoder.res3.fuse_conv.pointwise.weight',
            'sem_seg_head.decoder.res3.project_conv.weight',
            'sem_seg_head.decoder.res5.project_conv.convs.0.weight',
            'sem_seg_head.decoder.res5.project_conv.convs.1.pointwise.weight',
            'sem_seg_head.decoder.res5.project_conv.convs.2.pointwise.weight',
            'sem_seg_head.decoder.res5.project_conv.convs.3.pointwise.weight',
            'sem_seg_head.decoder.res5.project_conv.convs.4.1.weight',

            ]

#model.load_state_dict(torch.load(model_path))
# concat input conv layer를 다른 conv로 대체
# 다른 conv는 zeros_like 로 initialize, 기존 conv의 weight를 붙인다.
# new_conv.weight[:c,...] = conv.weight
print(model.ins_embed_head.decoder.res2.project_conv)
layer_weight = model.ins_embed_head.decoder.res2.project_conv.weight
print(layer_weight.size())
conv1 = nn.Conv2d()
conv2 = nn.Conv2d()
'''
for name, param in model.named_parameters():
    if param.requires_grad:
        if name in name_list:
            print(name)
            print(param.size())
'''