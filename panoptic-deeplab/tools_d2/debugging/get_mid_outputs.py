import onnx

model = onnx.load('/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal100-val.onnx')
inter_layers = ['input.27_dequantized'] # output tensor names
value_info_protos = []
shape_info = onnx.shape_inference.infer_shapes(model)
for idx, node in enumerate(shape_info.graph.value_info):
    if node.name in inter_layers:
        print(idx, node)
        value_info_protos.append(node)
assert len(value_info_protos) == len(inter_layers)
model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
onnx.checker.check_model(model)
onnx.save(model, '/root/ljh726/PanopticDeepLab/panoptic-deeplab/tools_d2/debugging/test.onnx')
