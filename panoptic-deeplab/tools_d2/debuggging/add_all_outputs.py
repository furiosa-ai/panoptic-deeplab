import onnx

quant_sdk = '/root/ljh726/PanopticDeepLab/warboy/xception65_dsconv_4812_1024_2048/panoptic-int8-cal50.onnx'
 

def add_mid_nodes(path):
    model = onnx.load(path)
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        print(idx, node.name)
        value_info_protos.append(node)

    print(len(value_info_protos))
    model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
    onnx.checker.check_model(model)
    onnx.save(model, path[:-5]+'-test.onnx')

def add_specific_node(path):
    model = onnx.load(path)
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name == 'input.7_quantized' or node.name== 'input.11_dequantized':
            print(node.name)
            value_info_protos.append(node)

    print(idx)
    print(len(value_info_protos))
    model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
    onnx.checker.check_model(model)
    onnx.save(model, path[:-5]+'-test-2output.onnx')






#add_mid_nodes(quant_sdk)
add_specific_node(quant_sdk)