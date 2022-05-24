import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto, numpy_helper
import numpy as np

in_onnx_file = "/workspaces/deort/onnx/fp16.onnx"
out_onnx_file = "/workspaces/deort/onnx/fp32.onnx"

fp16_model = onnx.load_model(in_onnx_file)

# change all initializer from fp16 to fp32
fp16_init_name = set()
fp32_init = []
for init in fp16_model.graph.initializer:
    if init.data_type == TensorProto.FLOAT16:
        print(init.name, init.dims)
        fp16_init_name.add(init.name)
        fp32_data = numpy_helper.to_array(init)
        if len(init.dims) == 0:
            fp32_data = [fp32_data.tolist()]
        fp32_tensor_init = onnx.helper.make_tensor(
            name= init.name+"f32",
            data_type=TensorProto.FLOAT,
            dims= init.dims,
            vals= fp32_data)
        fp32_init.append(fp32_tensor_init)

for init in fp16_model.graph.initializer:
    if init.name in fp16_init_name:
        fp16_model.graph.initializer.remove(init)

for init in fp32_init:
    fp16_model.graph.initializer.append(init)

# Remove cast to f32, f16
remove_node_list = []
remove_tensor_map = dict()
for node in fp16_model.graph.node:
    if node.op_type == "Cast":
        for attr in node.attribute:
            if attr.i == TensorProto.FLOAT:
                remove_node_list.append(node)
                remove_tensor_map[node.output[0]] = node.input[0]

for node in remove_node_list:
    fp16_model.graph.node.remove(node)

for node in fp16_model.graph.node:
    for i in range(0, len(node.input)):
        if node.input[i] in remove_tensor_map.keys():
            node.input[i] = remove_tensor_map[node.input[i]]
    
    for i in range(0, len(node.output)):
        if node.output[i] == "output_0_float16":
            node.output[i] = "output_0"

for node in fp16_model.graph.node:
    if node.op_type == "Cast":
        for attr in node.attribute:
            if attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
    
    for i in range(0, len(node.input)):
        if node.input[i] in fp16_init_name:
            node.input[i] = node.input[i] + "f32"

onnx.save_model(fp16_model, out_onnx_file)

if __name__ == "__main__":
    pass 