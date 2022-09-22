import onnx
from onnx import helper, AttributeProto, TensorProto, TensorShapeProto, GraphProto, numpy_helper
import numpy as np

shape = {"clip_input":{"batch":1,"channels":3,"height":224,"width":224},"images":{"batch":1,"channels":512,"height":512,"width":3}}

in_onnx_file = "model.onnx"
out_onnx_file = "model.fixed_dim.onnx"
input_model : GraphProto = onnx.load_model(in_onnx_file)

for input_tensor in input_model.graph.input:
    for dim_i in range(len(input_tensor.type.tensor_type.shape.dim)):
        dim_param = input_tensor.type.tensor_type.shape.dim[dim_i].dim_param
        new_dim_val = -1
        if dim_param in shape:
            new_dim_val = shape[dim_param]
        if input_tensor.name in shape and dim_param in shape[input_tensor.name]:
            new_dim_val = shape[input_tensor.name][dim_param]
        input_tensor.type.tensor_type.shape.dim[dim_i].dim_value = new_dim_val
    print(input_tensor)

onnx.save_model(input_model, out_onnx_file)
