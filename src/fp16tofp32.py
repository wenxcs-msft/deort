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
            name= init.name,
            data_type=TensorProto.FLOAT,
            dims= init.dims,
            vals= fp32_data)


if __name__ == "__main__":
    pass