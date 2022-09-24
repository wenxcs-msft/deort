#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0

import os, sys, glob
from typing import IO, Any, Dict, List, Sequence, Union
import numpy as np
import onnx
import torch
from onnx import AttributeProto, defs, load, ModelProto, NodeProto, TypeProto, numpy_helper
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
from onnx.backend.base import Backend


os.environ["PATH"] = os.path.abspath("/home/affinity/nnfusion/nnfusion/build/src/tools/nnfusion/") + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath("/home/affinity/nnfusion/nnfusion/src/python"))
import nnfusion
from nnfusion.executor import Executor
from nnfusion.session import generate_sample, codegen, modify_nnfusion_rt, build
from nnfusion.data_format import cast_pytorch_tensor, cast_hlsl_tensor, HLSLTensor

workdir = "nnfusion_work"
arg_str = "-f onnx -fmulti_shape=false -fdefault_device=CUDA -fhlsl_codegen_type=cpp -fantares_mode=true -fblockfusion_level=0 -fkernel_fusion_level=0 -fantares_codegen_server=127.0.0.1:8880 -fkernel_tuning_steps=0 -ffold_where=1 -fsymbolic=1 -fort_folding=0 -fsplit_softmax=1 -fhost_entry=0 -fir_based_fusion=1 -fextern_result_memory=1"

def build_model(model_path) -> Executor:
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    codegen(model_path, arg_str, workdir)
    rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
    modify_nnfusion_rt(rt_dir)
    build(rt_dir)
    return Executor(rt_dir)

def debug_tensor(rt: Executor) -> None:
    for i in rt.get_inputs():
        print(i.name)
    for i in rt.get_outputs():
        print(i.name)

def run(model_test, device: str) -> None:
    model_dir = model_test.model_dir
    model_pb_path = os.path.join(model_dir, "model.onnx")
    model = onnx.load(model_pb_path)
    rt = build_model(model_pb_path)
    # debug_tensor(rt)
    for test_data_npz in glob.glob(os.path.join(model_dir, "test_data_*.npz")):
        test_data = np.load(test_data_npz, encoding="bytes")
        inputs = list(test_data["inputs"])
        outputs = [] #list(prepared_model.run(inputs))
        ref_outputs = test_data["outputs"]
        assert_similar_outputs(
            ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
        )
    for test_data_dir in glob.glob(os.path.join(model_dir, "test_data_set*")):
        print(test_data_dir)
        inputs = []
        inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
        for i in range(inputs_num):
            input_file = os.path.join(test_data_dir, f"input_{i}.pb")
            _load_proto(input_file, inputs, model.graph.input[i].type)
        ref_outputs = []
        ref_outputs_num = len(
            glob.glob(os.path.join(test_data_dir, "output_*.pb"))
        )
        for i in range(ref_outputs_num):
            output_file = os.path.join(test_data_dir, f"output_{i}.pb")
            _load_proto(
                output_file, ref_outputs, model.graph.output[i].type
            )

        nnf_inputs = dict()
        nnf_torch_inputs = list()
        for input_i in range(len(inputs)):
            name = model.graph.input[input_i].name
            nnf_torch_inputs.append(torch.tensor(inputs[input_i]).cuda())
            nnf_inputs[name] = cast_pytorch_tensor(nnf_torch_inputs[-1])
        nnf_outputs = dict()
        nnf_torch_outputs = list()
        for output_i in range(len(ref_outputs)):
            name = model.graph.output[output_i].name
            nnf_torch_outputs.append(torch.tensor(ref_outputs[output_i]).cuda())
            nnf_torch_outputs[-1].zero_()
            nnf_outputs[name] = cast_pytorch_tensor(nnf_torch_outputs[-1])
        rt.feed_data(nnf_inputs, nnf_outputs)
        outputs = [t.cpu().numpy() for t in nnf_torch_outputs]#list(prepared_model.run(inputs))
        assert_similar_outputs(
            ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
        )

def assert_similar_outputs(
    ref_outputs: Sequence[Any],
    outputs: Sequence[Any],
    rtol: float,
    atol: float,
) -> None:
    np.testing.assert_equal(len(outputs), len(ref_outputs))
    for i in range(len(outputs)):
        if isinstance(outputs[i], (list, tuple)):
            for j in range(len(outputs[i])):
                assert_similar_outputs(
                    ref_outputs[i][j], outputs[i][j], rtol, atol
                )
        else:
            np.testing.assert_equal(outputs[i].dtype, ref_outputs[i].dtype)
            if ref_outputs[i].dtype == np.object:
                np.testing.assert_array_equal(outputs[i], ref_outputs[i])
            else:
                np.testing.assert_allclose(
                    outputs[i], ref_outputs[i], rtol=rtol, atol=atol
                )

def _load_proto(
    proto_filename: str,
    target_list: List[Union[np.ndarray, List[Any]]],
    model_type_proto: TypeProto,
) -> None:
    with open(proto_filename, "rb") as f:
        protobuf_content = f.read()
        if model_type_proto.HasField("sequence_type"):
            sequence = onnx.SequenceProto()
            sequence.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_list(sequence))
        elif model_type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_array(tensor))
        elif model_type_proto.HasField("optional_type"):
            optional = onnx.OptionalProto()
            optional.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_optional(optional))
        else:
            print(
                "Loading proto of that specific type (Map/Sparse Tensor) is currently not supported"
            )

def main() -> None:
    for rt in load_model_tests(data_dir='/home/affinity/nnfusion/models/onnx/onnx/backend/test/data', kind="node"):
        if not rt.name == "test_instancenorm_epsilon":
            continue
        run(rt, "")

if __name__ == "__main__":
    main()
