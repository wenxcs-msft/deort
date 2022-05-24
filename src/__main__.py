import numpy as np
import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto, numpy_helper
import torch
from torch import Graph, nn
from typing import Optional, Tuple
import transformers
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import *

#https://github.com/microsoft/onnxruntime/issues/10339
#https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/Dev_Guide.md
#https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/onnxruntime/python/tools/transformers/fusion_attention.py#L281-L441

class Attention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, hidden_state, attention_mask):
        B = torch.rand(768)
        hidden_state = hidden_state + B
        attention_mask = attention_mask.float()
        attention_mask = torch.unsqueeze(attention_mask, 1)
        attention_mask = torch.unsqueeze(attention_mask, 2)
        attention_mask = (1 - attention_mask) * - 10000
        return super().forward(hidden_state, attention_mask)

enable_overwrite = True
export_model_path = "attention.onnx"
attention_cfg = BertConfig(num_attention_heads=12)
attention_nn = Attention(attention_cfg).eval()
inputs = {
    'input_ids':      torch.rand(1, 768, 768),
    'input_mask':     torch.randint(high=768, size=[1, 768])
}

if enable_overwrite or not os.path.exists(export_model_path):
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(attention_nn,                                     # model being run
                          args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                          f=export_model_path,                              # where to save the model (can be a file or file-like object)
                          opset_version=11,
                          do_constant_folding=True,                         # whether to execute constant folding for optimization
                          input_names=['input_ids',                         # the model's input names
                                       'input_mask'],
                          dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                        'input_mask' : symbolic_names})
        print("Model exported at ", export_model_path)

os.system("python -m onnxruntime.transformers.optimizer --input attention.onnx --output attention.opt.onnx")

# Added template attention layers
onnx_opted_attention = onnx.load_model("optimized.onnx")
onnx_attention_opted_graph = onnx_opted_attention.graph

def split_qkv_weight(graph, qkv_weight, prefix):
    for init in graph.initializer:
        if init.name == qkv_weight:
            # [768, 2304]
            # [768, 768], [768, 768], [768, 768]

            qkv = numpy_helper.to_array(init)
            ws = np.split(qkv, [768, 768*2], axis=1)
            qw = ws[0]
            kw = ws[1]
            vw = ws[2]

            q_weight = onnx.helper.make_tensor(
                name=prefix + '109',
                data_type=TensorProto.FLOAT16,
                dims=[768, 768],
                vals= qw.tobytes(),
                raw=True)

            k_weight = onnx.helper.make_tensor(
                name=prefix + '110',
                data_type=TensorProto.FLOAT16,
                dims=[768, 768],
                vals= kw.tobytes(),
                raw=True)
            
            v_weight = onnx.helper.make_tensor(
                name=prefix + '113',
                data_type=TensorProto.FLOAT16,
                dims=[768, 768],
                vals= vw.tobytes(),
                raw=True)
            
            graph.initializer.remove(init)
            
            return (q_weight, k_weight, v_weight)

def split_qkv_bias(graph, qkv_bias, prefix):
    for init in graph.initializer:
        if init.name == qkv_bias:
            qkv = numpy_helper.to_array(init)
            ws = np.split(qkv, [768, 768*2], axis=0)
            qw = ws[0]
            kw = ws[1]
            vw = ws[2]

            q_bias = onnx.helper.make_tensor(
                name=prefix + 'self.query.bias',
                data_type=TensorProto.FLOAT16,
                dims=[768],
                vals= qw.tobytes(),
                raw=True)

            k_bias = onnx.helper.make_tensor(
                name=prefix + 'self.key.bias',
                data_type=TensorProto.FLOAT16,
                dims=[768],
                vals= kw.tobytes(),
                raw=True)
            
            v_bias = onnx.helper.make_tensor(
                name=prefix + 'self.value.bias',
                data_type=TensorProto.FLOAT16,
                dims=[768],
                vals= vw.tobytes(),
                raw=True)
            
            graph.initializer.remove(init)

            return (q_bias, k_bias, v_bias)

def generate_attention_subgraph(hidden_state, qkv_weight, qkv_bias, attention_mask, attention_out, origin_attention_graph, prefix = "sg"):
    # subgraph info:
    # input:
    # attention_mask -set-> Add_51:in:2, hidden_state -replace-> atn_0_13
    # output:
    # attention_out -replace-> atn_0_93
    # Change from first Add(Attention)Matmul
    # to broken un-optimized version
    onnx_attention = onnx.load_model("attention.onnx")
    onnx_attention_graph = onnx_attention.graph
    remove_list = ['Cast_2', 'Constant_5', 'Constant_7', 'Constant_49','Unsqueeze_3', 'Unsqueeze_4', 'Sub_6', 'Mul_8']

    nodes_list = []
    appened_mode = False
    for node in onnx_attention_graph.node:
        if node.name== "Add_1":
            appened_mode = True
            continue
        if node.name == "MatMul_65":
            appened_mode = False
            continue
        if node.name in remove_list:
            continue
        if appened_mode:
            nodes_list.append(node)
    
    for node in nodes_list:
        node.name = prefix+node.name
        for i in range(0, len(node.input)):
            node.input[i] = prefix + node.input[i]
        for i in range(0, len(node.output)):
            node.output[i] = prefix + node.output[i]
        
        if node.name == prefix + "Add_51":
            node.input[1] = attention_mask

        if node.name == prefix + "Reshape_64":
            node.output[0] = attention_out
        
        if node.name == prefix + "MatMul_9":
            node.input[0] = hidden_state
    
        if node.name == prefix + "MatMul_11":
            node.input[0] = hidden_state

        if node.name == prefix + "MatMul_23":
            node.input[0] = hidden_state
        
        if node.name == prefix + "Div_50":
            node.input[1] = prefix + "76_half"
    
    q_weight, k_weight, v_weight = split_qkv_weight(origin_attention_graph, qkv_weight, prefix)
    q_bias, k_bias, v_bias = split_qkv_bias(origin_attention_graph, qkv_bias, prefix)

    '''
    t_13 = helper.make_tensor_value_info('13', TensorProto.FLOAT, [1, 768])
    input_mask = helper.make_tensor_value_info('input_mask', TensorProto.INT64, [1, 768])
    t_93 = helper.make_tensor_value_info('93', TensorProto.FLOAT, [1, 768, 768])
    '''

    graph_def = helper.make_graph(
        [], #nodes_list,        # nodes
        'test-model',      # name
        [],  # inputs
        []
    )

    graph_def.initializer.append(q_weight)
    graph_def.initializer.append(k_weight)
    graph_def.initializer.append(v_weight)
    graph_def.initializer.append(q_bias)
    graph_def.initializer.append(k_bias)
    graph_def.initializer.append(v_bias)
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '116', data_type=TensorProto.INT64, dims=[1], vals=[12]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '117', data_type=TensorProto.INT64, dims=[1], vals=[64]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '111', data_type=TensorProto.INT64, dims=[1], vals=[12]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '112', data_type=TensorProto.INT64, dims=[1], vals=[64]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '114', data_type=TensorProto.INT64, dims=[1], vals=[12]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '115', data_type=TensorProto.INT64, dims=[1], vals=[64]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '118', data_type=TensorProto.INT64, dims=[1], vals=[768]))
    graph_def.initializer.append(onnx.helper.make_tensor( name=prefix + '76_half', data_type=TensorProto.FLOAT16, dims=[], vals=[8]))

    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    #onnx.checker.check_model(model_def)
    #onnx.save_model(model_def, "attention.sub.onnx")
    return nodes_list, graph_def.initializer

def replace_graph_attention_mask(graph):
    attention_mask_name = "attention_mask"
    input_mask_old_name = ""
    input_mask_new_name = ""
    for node in graph.node:
        if node.op_type == "Cast":
            if node.input[0] == attention_mask_name:
                input_mask_old_name = node.output[0]
                graph.node.remove(node)
    unsqz1 = onnx.helper.make_node(
        'Unsqueeze',
        inputs=[attention_mask_name],
        outputs=['0unsqz1'],
        axes = [1]
    )

    unsqz2 = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['0unsqz1'],
        outputs=['0unsqz2'],
        axes = [2]
    )

    cast3 = onnx.helper.make_node(
        'Cast',
        inputs=['0unsqz2'],
        outputs=['0cast3'],
        to=getattr(TensorProto, 'FLOAT16'),
    )

    sub4 = onnx.helper.make_node(
    'Sub',
    inputs=['0sub4_a', '0cast3'],
    outputs=['0sub4'],
    )

    mul5 = onnx.helper.make_node(
    'Mul',
    inputs=['0sub4', '0mul5_b'],
    outputs=['0mul5'],
    )

    input_mask_new_name = "0mul5"
    apl = [unsqz1, unsqz2, cast3, sub4, mul5]
    for node in apl:
        graph.node.append(node)

    for node in graph.node:
        for i in range(0, len(node.input)):
            if node.input[i] == input_mask_old_name:
                node.input[i] = input_mask_new_name
    
    sub4_a = onnx.helper.make_tensor(
        name='0sub4_a',
        data_type=TensorProto.FLOAT16,
        dims=[],
        vals=[1])
    mul5_b = onnx.helper.make_tensor(
        name='0mul5_b',
        data_type=TensorProto.FLOAT16,
        dims=[],
        vals=[-10000])

    graph.initializer.append(sub4_a)
    graph.initializer.append(mul5_b)

    return input_mask_new_name

def replace_graph_attention(graph):
    attention_mask_name = replace_graph_attention_mask(graph)
    attention_cnt = 0
    atn_list = []
    for node in graph.node:
        if node.op_type == "Attention":
            new_sub_graph, new_sub_init = generate_attention_subgraph(node.input[0], node.input[1], node.input[2], attention_mask_name, node.output[0], graph, "atn_%d_"%(attention_cnt))
            for tnode in new_sub_graph:
                graph.node.append(tnode)
            for init in new_sub_init:
                graph.initializer.append(init)
            attention_cnt = attention_cnt + 1
            atn_list.append(node)
    for node in atn_list:
        graph.node.remove(node)
    
replace_graph_attention(onnx_attention_opted_graph)
onnx_opted_attention.ir_version = 6
#onnx_opted_attention.graph.ClearField('initializer')
onnx.save_model(onnx_opted_attention, "de_optimized.onnx")