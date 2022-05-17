from turtle import forward
import onnx
import torch
from torch import nn
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
        attention_mask = (1 - attention_mask) * -1000
        return super().forward(hidden_state, attention_mask)

enable_overwrite = True
export_model_path = "attention.onnx"
attention_cfg = BertConfig(num_attention_heads=16)
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