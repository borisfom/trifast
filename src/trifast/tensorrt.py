import math
from typing import Dict, List, Tuple
import ctypes
import cupy as cp
import numpy as np
import tensorrt as trt
import torch
from polygraphy.json import from_json, to_json
from onnx import defs
import onnxscript
from onnxscript import FLOAT, BOOL

trifast_opset = onnxscript.values.Opset(domain="trifast", version=1)

# Define the schema for the custom operator
schema = defs.OpSchema(
    name="triangle_attention",
    domain="trifast",
    since_version=1,
    doc="ONNX/TRT plugin for trifast::triangle_attention.",
    inputs=[
        defs.OpSchema.FormalParameter("q", "tensor(float)", "Query tensor"),
        defs.OpSchema.FormalParameter("k", "tensor(float)", "Key tensor"),
        defs.OpSchema.FormalParameter("v", "tensor(float)", "Value tensor"),
        defs.OpSchema.FormalParameter("b", "tensor(float)", "Bias tensor"),
        defs.OpSchema.FormalParameter("m", "tensor(bool)", "Mask tensor"),
    ],
    outputs=[
        defs.OpSchema.FormalParameter("o", "tensor(float)", "Output tensor"),
        defs.OpSchema.FormalParameter("l", "tensor(float)", "Auxiliary Output tensor"),
    ],
)

triangle_attention_op = onnxscript.values.Op(
    opset=trifast_opset, name="triangle_attention", op_schema=schema
)


@onnxscript.script(trifast_opset)
def onnxscript_triangle_attention(q:FLOAT, k:FLOAT, v:FLOAT, b:FLOAT, mask:BOOL) -> Tuple[FLOAT, FLOAT]:
    o, l = triangle_attention_op(
        q, k, v, b, mask
    )
    return o, l

trifast_op_table = {
    torch.ops.trifast.triangle_attention.default: onnxscript_triangle_attention
}

def symbolic_triangle_attention(g, q, k, v, b, mask):
    out_type = q.type()
    out2_shape = list(out_type.sizes())
    out2_type = out_type.with_sizes(tuple(out2_shape[:2]+out2_shape[3:]))
    out = g.op(
        "trifast::triangle_attention",
        q, k, v, b, mask,
        outputs = 2
    )
    out[0].setType(out_type)
    out[1].setType(out2_type)
    return out[0], out[1]


torch.onnx.register_custom_op_symbolic(
    "trifast::triangle_attention",
    symbolic_triangle_attention,
    1,
)

"""
import tensorrt.plugin as trtp
import numpy.typing as npt

@trtp.register("trifast::triangle_attention")
def triangle_attention_desc(
        inp0: trtp.TensorDesc, 
        inp1: trtp.TensorDesc, 
        inp2: trtp.TensorDesc, 
        inp3: trtp.TensorDesc, 
        inp4: trtp.TensorDesc,
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    ndim = inp0.ndim
    o_desc = inp0.like()
    out2_shape = list(inp0.shape)
    l_desc = trtp.TensorDesc(shape=tuple(out2_shape[:2]+out2_shape[3:]), dtype=inp0.dtype)
    return o_desc, l_desc


@trtp.impl("trifast::triangle_attention")
def triangle_attention_impl(
    inp0: trtp.Tensor,
    inp1: trtp.Tensor,
    inp2: trtp.Tensor,
    inp3: trtp.Tensor,
    inp4: trtp.Tensor,
    outputs: Tuple[trtp.Tensor],
    stream: int
) -> None:
    q = torch.as_tensor(inp0, device="cuda")
    k = torch.as_tensor(inp1, device="cuda")
    v = torch.as_tensor(inp2, device="cuda")
    b = torch.as_tensor(inp3, device="cuda")
    mask = torch.as_tensor(inp4, device="cuda")
    out_t = torch.as_tensor(outputs[0], device="cuda")
    out = triangle_attention(q, k, v, b, mask)
    out_t.copy_(out)


"""
trt_to_torch = {
    trt.DataType.FLOAT: torch.float,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}

class TriangleAttentionPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None):
        trt.IPluginV2DynamicExt.__init__(self)

        self.num_outputs = 2
        self.plugin_namespace = ""
        self.plugin_type = "triangle_attention"
        self.plugin_version = "1"

        fc_dict = {}

    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):
        if output_index == 0:
            output_dims = trt.DimsExprs(inputs[0])
        else:
            expr_list = list(trt.DimsExprs(inputs[0]))
            output_dims = trt.DimsExprs(expr_list[:2]+expr_list[3:])
        return output_dims

    def serialize(self):
        return to_json(self.__dict__)

    def configure_plugin(self, inp, out):
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 5
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be (b)float16 or float32
        if pos == 0:
            return (
                desc.type == trt.DataType.FLOAT
                or desc.type == trt.DataType.HALF
                or desc.type == trt.DataType.BF16
            )
        elif pos == 4:
            return desc.type == trt.DataType.BOOL
        else:
            # should have the same type as the input[0]
            return in_out[0].type == desc.type

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            i_bs = [np.prod(i.dims) * i.type.itemsize for i in input_desc]
            o_bs = [np.prod(o.dims) * o.type.itemsize for o in output_desc]
            
            i_mem = [
                cp.cuda.UnownedMemory(inputs[i], i_bs[i], self) for i in range(len(inputs))
            ]
            o_mem = cp.cuda.UnownedMemory(outputs[0], o_bs[0], self)
            l_mem = cp.cuda.UnownedMemory(outputs[1], o_bs[1], self)
            
            i_ptr = [cp.cuda.MemoryPointer(i, 0) for i in i_mem]
            o_ptr = cp.cuda.MemoryPointer(o_mem, 0)
            l_ptr = cp.cuda.MemoryPointer(l_mem, 0)
            
            i_nd = [
                cp.ndarray((i_bs[i],), dtype=cp.uint8, memptr=i_ptr[i])
                for i in range(len(inputs))
            ]
            o_nd = cp.ndarray((o_bs[0],), dtype=cp.uint8, memptr=o_ptr)
            l_nd = cp.ndarray((o_bs[1],), dtype=cp.uint8, memptr=l_ptr)
            
            i_t = [
                torch.as_tensor(i_nd[i], device="cuda")
                .view(dtype=trt_to_torch[input_desc[i].type])
                .view(tuple(input_desc[i].dims))
                for i in range(len(inputs))
            ]
            ret = (
                torch.as_tensor(o_nd, device="cuda")
                .view(dtype=trt_to_torch[output_desc[0].type])
                .view(tuple(output_desc[0].dims))
            )
            l_ret = (
                torch.as_tensor(l_nd, device="cuda")
                .view(dtype=trt_to_torch[output_desc[1].type])
            .view(tuple(output_desc[1].dims))
            )

            o, l = torch.ops.trifast.triangle_attention(
                i_t[0],
                i_t[1],
                i_t[2],
                i_t[3],
                i_t[4],
            )
            
            ret.copy_(o)
            l_ret.copy_(l)
        return 0

    def clone(self):
        cloned_plugin = TriangleAttentionPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def get_serialization_size(self):
        return len(to_json(self.__dict__))


class TriangleAttentionPluginCreator(trt.IPluginCreator):
    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "triangle_attention"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
            ]
        )

    def create_plugin(self, name, fc):
        pl = TriangleAttentionPlugin(fc)
        return pl

    def deserialize_plugin(self, name, data):
        j = dict(from_json(data.decode("utf-8")))
        deserialized = TriangleAttentionPlugin()
        deserialized.__dict__.update(j)
        return deserialized

PLUGINS_REGISTRY = None

def register_plugins():
    global PLUGINS_REGISTRY
    if PLUGINS_REGISTRY is None:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
        PLUGINS_REGISTRY = trt.get_plugin_registry()
        PLUGINS_REGISTRY.register_creator(TriangleAttentionPluginCreator(), "")

# %%
# Using Torch-TensorRT to Insert the Kernel
# -------------------------------------------
# Create converters so that Torch-TensorRT knows how to insert them in place of equivariance ops.
# More information `here <https://pytorch.org/TensorRT/contributors/dynamo_converters.html>`_

try:
    from torch.fx.node import Argument, Target
    from torch_tensorrt.dynamo.conversion import (
        ConversionContext,
        dynamo_tensorrt_converter,
    )
    from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

    def _converter(
        ctx: ConversionContext,
        name: str,
        ptype: str,
        input_tensors: List[Argument],
    ):
        register_plugins()
        plugin_creator = PLUGINS_REGISTRY.get_plugin_creator(
            type=ptype, version="1", plugin_namespace=""
        )
        assert plugin_creator, f"Unable to find {ptype} plugin creator"

        plugin = plugin_creator.create_plugin(name, field_configs)
        assert plugin, f"Unable to create {ptype} plugin"
        for i in range(len(input_tensors)):
                # Freeze input tensor if not TensorRT Tensor already
                input_tensors[i] = get_trt_tensor(
                    ctx, input_tensors[i], f"{name}_input_{i}"
                )

        layer = ctx.net.add_plugin_v2(
            input_tensors, plugin
        )  # Add the plugin to the network being constructed
        layer.name = f"{ptype}-{name}"
        return layer.get_output(0)

    @dynamo_tensorrt_converter(
        torch.ops.trifast.triangle_attention.default
    )  # type: ignore
    def _(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ):
        input_tensors = list(args)
        return _converter(
            ctx, name, "triangle_attention", input_tensors
        )
finally:
    pass
