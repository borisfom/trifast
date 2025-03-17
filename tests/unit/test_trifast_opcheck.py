import torch
import pytest
from trifast.torch import triangle_attention
from trifast.utils import gen_tensors
from torch.library import opcheck


@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (16, 1, 16),
    ],
)
def test_opcheck(
        n: int, h: int, d: int, mask: bool, bs: int
):
    dtype = torch.float32
    device = torch.device("cuda")
    q, k, v, b, m = gen_tensors(
        n, d, h, use_mask=mask, device=device, dtype=torch.float32, batch=bs, std=1.0
    )
    from torch.library import opcheck
    from torch.autograd import gradcheck
    #gradcheck(
    #    triangle_attention,
    #    [q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m],
    #    atol=0.1, rtol=0.1
    #)
    opcheck(
        torch.ops.trifast.triangle_attention,
        [q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m],
        test_utils="ALL",
        raise_exception = True
    )
