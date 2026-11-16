"""Replacements for coremltools' own translations of PyTorch ops, for conversions that need them. Importing this module
registers them: each convert.py whose wrapper uses one of these ops imports it before converting.

    import common.coreml_torch_ops  # noqa: F401  (registers the tanh SiLU)
"""

from coremltools.converters.mil import Builder as mil_builder
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs


@register_torch_op(override=True)
def silu(context, node):  # named silu so it replaces coremltools' own translation of torch's silu
    """Replaces Core ML's silu op, which the Neural Engine computes too loosely: silu(0) comes out as up to 1.7e-3
    instead of 0. In Step 1, conv2 amplified the error (float16 features cosine 0.984 on silence, 0.997 on speech).

    silu(x) = x * sigmoid(x) = (x/2) * (1 + tanh(x/2)) gives 0.99999+ on the Neural Engine, all ops still on it
    (step1_frontend/tools/diagnose_float16.py). With no sigmoid op, coremltools' backend pass fuse_activation_silu
    can't fold it back into the silu op.

    Weak spot: for very negative x, 1 + tanh(x/2) cancels in float16. The absolute error stays under about 1e-3, and
    only where SiLU is near 0.

    Used by Step 1 (frontend: after the linear layer and conv1) and Step 5 (decoder: the gate of every layer's MLP).
    """
    values = _get_inputs(context, node, expected=1)[0]
    half_values = mil_builder.mul(x=values, y=0.5)
    one_plus_tanh = mil_builder.add(x=mil_builder.tanh(x=half_values), y=1.0)
    context.add(mil_builder.mul(x=half_values, y=one_plus_tanh, name=node.name))
