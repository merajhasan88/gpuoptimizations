import jax
import torch
from jax.dlpack import from_dlpack as jax_from_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack


# Reuse prepared parameter views for the most recently used model.
# This cache has one entry, so it cannot grow with the number of calls.
_parameter_cache = None


def solve(input: jax.Array, model) -> jax.Array:
    global _parameter_cache

    weight = model.weight
    bias = model.bias
    cached = _parameter_cache

    # Prepare views on the first call, or when parameters are replaced.
    # Identity checks do not read or compare values on the GPU.
    if cached is None or cached[0] is not weight or cached[1] is not bias:
        # detach() removes gradient tracking while sharing storage.
        # t() creates a transpose view: [out, in] becomes [in, out].
        weight_t = weight.detach().t()
        bias_view = None if bias is None else bias.detach()

        cached = (weight, bias, weight_t, bias_view)
        _parameter_cache = cached

    weight_t, bias_view = cached[2], cached[3]

    # Share the JAX input buffer with PyTorch on the GPU.
    x = torch_from_dlpack(input)

    # Neither x nor the detached parameters require gradients.
    # This avoids entering/exiting inference_mode() on every call.
    if bias_view is None:
        output = torch.mm(x, weight_t)
    else:
        # One PyTorch operation computes the matrix product plus bias.
        # The bias vector broadcasts across all rows in the batch.
        output = torch.addmm(bias_view, x, weight_t)

    # Return the GPU result as a JAX array through shared storage.
    return jax_from_dlpack(output)
