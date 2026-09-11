import jax
import torch
import torch.nn.functional as F
from jax.dlpack import from_dlpack as jax_from_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack


def solve(input: jax.Array, model) -> jax.Array:
    with torch.inference_mode():
        # Share the input buffer with PyTorch on the GPU.
        x = torch_from_dlpack(input)

        # Computes input @ weight.T + bias.
        # Also handles models with bias=None.
        output = F.linear(x, model.weight, model.bias)

    # Return the GPU result as a JAX array.
    return jax_from_dlpack(output)
