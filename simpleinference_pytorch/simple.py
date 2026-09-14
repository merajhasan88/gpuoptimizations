import torch
import torch.nn as nn


# Use the challenge's PyTorch backend for this signature.
# input, model parameters, and output are already on the GPU.
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    with torch.inference_mode():
        # Transpose view:
        # [output_size, input_size] -> [input_size, output_size]
        # This changes indexing without copying the weight data.
        weight_t = model.weight.t()
        bias = model.bias

        if bias is None:
            # A bias-free layer needs only matrix multiplication.
            torch.mm(input, weight_t, out=output)
        else:
            # Compute input @ weight_t + bias.
            # The bias broadcasts across every row in the batch.
            # Write directly into the supplied output tensor.
            torch.addmm(bias, input, weight_t, out=output)
