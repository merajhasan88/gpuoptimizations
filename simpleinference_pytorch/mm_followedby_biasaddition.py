import torch
import torch.nn as nn


# Keep the editor set to PyTorch.
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    with torch.inference_mode():
        # Restore the winning transpose view: no weight data is copied.
        weight_t = model.weight.t()
        bias = model.bias

        # NEW TEST: perform matrix multiplication independently.
        # Write directly into the supplied output tensor.
        torch.mm(input, weight_t, out=output)

        if bias is not None:
            # Add the bias to every row, modifying output in place.
            # This launches a separate operation.
            output.add_(bias)
