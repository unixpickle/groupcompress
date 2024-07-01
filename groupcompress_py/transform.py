import torch
import torch.nn as nn


class Transform(nn.Module):
    """
    A permutation of some joint bits in a larger bitmap.

    :param indices: a Long tensor of shape [num_bits]
    :param perm: a Long tensor of shape [2**num_bits]
    """

    indices: torch.Tensor
    perm: torch.Tensor

    def __init__(self, indices: torch.Tensor, perm: torch.Tensor):
        super().__init__()
        self.register_buffer("indices", indices)
        self.register_buffer("perm", perm)
        self.register_buffer(
            "bits", 2 ** torch.arange(0, len(indices), device=indices.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a uint8 [N x K] tensor of booleans.
        :return: an [N x K] tensor of booleans.
        """
        flags = x[:, self.indices]
        values = torch.einsum("ij,j->i", flags.long(), self.bits)
        permuted = self.perm[values]
        out = x.clone()
        out[:, self.indices] = (permuted.unsqueeze(-1) & self.bits) != 0
        return out

    def inverse(self) -> "Transform":
        perm = torch.empty_like(self.perm)
        perm.scatter_(
            0, self.perm, torch.arange(len(self.perm), device=self.perm.device)
        )
        return Transform(self.indices, perm)
