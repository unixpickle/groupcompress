from dataclasses import dataclass

import torch

from .metric import bitwise_entropy
from .transform import Transform


@dataclass
class EntropyResult:
    transform: Transform
    delta: float


def entropy_search(
    dataset: torch.Tensor,
    num_bits: int,
    num_samples: int,
    batch_size: int,
) -> EntropyResult:
    """
    :param dataset: [N x K] tensor of booleans.
    :param num_bits: number of bits to combine.
    :param num_samples: number of transforms to sample.
    """
    result: EntropyResult = None
    for i in range(0, num_samples, batch_size):
        n = min(num_samples - i, batch_size)
        r = _entropy_search(dataset, num_bits, n)
        if result is None or r.delta > result.delta:
            result = r
    return result


def _entropy_search(
    dataset: torch.Tensor,
    num_bits: int,
    num_samples: int,
) -> EntropyResult:
    """
    :param dataset: [N x K] tensor of booleans.
    :param num_bits: number of bits to combine.
    :param num_samples: number of transforms to sample.
    """
    indices = torch.topk(
        torch.rand(
            num_samples, dataset.shape[1], device=dataset.device, dtype=torch.float32
        ),
        k=num_bits,
    ).indices  # [num_samples x num_bits]
    permutations = torch.argsort(
        torch.rand(num_samples, 2**num_bits, device=dataset.device, dtype=torch.float32)
    )  # [num_samples x 2**num_bits]
    bitmask = 2 ** torch.arange(0, num_bits, device=indices.device)
    old_bits = dataset[:, indices].view(dataset.shape[0], num_samples, num_bits)
    patterns = torch.einsum("ijk,k->ij", old_bits.long(), bitmask)  # [N x num_samples]
    permuted = (
        permutations[None]
        .repeat(patterns.shape[0], 1, 1)
        .gather(-1, patterns[..., None])
        .squeeze(-1)
    )  # [N x num_samples]
    new_bits = (permuted[..., None] & bitmask) != 0

    old_ents = bitwise_entropy(old_bits)
    new_ents = bitwise_entropy(new_bits)
    improvements = old_ents - new_ents
    best_delta, best_index = improvements.max(0)
    best_indices = indices[best_index]
    best_perm = permutations[best_index]

    return EntropyResult(
        transform=Transform(best_indices, best_perm),
        delta=best_delta.item(),
    )
