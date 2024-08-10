import math
from dataclasses import dataclass

import torch

from .kernels import count_bit_patterns, greedy_permutation_search
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

    counts = count_bit_patterns(dataset.byte(), indices)
    perms = greedy_permutation_search(counts)

    new_counts = torch.zeros_like(counts)
    new_counts.scatter_(1, perms, counts)
    new_ent = total_bitwise_entropy(new_counts)
    old_ent = total_bitwise_entropy(counts)
    deltas = old_ent - new_ent
    best_idx = deltas.argmax()

    return EntropyResult(
        transform=Transform(indices[best_idx].clone(), perms[best_idx].clone()),
        delta=deltas[best_idx].item(),
    )


def total_bitwise_entropy(counts: torch.Tensor) -> torch.Tensor:
    """
    :param counts: [N x 2**num_bits]
    :return: [N] tensor of entropies
    """
    num_bits = int(math.log2(counts.shape[1]))
    values = torch.arange(0, 2**num_bits).to(counts)
    totals = counts.sum(-1).float()
    total = 0
    for bit in range(num_bits):
        probs_1 = (
            torch.where((values & (1 << bit) != 0), counts, torch.zeros_like(counts))
            .sum(-1)
            .float()
            / totals
        )
        probs_0 = 1 - probs_1
        total += probs_1 * probs_1.clamp(min=1e-18).log()
        total += probs_0 * probs_0.clamp(min=1e-18).log()
    return -total
