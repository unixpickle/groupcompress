import math

import pytest
import torch

from groupcompress_py.kernels import count_bit_patterns, greedy_permutation_search

devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]


@pytest.mark.parametrize("device", devices)
def test_count_bit_patterns_small(device: str):
    inputs = torch.randint(low=0, high=2, size=(100, 257), device=device).byte()
    indices = torch.randint(low=0, high=257, size=(37, 3), device=device).long()
    actual_output = count_bit_patterns(inputs, indices)
    expected_output = count_bit_pattern_naive(inputs, indices)
    assert (actual_output == expected_output).all().item()


@pytest.mark.parametrize("device", devices)
def test_count_bit_patterns_large(device: str):
    inputs = torch.randint(low=0, high=2, size=(512, 2048), device=device).byte()
    indices = torch.randint(low=0, high=2048, size=(10000, 8), device=device).long()
    actual_output = count_bit_patterns(inputs, indices)
    expected_output = count_bit_pattern_naive(inputs, indices)
    assert (actual_output == expected_output).all().item()


def count_bit_pattern_naive(
    inputs: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    counts = torch.zeros((indices.shape[0], 2 ** indices.shape[1]), dtype=torch.long)
    for i, pattern in enumerate(indices):
        values = 0
        for j, idx in enumerate(pattern):
            values |= inputs[:, idx].long() << j
        count = torch.bincount(values, minlength=2 ** indices.shape[1])
        counts[i] = count
    return counts.to(inputs.device)


@pytest.mark.parametrize("device", devices)
def test_count_bit_patterns_time(benchmark, device: str):
    # Simulate MNIST with 10k examples and 10k permutations.
    inputs = torch.randint(low=0, high=2, size=(10000, 28 * 28), device=device).byte()
    indices = torch.randint(low=0, high=28 * 28, size=(10000, 4), device=device).long()

    def fn():
        count_bit_patterns(inputs, indices).sum().item()

    benchmark(fn)


@pytest.mark.parametrize("device", devices)
def test_greedy_permutation_search(device: str):
    counts = torch.randint(low=0, high=512, size=(10000, 32), device=device).long()
    perm = greedy_permutation_search(counts)
    orig_ent = total_bitwise_entropy(counts)

    new_counts = torch.zeros_like(counts)
    new_counts.scatter_(1, perm, counts)
    new_ent = total_bitwise_entropy(new_counts)

    assert (new_ent <= orig_ent).all().item()


@pytest.mark.parametrize("device", devices)
def test_greedy_permutation_search_time(benchmark, device: str):
    # Simulate search for 1M examples and four bits.
    counts = torch.randint(low=0, high=10000, size=(100000, 32), device=device).long()

    def fn():
        greedy_permutation_search(counts).sum().item()

    benchmark(fn)


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
