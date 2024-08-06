import pytest
import torch

from groupcompress_py.kernels import count_bit_patterns


@pytest.mark.parametrize("device", ["cpu"])
def test_count_bit_patterns(device: str):
    inputs = torch.randint(low=0, high=2, size=(100, 257), device=device).byte()
    indices = torch.randint(low=0, high=257, size=(37, 3), device=device).long()
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
    return counts
