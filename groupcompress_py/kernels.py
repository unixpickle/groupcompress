import torch

import groupcompress_py_ext


def count_bit_patterns(inputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    :param inputs: an [N x D] byte tensor of input patterns.
    :param indices: a [K x num_bits] Long tensor of indices.
    :return: a [K x 2**num_bits] tensor of bit pattern counts.
    """
    assert inputs.dtype == torch.uint8
    assert indices.dtype == torch.long
    assert inputs.device == indices.device
    assert len(inputs.shape) == 2
    assert len(indices.shape) == 2

    num_bits = indices.shape[1]
    output = torch.zeros(
        (indices.shape[0], 2**num_bits), dtype=torch.long, device=inputs.device
    )

    if output.device.type == "cpu":
        groupcompress_py_ext.count_bit_patterns_cpu(inputs, indices, output)
    elif output.device.type == "cuda":
        groupcompress_py_ext.count_bit_patterns_cuda(inputs, indices, output)
    else:
        raise ValueError(f"unsupported device type: {output.device.type}")

    return output


def greedy_permutation_search(counts: torch.Tensor) -> torch.Tensor:
    """
    :param counts: a [K x 2**num_bits] Long tensor of bit pattern counts.
    :return: a [K x 2**num_bits] Long tensor of optimized permutations.
    """
    assert counts.dtype == torch.long
    assert len(counts.shape) == 2

    output = torch.zeros_like(counts)

    if output.device.type == "cpu":
        groupcompress_py_ext.greedy_permutation_search_cpu(counts, output)
    elif output.device.type == "cuda":
        # Kernel will write inverse permutation as well, but we will
        # ignore this output.
        groupcompress_py_ext.greedy_permutation_search_cuda(
            counts, output, output.clone()
        )
    else:
        raise ValueError(f"unsupported device type: {output.device.type}")

    return output
