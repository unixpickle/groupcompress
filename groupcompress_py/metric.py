import torch


def bitwise_entropy(bits: torch.Tensor) -> torch.Tensor:
    """
    :param bits: [N x ... x bits] tensor of booleans
    :return: [...] tensor of entropy
    """
    sum = bits.float().sum(0)
    prob = sum / bits.shape[0]
    ent = (
        prob * prob.clamp(min=1e-8).log()
        + (1 - prob) * (1 - prob).clamp(min=1e-8).log()
    )
    return -ent.sum(-1)
