import argparse
import random
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
from PIL import Image

from groupcompress_py.entropy_search import entropy_search
from groupcompress_py.metric import bitwise_entropy


def main():
    args = parse_args()

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    loader = load_data(device, args.batch_size)

    model = nn.Sequential()
    while True:
        batch = next(loader)
        batch = model(batch)
        entropy = bitwise_entropy(batch).item()
        result = entropy_search(
            dataset=batch,
            num_bits=args.num_bits,
            num_samples=args.samples,
            batch_size=args.microbatch,
        )
        print(f"step {len(model)}: loss={entropy} delta={result.delta}")
        model.append(result.transform)

        prior = estimate_prior(model, loader, args.prior_samples)
        samples = sample(model, prior, args.sample_grid**2)
        samples = samples.view(args.sample_grid, args.sample_grid, 28, 28, 1)
        samples = (
            samples.permute(0, 2, 1, 3, 4)
            .reshape(args.sample_grid * 28, args.sample_grid * 28, 1)
            .repeat(1, 1, 3)
        )
        samples = samples.cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(samples).save(args.sample_path)


def estimate_prior(
    model: nn.Sequential, loader: Iterator[torch.Tensor], num_samples: int
) -> torch.Tensor:
    """
    :return: a [D] tensor of probabilities.
    """
    n = 0
    total = 0.0
    for batch in loader:
        batch = batch[: num_samples - n]
        batch = model(batch)
        total += batch.float().sum(0)

        n += len(batch)
        if n == num_samples:
            break

    return total / num_samples


def sample(model: nn.Sequential, prior: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    :param model: sequence of Transform layers.
    :param prior: [D]
    :return: [batch_size x D]
    """
    p = prior[None].repeat(batch_size, 1)
    bits = torch.rand_like(p) < p
    for layer in model[::-1]:
        bits = layer.inverse()(bits)
    return bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=128, help="examples per layer"
    )
    parser.add_argument(
        "--microbatch",
        type=int,
        default=128,
        help="samples per entropy evaluation, used to save memory",
    )
    parser.add_argument("--num-bits", type=int, default=3, help="bits per group")
    parser.add_argument("--samples", type=int, default=100000, help="groups to sample")
    parser.add_argument(
        "--sample-grid", type=int, default=4, help="size of sample grid"
    )
    parser.add_argument(
        "--prior-samples",
        type=int,
        default=60000,
        help="number of samples to compute prior",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default="samples.png",
        help="path of samples output image",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="model.json",
        help="path to save model checkpoint",
    )
    return parser.parse_args()


def load_data(device: torch.device, batch_size: int) -> Iterator[torch.Tensor]:
    dataset = torchvision.datasets.MNIST("data/mnist", train=True, download=True)
    batch = []
    while True:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for i in indices:
            example, _ = dataset[i]
            batch.append(np.array(example).flatten())
            if len(batch) == batch_size:
                probs = torch.from_numpy(np.stack(batch)).to(device).float() / 255
                yield torch.rand_like(probs) < probs
                batch = []


if __name__ == "__main__":
    main()
