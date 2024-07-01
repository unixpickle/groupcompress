import argparse
import random
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets

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
