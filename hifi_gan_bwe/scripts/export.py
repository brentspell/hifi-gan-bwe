""" model export script

This script takes a training checkpoint and converts it to a pretrained
HiFi-GAN+ model, removing discriminator weights, weight norm scalers,
and optimizing the model for inference.

"""

import argparse
from pathlib import Path

import git

from hifi_gan_bwe import datasets, models


def main() -> None:
    parser = argparse.ArgumentParser(description="HiFi-GAN+ Model Exporter")
    parser.add_argument(
        "model",
        help="model training prefix",
    )
    parser.add_argument(
        "--target_path",
        type=Path,
        help="exported model file name (defaults to models/<model name>.pt)",
    )
    parser.add_argument(
        "--checkpoint",
        help="checkpoint number to export (defaults to latest)",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default="logs",
        help="training log root path",
    )

    args = parser.parse_args()

    # find the source log directory
    source_paths = sorted(args.log_path.glob(f"{args.model}*"))
    if not source_paths:
        raise Exception(f"source model {args.model} not found")
    source_path = source_paths[0]

    # check the commit hash
    git_repo = git.Repo()
    git_hash = git_repo.head.object.hexsha[:7]
    if git_repo.is_dirty():
        print("warning: local git repo is dirty")
    if git_hash not in source_path.name:
        print("warning: current git hash doesn't match model")

    # load the model checkpoint and detach weightnorm
    model = models.BandwidthExtender.from_checkpoint(source_path, args.checkpoint)
    model.remove_weightnorm()

    # create the target model path
    target_path = args.target_path
    if target_path is None:
        model_name = f"{source_path.name}-{datasets.SAMPLE_RATE // 1000}kHz.pt"
        target_path = Path(__file__).parents[2] / "models" / model_name
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # save the model
    model.save(target_path)

    print(f"exported {source_path.name} to {target_path}")


if __name__ == "__main__":
    main()
