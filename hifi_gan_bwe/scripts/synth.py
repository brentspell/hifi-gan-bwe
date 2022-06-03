""" HiFi-GAN+ audio synthesis

This script runs inference using a pretrained HiFi-GAN+ model. It loads
an audio file in an format supported by the audioread package, runs the
model forward, and then writes the results to an output file, in any
format supported by the soundfile library.

"""

import argparse
from pathlib import Path

import audioread
import numpy as np
import soundfile
import torch

from hifi_gan_bwe import models


def main() -> None:
    parser = argparse.ArgumentParser(description="HiFi-GAN+ Bandwidth Extender")
    parser.add_argument(
        "model",
        help="pretrained model name or path",
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="input audio file path",
    )
    parser.add_argument(
        "target_path",
        type=Path,
        help="output audio file path",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device to use for synthesis (ex: cpu, cuda, cuda:1, etc.)",
    )

    args = parser.parse_args()

    # load the model
    torch.set_grad_enabled(False)
    model = models.BandwidthExtender.from_pretrained(args.model).to(args.device)

    # load the source audio file
    with audioread.audio_open(str(args.source_path)) as input_:
        sample_rate = input_.samplerate
        audio = (
            np.hstack([np.frombuffer(b, dtype=np.int16) for b in input_])
            .reshape([-1, input_.channels])
            .astype(np.float32)
            / 32767.0
        )

    # run the bandwidth extender on each audio channel
    inputs = torch.from_numpy(audio).to(args.device)
    audio = torch.stack([model(x, sample_rate) for x in inputs.T]).T.cpu().numpy()

    # save the output file
    soundfile.write(args.target_path, audio, samplerate=int(model.sample_rate))


if __name__ == "__main__":
    main()
