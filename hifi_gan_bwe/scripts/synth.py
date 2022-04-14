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

    args = parser.parse_args()

    # load the model
    torch.set_grad_enabled(False)
    model = models.BandwidthExtender.from_pretrained(args.model)

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
    audio = np.stack([model(torch.from_numpy(x), sample_rate) for x in audio.T]).T

    # save the output file
    soundfile.write(args.target_path, audio, samplerate=int(model.sample_rate))


if __name__ == "__main__":
    main()
