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
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio
import librosa
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
        default="cuda",
        help="torch device to use for synthesis (ex: cpu, cuda, cuda:1, etc.)",
    )
    parser.add_argument(
        "--fade_stride",
        type=float,
        default=30,
        help="streaming chunk length, in seconds",
    )
    parser.add_argument(
        "--fade_length",
        type=float,
        default=0.025,
        help="cross-fading overlap, in seconds",
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
    inputs = torch.from_numpy(audio.copy()).to(args.device)
    audio = (
        torch.stack([_stream(args, model, x, sample_rate) for x in inputs.T])
        .T.cpu()
        .numpy()
    )

    # save the output file
    soundfile.write(args.target_path, audio, samplerate=int(model.sample_rate))


def _stream(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    stride_samples = int(args.fade_stride) * sample_rate
    fade_samples = int(args.fade_length * sample_rate)
    # create a linear cross-fader
    # upsample_rate=model.sample_rate//sample_rate
    upsample_rate=torch.div(model.sample_rate, sample_rate, rounding_mode='floor')
    fade_in = torch.linspace(0, 1, fade_samples*upsample_rate).to(x.device)
    fade_ou = fade_in.flip(0)
    # Calculate the number of elements to pad
    res=len(x)%stride_samples
    
    # Pad the input tensor with zeros
    if res > 0:
        n=int(np.ceil(len(x)/stride_samples))
        pad_size = n*stride_samples +fade_samples - len(x)
        padded_x=F.pad (x, (0,int(pad_size)), mode = "constant", value = 0.0)
    else:
        pad_size=0
        padded_x = x
    
    # window the audio into overlapping frames
    frames = padded_x.unfold(
        dimension=0,
        size=stride_samples + fade_samples,
        step=stride_samples,
    )
    prev = torch.zeros_like(fade_ou)
    output = []
    first_frame=True
    for frame in tqdm(frames):
        # run the bandwidth extender on the current frame
        y = model(frame, sample_rate)
        # fade out the previous frame, fade in the current
        if first_frame:
            first_frame=False
        else:
            y[:fade_samples*upsample_rate] = prev * fade_ou + y[:fade_samples*upsample_rate] * fade_in
        # save off the previous frame for fading into the next
        # and add the current frame to the output
        prev = y[-fade_samples*upsample_rate:]
        output.append(y[:-fade_samples*upsample_rate])

    # tack on the fade out of the last frame
    output.append(prev)
    return torch.cat(output)[:-pad_size*upsample_rate]


if __name__ == "__main__":
    main()
