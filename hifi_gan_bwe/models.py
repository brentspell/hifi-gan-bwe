""" HiFi-GAN+ bandwidth extension models

The bandwidth extender upsamples an audio signal from a lower sample rate
(typically 8/16/24/44.1kHz) to a full-band signal (usually 48kHz). Unlike
a standard resampler, the bandwidth extender attempts to reconstruct high
frequency components that were lost/never present in the band-limited
signal.

To do this, HiFi-GAN+ first upsamples the signal to the target sample rate
using bandlimited interpolation. It then passes the signal through a stack of
non-causal, non-conditional WaveNet blocks, essentially a dilated residual
convolution with a large receptive field.

The bandwidth extender is trained in the LS-GAN framework, with multiple
discriminators in the time and frequency domains. The waveform discriminators
are 1D convolutional filters that are applied at different sample rates
to the output signal. The spectrogram discriminator is a 2D convolutional
filter over the log-scale mel spectrogram.

The bandwidth extender is normalized using weightnorm over all convolutional
filters, which can be removed after training to reduce computation during
inference.

https://pixl.cs.princeton.edu/pubs/Su_2021_BEI/ICASSP2021_Su_Wang_BWE.pdf

"""

import typing as T
from pathlib import Path

import requests
import torch
import torchaudio
from tqdm import tqdm

from hifi_gan_bwe.datasets import SAMPLE_RATE

CDN_URL = "https://cdn.brentspell.com/models/hifi-gan-bwe"


class BandwidthExtender(torch.nn.Module):
    """HiFi-GAN+ generator model"""

    def __init__(self) -> None:
        super().__init__()

        # store the training sample rate in the state dict, so that
        # we can run inference on a model trained for a different rate
        self.sample_rate: torch.Tensor
        self.register_buffer("sample_rate", torch.tensor(SAMPLE_RATE))

        self._wavenet = WaveNet(
            stacks=2,
            layers=8,
            in_channels=1,
            wavenet_channels=128,
            out_channels=1,
            kernel_size=3,
            dilation_base=3,
        )

    def save(self, path: str) -> None:
        torch.jit.save(torch.jit.script(self), path)

    @staticmethod
    def from_pretrained(path: str) -> "BandwidthExtender":
        # first see if this is a hosted pretrained model, download it if so
        if not path.endswith(".pt"):
            path = _download(path)

        # load the local model file as a script module
        model = torch.jit.load(path)
        return model

    @staticmethod
    def from_checkpoint(
        log_path: str,
        checkpoint: T.Optional[str] = None,
    ) -> "BandwidthExtender":
        # load the latest/specified model state from the log path
        ckpt_paths = sorted(Path(log_path).glob(f"ckpt-{checkpoint or '*'}k.pt"))
        if not ckpt_paths:
            raise Exception("checkpoint not found")
        state = torch.load(ckpt_paths[-1])

        # create the model and load its weights from the checkpoint
        model = BandwidthExtender()
        model.apply_weightnorm()
        model.load_state_dict(state["gen_model"])
        return model

    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # allow simple synthesis over vectors by automatically unsqueezing
        squeeze = len(x.shape) == 1
        if squeeze:
            x = x.unsqueeze(0).unsqueeze(0)

        # first upsample the signal to the target sample rate
        # using bandlimited interpolation
        x = torchaudio.functional.resample(
            x,
            sample_rate,
            self.sample_rate,
            resampling_method="kaiser_window",
            lowpass_filter_width=16,
            rolloff=0.945,
            beta=14.769656459379492,
        )

        # in order to reduce edge artificacts due to residual conv padding,
        # pad the signal with silence before applying the wavenet, then
        # remove the padding afterward to get the desired signal length
        pad = self._wavenet.receptive_field // 2
        x = torch.nn.functional.pad(x, [pad, pad])
        x = torch.tanh(self._wavenet(x))
        x = x[..., pad:-pad]

        # if a single vector was requested, squeeze back to it
        if squeeze:
            x = x.squeeze(0).squeeze(0)

        return x

    def apply_weightnorm(self) -> None:
        self.apply(lambda m: self._apply_conv(torch.nn.utils.weight_norm, m))

    def remove_weightnorm(self) -> None:
        self.apply(lambda m: self._apply_conv(torch.nn.utils.remove_weight_norm, m))

    def _apply_conv(self, fn: T.Callable, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Conv1d):
            fn(module)


class WaveNet(torch.nn.Module):
    """stacked gated residual 1D convolutions

    This is a non-causal, non-conditional variant of the WaveNet architecture
    from van den Oord, et al.

    https://arxiv.org/abs/1609.03499

    """

    def __init__(
        self,
        stacks: int,
        layers: int,
        in_channels: int,
        wavenet_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation_base: int,
    ):
        super().__init__()

        # initial 1x1 convolution to match the residual channels
        self._conv_in = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=wavenet_channels,
            kernel_size=1,
        )

        # stacked gated residual convolution layers
        self._layers = torch.nn.ModuleList()
        for _ in range(stacks):
            for i in range(layers):
                layer = WaveNetLayer(
                    channels=wavenet_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                )
                self._layers.append(layer)

        # output 1x1 convolution to project to the desired output dimension
        self._conv_out = torch.nn.Conv1d(
            in_channels=wavenet_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        # calculate the network's effective receptive field
        self.receptive_field = (
            (kernel_size - 1) * stacks * sum(dilation_base**i for i in range(layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply the input projection to wavenet channels
        x = self._conv_in(x)

        # apply the wavenet layers
        s = 0
        for n in self._layers:
            x, h = n(x)
            s += h
        x = s * torch.tensor(1.0 / len(self._layers)).sqrt()

        # apply the output projection
        x = self._conv_out(x)

        return x


class WaveNetLayer(torch.nn.Module):
    """a single gated residual wavenet layer"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()

        # combined gate+activation convolution
        self._conv = torch.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

        # skip connection projection
        self._conv_skip = torch.nn.Conv1d(
            in_channels=channels // 2,
            out_channels=channels,
            kernel_size=1,
        )

        # output projection
        self._conv_out = torch.nn.Conv1d(
            in_channels=channels // 2,
            out_channels=channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        # save off the residual connection
        r = x

        # apply dilated convolution
        x = self._conv(x)

        # split and gate
        x, g = x.split(x.size(1) // 2, dim=1)
        x = torch.tanh(x) * torch.sigmoid(g)

        # apply skip and output convolutions
        s = self._conv_skip(x)
        x = self._conv_out(x)

        # add residual and apply a normalizing gain
        x = (x + r) * torch.tensor(0.5).sqrt()

        return x, s


class Discriminator(torch.nn.Module):
    """HiFi-GAN+ discriminator wrapper"""

    def __init__(self) -> None:
        super().__init__()

        # attach the spectrogram discriminator and waveform discriminators
        self._dsc = torch.nn.ModuleList()
        self._dsc.append(MelspecDiscriminator())
        for sample_rate in [6000, 12000, 24000, 48000]:
            self._dsc.append(WaveDiscriminator(sample_rate))

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        # collect the outputs and feature maps for each discriminator
        f = []
        y = []
        for dsc_model in self._dsc:
            yi, fi = dsc_model(x)
            y.append(yi)
            f.extend(fi)

        # stack the adversarial outputs into a single tensor, but return
        # the heterogeneous feature maps as a simple list
        return torch.cat(y, dim=-1), f


class WaveDiscriminator(torch.nn.Module):
    """waveform (time domain) discriminator"""

    def __init__(self, sample_rate: int):
        super().__init__()

        self._sample_rate = sample_rate

        # time domain 1D convolutions
        kernel_sizes = [15, 41, 41, 41, 41, 5, 3]
        strides = [1, 4, 4, 4, 4, 1, 1]
        channels = [16, 64, 256, 1024, 1024, 1024]
        groups = [1, 4, 16, 64, 256, 1, 1]
        self._convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    in_channels=i,
                    out_channels=c,
                    kernel_size=k,
                    stride=s,
                    groups=g,
                    padding="valid",
                )
                for k, s, i, c, g in zip(
                    kernel_sizes, strides, [1] + channels, channels, groups
                )
            ]
        )

        # output adversarial projection
        self._postnet = torch.nn.Conv1d(
            in_channels=channels[-1],
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        # resample the signal to this discriminator's sample rate
        if self._sample_rate != SAMPLE_RATE:
            x = torchaudio.functional.resample(x, SAMPLE_RATE, self._sample_rate)

        # compute hidden layers and feature maps
        f = []
        for c in self._convs:
            x = torch.nn.functional.leaky_relu(c(x), negative_slope=0.1)
            f.append(x)

        # apply the output projection and global average pooling
        x = self._postnet(x)
        x = x.mean(dim=-1)

        return x, f


class MelspecDiscriminator(torch.nn.Module):
    """mel spectrogram (frequency domain) discriminator"""

    def __init__(self) -> None:
        super().__init__()

        # mel filterbank transform
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048,
            win_length=int(0.025 * SAMPLE_RATE),
            hop_length=int(0.010 * SAMPLE_RATE),
            n_mels=128,
            power=1,
        )

        # time-frequency 2D convolutions
        kernel_sizes = [(7, 7), (4, 4), (4, 4), (4, 4)]
        strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        self._convs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=1 if i == 0 else 32,
                        out_channels=64,
                        kernel_size=k,
                        stride=s,
                        padding=(1, 2),
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(num_features=64),
                    torch.nn.GLU(dim=1),
                )
                for i, (k, s) in enumerate(zip(kernel_sizes, strides))
            ]
        )

        # output adversarial projection
        self._postnet = torch.nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(15, 5),
            stride=(1, 2),
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        # apply the log-scale mel spectrogram transform
        x = torch.log(self._melspec(x) + 1e-5)

        # compute hidden layers and feature maps
        f = []
        for c in self._convs:
            x = c(x)
            f.append(x)

        # apply the output projection and global average pooling
        x = self._postnet(x)
        x = x.mean(dim=[-2, -1])

        return x, f


def _download(name: str) -> str:
    # first see if we have a copy of the model locally
    path = Path.home() / ".local" / "hifi-gan-bwe"
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{name}.pt"
    if not path.exists():
        # if not, download it from the CDN
        with requests.get(f"{CDN_URL}/{path.name}", stream=True) as response:
            response.raise_for_status()
            path.write_bytes(
                b"".join(
                    tqdm(
                        response.iter_content(1024),
                        desc=f"downloading {path.name}",
                        unit="KB",
                    )
                )
            )

    return str(path)
