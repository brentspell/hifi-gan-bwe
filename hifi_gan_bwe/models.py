import typing as T
from pathlib import Path

import numpy as np
import requests
import torch
import torchaudio
from tqdm import tqdm

from hifi_gan_bwe.datasets import SAMPLE_RATE

CDN_URL = "https://cdn.brentspell.com/models/hifi-gan-bwe"


class BandwidthExtender(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # store the training sample rate in the state dict, so that
        # we can run inference on a model trained for a different rate
        self.sample_rate: torch.Tensor
        self.register_buffer("sample_rate", torch.as_tensor(SAMPLE_RATE))

        self._wavenet = WaveNet(
            stacks=2,
            layers=8,
            in_channels=1,
            wavenet_channels=128,
            out_channels=1,
            kernel_size=3,
            dilation_base=3,
        )

    @staticmethod
    def from_pretrained(path: str) -> "BandwidthExtender":
        # first see if this is a hosted pretrained model, download it if so
        if not path.endswith(".pt"):
            path = _download(path)

        # load the pretrained model's weights from the path
        state = torch.load(path)
        model = BandwidthExtender()
        model.load_state_dict(state)
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

    @property
    def receptive_field(self) -> int:
        return self._wavenet.receptive_field

    def apply_weightnorm(self) -> None:
        self.apply(lambda m: self._apply_conv(torch.nn.utils.weight_norm, m))

    def remove_weightnorm(self) -> None:
        self.apply(lambda m: self._apply_conv(torch.nn.utils.remove_weight_norm, m))

    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if squeeze := len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=float(self.sample_rate) / sample_rate,
            mode="linear",
        )

        pad = self.receptive_field // 2
        x = torch.nn.functional.pad(x, [pad, pad])
        x = torch.tanh(self._wavenet(x))
        x = x[..., pad:-pad]

        if squeeze:
            x = x.squeeze(0).squeeze(0)

        return x

    def _apply_conv(self, fn: T.Callable, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Conv1d):
            fn(module)


class WaveNet(torch.nn.Module):
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

        self._conv_in = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=wavenet_channels,
            kernel_size=1,
        )

        self._layers = torch.nn.ModuleList()
        for _ in range(stacks):
            for i in range(layers):
                layer = WaveNetLayer(
                    channels=wavenet_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                )
                self._layers.append(layer)

        self._conv_out = torch.nn.Conv1d(
            in_channels=wavenet_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

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
        x = s * np.sqrt(1.0 / len(self._layers))

        # apply the output projection
        x = self._conv_out(x)

        return x


class WaveNetLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()

        self._conv = torch.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

        self._conv_skip = torch.nn.Conv1d(
            in_channels=channels // 2,
            out_channels=channels,
            kernel_size=1,
        )
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
        x = (x + r) * np.sqrt(0.5)

        return x, s


class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._dsc = torch.nn.ModuleList()
        self._dsc.append(MelspecDiscriminator())
        for fs in [6000, 12000, 24000, 48000]:
            self._dsc.append(WaveDiscriminator(fs))

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        f = []
        y = []
        for dsc_model in self._dsc:
            yi, fi = dsc_model(x)
            y.append(yi)
            f.extend(fi)
        return torch.cat(y, dim=-1), f


class WaveDiscriminator(torch.nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()

        self._sample_rate = sample_rate

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

        self._postnet = torch.nn.Conv1d(
            in_channels=channels[-1],
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        if self._sample_rate != SAMPLE_RATE:
            x = torchaudio.functional.resample(x, SAMPLE_RATE, self._sample_rate)

        f = []
        for c in self._convs:
            x = torch.nn.functional.leaky_relu(c(x), negative_slope=0.1)
            f.append(x)

        x = self._postnet(x)
        x = x.mean(dim=-1)

        return x, f


class MelspecDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048,
            win_length=int(0.025 * SAMPLE_RATE),
            hop_length=int(0.010 * SAMPLE_RATE),
            n_mels=128,
            power=1,
        )

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

        self._postnet = torch.nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(15, 5),
            stride=(1, 2),
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        x = torch.log(self._melspec(x) + 1e-5)

        f = []
        for c in self._convs:
            x = c(x)
            f.append(x)

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
