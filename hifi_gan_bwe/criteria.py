""" custom model losses """

import torch
import torchaudio

from hifi_gan_bwe.datasets import RESAMPLE_RATES, SAMPLE_RATE


class ContentCriteria(torch.nn.Module):
    """HiFi-GAN+ generator content losses

    These are the non-adversarial content losses described in the
    original paper. The losses include L1 losses on the raw waveform,
    a set of log-scale STFTs, and the mel spectrogram.

    """

    def __init__(self) -> None:
        super().__init__()

        self._l1_loss = torch.nn.L1Loss()
        self._stft_xforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=frame_length,
                    hop_length=frame_length // 4,
                    power=1,
                )
                for frame_length in [512, 1024, 2048, 4096]
            ]
        )
        self._melspec_xform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            f_min=RESAMPLE_RATES[0] // 2,
            f_max=SAMPLE_RATE // 2,
            n_fft=2048,
            win_length=int(0.025 * SAMPLE_RATE),
            hop_length=int(0.010 * SAMPLE_RATE),
            n_mels=128,
            power=1,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # L1 waveform loss
        wav_loss = self._l1_loss(y_pred, y_true)

        # L1 log spectrogram loss
        stft_loss = torch.tensor(0.0).to(y_pred.device)
        for stft in self._stft_xforms:
            s_true = torch.log(stft(y_true) + 1e-5)
            s_pred = torch.log(stft(y_pred) + 1e-5)
            stft_loss += self._l1_loss(s_pred, s_true)
        stft_loss /= len(self._stft_xforms)

        # mel spectrogram loss
        m_true = torch.log(self._melspec_xform(y_true) + 1e-5)
        m_pred = torch.log(self._melspec_xform(y_pred) + 1e-5)
        melspec_loss = self._l1_loss(m_pred, m_true)

        return wav_loss + stft_loss + melspec_loss
