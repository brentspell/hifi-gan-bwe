import mmap
import typing as T
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

A = T.TypeVar("A")
B = T.TypeVar("B")

SAMPLE_RATE = 48000
RESAMPLE_RATES = [8000, 16000, 24000]
BATCH_SIZE = 4
SEQ_LENGTH = int(1.0 * SAMPLE_RATE)
SIGNAL_RMS_MIN = -30
SIGNAL_RMS_MAX = -12
NOISE_SNR_MIN = 15
NOISE_SNR_MAX = 60
TRAIN_SPEAKERS = 99


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, paths: T.Iterator[Path], seq_length: int):
        super().__init__()

        self._paths = [p for p in paths if p.stat().st_size // 2 > seq_length]
        self._seq_length = seq_length

    @property
    def paths(self) -> T.List[Path]:
        return self._paths.copy()

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.load(index, seq_length=self._seq_length)

    def load(self, index: int, seq_length: int = -1) -> np.ndarray:
        with self._paths[index].open("rb") as file:
            with mmap.mmap(file.fileno(), length=0, prot=mmap.PROT_READ) as data:
                assert int.from_bytes(data[22:24], "little") == 1
                assert int.from_bytes(data[24:28], "little") == SAMPLE_RATE
                assert int.from_bytes(data[34:36], "little") == 16

                audio = np.frombuffer(data, dtype=np.int16)[22:]
                if seq_length != -1:
                    offset = np.random.randint(len(audio) - seq_length)
                    audio = audio[offset : offset + seq_length]
                return audio.astype(np.float32) / 32767.0


class VCTKDataset(WavDataset):
    def __init__(self, path: str, training: bool):
        paths = sorted((Path(path) / "wav48").glob("*"))
        paths = paths[:TRAIN_SPEAKERS] if training else paths[TRAIN_SPEAKERS:]
        super().__init__(
            paths=(p for s in paths for p in s.glob("*.wav")),
            seq_length=SEQ_LENGTH,
        )

    @property
    def eval_set(self) -> T.List[np.ndarray]:
        speaker_paths = group_by(self.paths, lambda p: p.parent.name)
        return [self.load(self.paths.index(p[0])) for p in speaker_paths.values()]


class DNSDataset(WavDataset):
    def __init__(self, path: str):
        noise_path = Path(path) / "datasets_fullband" / "noise_fullband"
        super().__init__(
            paths=noise_path.glob("*.wav"),
            seq_length=BATCH_SIZE * SEQ_LENGTH,
        )


class Preprocessor:
    def __init__(self, noise_set: WavDataset, training: bool, device: str = "cuda"):
        self._device = device
        self._training = training
        self._noise_set = noise_set

    def __call__(
        self,
        batch: T.List[np.ndarray],
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.utils.data.dataloader.default_collate(batch).unsqueeze(1)
        y = y.to(self._device)

        if self._training:
            signal_rms = torch.sqrt((y**2).mean(-1, keepdim=True))
            target_rms = torch.from_numpy(
                np.random.uniform(
                    SIGNAL_RMS_MIN,
                    SIGNAL_RMS_MAX,
                    size=[BATCH_SIZE, 1, 1],
                )
            ).to(y.device)
            target_rms = 10 ** (target_rms / 20)
            gain = target_rms / (signal_rms + 1e-5)
            y *= gain
            signal_rms *= gain

            noise_index = np.random.randint(len(self._noise_set))
            noise = torch.from_numpy(self._noise_set[noise_index]).to(y.device)
            noise = noise.reshape([BATCH_SIZE, 1, SEQ_LENGTH])
            noise_rms = torch.sqrt((noise**2).mean(-1, keepdim=True))

            source_snr = signal_rms / (noise_rms + 1e-5)
            target_snr = torch.from_numpy(
                np.random.uniform(
                    NOISE_SNR_MIN,
                    NOISE_SNR_MAX,
                    size=[BATCH_SIZE, 1, 1],
                )
            ).to(y.device)
            target_snr = 10 ** (target_snr / 20)
            noise *= source_snr / target_snr
            y += noise

            signal_peak = y.abs().max(-1, keepdim=True).values
            gain = 1.0 / torch.maximum(signal_peak, torch.tensor(1.0))
            y *= gain

        r = np.random.choice(RESAMPLE_RATES)
        x = torchaudio.functional.resample(y, SAMPLE_RATE, r)

        return x, r, y


def group_by(seq: T.Iterable[A], key: T.Callable[[A], B]) -> T.Dict[B, T.List[A]]:
    """groups an input sequence into a dictionary of lists of elements

    Args:
        seq: the sequence to group
        key: a lambda used to group each element

    Returns:
        groups: a dictionary of elements grouped by key(element)

    """

    groups = defaultdict(list)
    for value in seq:
        groups[key(value)].append(value)
    return groups
