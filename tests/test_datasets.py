import typing as T
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from hifi_gan_bwe import datasets

SAMPLE_COUNT = 10
SAMPLE_LENGTH = 96000


@pytest.fixture(scope="session")
def dataset_root(tmpdir_factory: pytest.TempPathFactory) -> T.Iterator[Path]:
    tmpdir = Path(tmpdir_factory.mktemp("datasets"))

    # create the vctk test dataset
    vctk_path = Path(tmpdir) / "vctk"
    wav_path = vctk_path / "wav48"
    for i in range(datasets.TRAIN_SPEAKERS + 1):
        speaker_path = wav_path / f"{i:02d}"
        speaker_path.mkdir(parents=True)
        for j in range(SAMPLE_COUNT):
            audio_path = speaker_path / f"{j:02d}.wav"
            audio = np.random.normal(size=SAMPLE_LENGTH)
            soundfile.write(audio_path, audio, datasets.SAMPLE_RATE)

    # create the DNS4 noise dataset
    dns_path = Path(tmpdir) / "dns4"
    noise_path = dns_path / "datasets_fullband" / "noise_fullband"
    noise_path.mkdir(parents=True)
    for i in range(SAMPLE_COUNT):
        audio_path = noise_path / f"{i:02d}.wav"
        audio = np.random.normal(size=datasets.BATCH_SIZE * SAMPLE_LENGTH)
        soundfile.write(audio_path, audio, datasets.SAMPLE_RATE)

    yield tmpdir


def test_vctk(dataset_root: Path) -> None:
    vctk_path = dataset_root / "vctk"
    train_set = datasets.VCTKDataset(str(vctk_path), training=True)
    valid_set = datasets.VCTKDataset(str(vctk_path), training=False)
    eval_set = valid_set.eval_set
    assert len(train_set) == datasets.TRAIN_SPEAKERS * SAMPLE_COUNT
    assert len(train_set.paths) == len(train_set)
    assert len(valid_set) == SAMPLE_COUNT

    x = train_set[0]
    assert x.dtype == np.float32
    assert x.shape == (datasets.SEQ_LENGTH,)

    x = eval_set[0]
    assert x.dtype == np.float32
    assert x.shape == (SAMPLE_LENGTH,)


def test_dns(dataset_root: Path) -> None:
    dns_path = dataset_root / "dns4"
    noise_set = datasets.DNSDataset(str(dns_path))
    assert len(noise_set) == SAMPLE_COUNT

    x = noise_set[0]
    assert x.dtype == np.float32
    assert x.shape == (datasets.BATCH_SIZE * datasets.SEQ_LENGTH,)


def test_preprocessor(dataset_root: Path) -> None:
    vctk_path = dataset_root / "vctk"
    dns_path = dataset_root / "dns4"

    train_set = datasets.VCTKDataset(str(vctk_path), training=True)
    valid_set = datasets.VCTKDataset(str(vctk_path), training=False)
    noise_set = datasets.DNSDataset(str(dns_path))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=datasets.Preprocessor(
            noise_set=noise_set,
            training=True,
            device="cpu",
        ),
        batch_size=datasets.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        collate_fn=datasets.Preprocessor(
            noise_set=noise_set,
            training=False,
            device="cpu",
        ),
        batch_size=datasets.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    x, r, y = next(iter(train_loader))
    x_len = datasets.SEQ_LENGTH * r // datasets.SAMPLE_RATE
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (datasets.BATCH_SIZE, 1, x_len)
    assert y.shape == (datasets.BATCH_SIZE, 1, datasets.SEQ_LENGTH)

    x, r, y = next(iter(valid_loader))
    x_len = datasets.SEQ_LENGTH * r // datasets.SAMPLE_RATE
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (datasets.BATCH_SIZE, 1, x_len)
    assert y.shape == (datasets.BATCH_SIZE, 1, datasets.SEQ_LENGTH)
