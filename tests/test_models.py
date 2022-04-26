from pathlib import Path

import numpy as np
import pytest
import torch

from hifi_gan_bwe import models


@torch.no_grad()
def test_generator() -> None:
    model = models.BandwidthExtender()

    y = model(torch.zeros([1, 1, 80]), 8000)
    assert list(y.shape) == [1, 1, 480]

    y = model(torch.zeros([80]), 8000)
    assert list(y.shape) == [480]

    y = model(torch.zeros([160]), 16000)
    assert list(y.shape) == [480]

    y = model(torch.zeros([240]), 24000)
    assert list(y.shape) == [480]

    y = model(torch.zeros([441]), 44100)
    assert list(y.shape) == [480]


@torch.no_grad()
def test_save_load(tmpdir: Path) -> None:
    # invalid checkpoint
    with pytest.raises(Exception):
        models.BandwidthExtender.from_checkpoint(str(tmpdir / "invalid"))

    # checkpoint save/load
    model = models.BandwidthExtender()
    model.apply_weightnorm()

    ckpt_path = tmpdir / "ckpt-0100k.pt"
    torch.save(dict(gen_model=model.state_dict()), str(ckpt_path))
    loaded = models.BandwidthExtender.from_checkpoint(str(tmpdir))
    assert_params(model, loaded)

    # pretrained model save/load
    model = models.BandwidthExtender()
    model.apply_weightnorm()
    model.remove_weightnorm()

    model_path = tmpdir / "pretrained.pt"
    model.save(str(model_path))
    loaded = models.BandwidthExtender.from_pretrained(str(model_path))
    assert_params(model, loaded)


@torch.no_grad()
def test_hosted() -> None:
    model_name = "hifi-gan-bwe-05-cd9f4ca-vctk-48kHz"
    model = models.BandwidthExtender.from_pretrained(model_name)
    y = model(torch.zeros([80]), 8000)
    assert list(y.shape) == [480]


@torch.no_grad()
def test_discriminator() -> None:
    gen = models.BandwidthExtender()
    dsc = models.Discriminator()

    y_pred = gen(torch.zeros([1, 1, 8000]), 8000)
    y_true = torch.zeros_like(y_pred)
    y, f = dsc(torch.cat([y_pred, y_true], dim=0))
    assert y.size(0) == y_pred.size(0) + y_true.size(0)
    for f in f:
        assert f.size(0) == y_pred.size(0) + y_true.size(0)


def assert_params(model1: torch.nn.Module, model2: torch.nn.Module) -> None:
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    assert len(params1) == len(params2)
    for name, p1 in params1.items():
        assert np.allclose(p1, params2[name])
