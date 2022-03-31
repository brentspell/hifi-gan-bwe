from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from hifi_gan_bwe import metrics


def test_summary(tmpdir: Path) -> None:
    summary = metrics.Summary(
        project="hifi-gan-bwe",
        name="bwe-01",
        log_path=str(tmpdir / "logs"),
        scalars=[metrics.Ema("gen_loss"), metrics.Mean("gen_fit")],
        use_wandb=False,
    )

    # scalars
    assert summary.scalars == dict(gen_loss=np.nan, gen_fit=np.nan)

    summary.update(gen_loss=0.5, gen_fit=1.0)
    assert summary.scalars == dict(gen_loss=0.5, gen_fit=1.0)

    summary.save(iterations=1000)
    assert summary.scalars == dict(gen_loss=np.nan, gen_fit=np.nan)

    # figure
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.zeros([10]))
    summary.figure(iterations=1000, figure=fig, name="figure")

    # audio
    audio = np.zeros([8000])
    summary.audio(iterations=1000, audio=audio, sample_rate=8000, name="audio")


def test_mean() -> None:
    metric = metrics.Mean("test")
    assert metric.value is np.nan
    assert metric.count == 0

    values = np.arange(5)
    for x in values:
        metric.update(x)
    assert metric.value == values.mean()
    assert metric.count == len(values)

    metric.reset()
    assert metric.value is np.nan
    assert metric.count == 0


def test_ema() -> None:
    alpha = 0.9
    metric = metrics.Ema("test", alpha=alpha)
    assert metric.value is np.nan

    values = np.arange(5)
    ema = np.nan
    for x in values:
        metric.update(x)
        ema = x if ema is np.nan else ema * alpha + x * (1 - alpha)
    assert np.allclose(metric.value, ema)

    metric.reset()
    assert metric.value is np.nan


def test_grad_norm() -> None:
    model = torch.nn.Linear(32, 64)
    y_pred = model(torch.zeros([1, 32]))
    y_true = torch.ones_like(y_pred)
    loss = torch.nn.MSELoss()(y_pred, y_true)
    loss.backward()
    assert metrics.grad_norm(model).shape == ()


@torch.no_grad()
def test_weight_norm() -> None:
    model = torch.nn.Linear(32, 64)
    assert np.allclose(metrics.weight_norm(model), model.weight.norm(2))
