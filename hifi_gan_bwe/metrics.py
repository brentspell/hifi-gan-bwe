""" training metrics interface

Metrics are written to Weights and Biases (wandb.ai) during training and
validation for scalars, images, etc. Each metric is recorded by name and
for the current training iteration number. Scalars are maintained as
running statistics (mean, ema, etc.) over each virtual epoch, after which
they are reset, similar to tf.keras.metrics.

"""

import typing as T
from abc import abstractmethod

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt


class Summary:
    """metrics summary writer

    Args:
        project: project/task name
        name: training run name
        log_path: path to the log directory for this training run
        scalars: list of scalar metric instances to log during training
        use_wandb: True to enable wandb logging, False otherwise

    """

    def __init__(
        self,
        project: str,
        name: str,
        log_path: str,
        scalars: T.List["Scalar"],
        use_wandb: bool = True,
    ):
        self._scalars = scalars
        self._dirty: T.Set[str] = set()

        wandb.init(
            project=project,
            id=name,
            name=name,
            dir=str(log_path),
            mode="online" if use_wandb else "disabled",
        )

    @property
    def scalars(self) -> T.Dict[str, T.Any]:
        """retrieve the current state of all scalar metrics"""

        return {v.name: v.value for v in self._scalars}

    def update(self, *args: T.Any, **kwargs: T.Any) -> T.Dict[str, T.Any]:
        """update the state of a subset of scalar metrics"""

        scalars = args[0] if args else kwargs
        for scalar in self._scalars:
            if (value := scalars.get(scalar.name)) is not None:
                self._dirty.add(scalar.name)
                scalar.update(float(value))

        return {v.name: v.value for v in self._scalars if v.name in scalars}

    def figure(self, iterations: int, figure: plt.Figure, name: str = "image") -> None:
        """record a pyplot figure as a metric

        Args:
            iterations: current iteration counter
            figure: pyplot figure to log
            name: name of the metric to record

        """

        wandb.log({name: wandb.Image(figure)}, step=iterations)
        plt.close(figure)

    def audio(
        self,
        iterations: int,
        audio: np.ndarray,
        sample_rate: int,
        name: str = "audio",
    ) -> None:
        """record an audio sample metric

        Args:
            iterations: current iteration counter
            audio: array of floating point raw audio samples
            sample_rate: audio sample rate, in samples/sec
            name: name of the metric to record

        """

        wandb.log(
            {name: wandb.Audio(audio, sample_rate=sample_rate)},
            step=iterations,
        )

    def save(self, iterations: int) -> None:
        """flush metric values to the backend"""

        wandb.log(
            {v.name: v.value for v in self._scalars if v.name in self._dirty},
            step=iterations,
        )
        for v in self._scalars:
            v.reset()
        self._dirty.clear()


class Scalar:
    """scalar metric base class

    A scalar represents a single named metric that is collected and
    maintained as a running statistic throughout training. The scalar can
    be incrementally updated during each step and the current value can
    be sampled at any point. The scalar can also be reset periodically
    to discard the effects of earlier samples.

    Args:
        name: name of the metric

    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update(self, value: float) -> None:
        raise NotImplementedError()


class Mean(Scalar):
    """simple cumulative mean metric"""

    def __init__(self, name: str):
        super().__init__(name)

        self._value = np.nan
        self._count = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._value = np.nan
        self._count = 0

    def update(self, value: float) -> None:
        if self._count == 0:
            self._value = 0
        self._count += 1
        self._value += (value - self._value) / self._count


class Ema(Scalar):
    """exponentially-weighted moving average metric"""

    def __init__(self, name: str, alpha: float = 0.9):
        super().__init__(name)

        self._alpha = alpha
        self._value = np.nan

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = np.nan

    def update(self, value: float) -> None:
        if np.isnan(self._value):
            self._value = value
        else:
            self._value = self._value * self._alpha + value * (1.0 - self._alpha)


def grad_norm(model: torch.nn.Module) -> torch.Tensor:
    """compute the L2 norm of the gradients for a model"""

    norms = [v.grad.detach().norm(2) for v in model.parameters() if v.grad is not None]
    return torch.stack(norms).square().sum().sqrt()


def weight_norm(model: torch.nn.Module) -> torch.Tensor:
    """compute the L2 norm of the weights in a model"""

    norms = [
        v.norm(2)
        for n, v in model.named_parameters()
        if "bias" not in n and "weight_g" not in n
    ]
    return torch.stack(norms).square().sum().sqrt()
