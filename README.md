# HiFi-GAN+
This project is an unoffical implementation of the HiFi-GAN+ model for
audio bandwidth extension, from the paper
[Bandwidth Extension is All You Need](https://doi.org/10.1109/ICASSP39728.2021.9413575)
by Jiaqi Su, Yunyun Wang, Adam Finkelstein, and Zeyu Jin.

The model takes a band-limited audio signal (usually 8/16/24kHz) and
attempts to reconstruct the high frequency components needed to restore
a full-band signal at 48kHz. This is useful for upsampling low-rate
outputs from upstream tasks like text-to-speech, voice conversion, etc. or
enhancing audio that was filtered to remove high frequency noise. For more
information, please see this
[blog post](https://brentspell.com/2022/hifi-gan-bwe/).

## Status
[![PyPI](https://badge.fury.io/py/hifi-gan-bwe.svg)](https://badge.fury.io/py/hifi-gan-bwe)
![Tests](https://github.com/brentspell/hifi-gan-bwe/workflows/test/badge.svg)
[![Coveralls](https://coveralls.io/repos/github/brentspell/hifi-gan-bwe/badge.svg?branch=main)](https://coveralls.io/github/brentspell/hifi-gan-bwe)
[![DOI](https://zenodo.org/badge/DOI/10.1109/ICASSP39728.2021.9413575.svg)](https://doi.org/10.1109/ICASSP39728.2021.9413575)

[![Wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/brentspell/hifi-gan-bwe?workspace=user-brentspell)
[![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/brentspell/hifi-gan-bwe)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dlw9SipnWZ0xqTquJ_-YfFqUdGD40b0a?usp=sharing)

## Usage

The example below uses a pretrained HiFi-GAN+ model to upsample a 1 second
24kHz sawtooth to 48kHz.

```python
import torch
from hifi_gan_bwe import BandwidthExtender

model = BandwidthExtender.from_pretrained("hifi-gan-bwe-10-42890e3-vctk-48kHz")

fs = 24000
x = torch.full([fs], 261.63 / fs).cumsum(-1) % 1.0 - 0.5
y = model(x, fs)
```

There is a [Gradio demo](https://huggingface.co/spaces/brentspell/hifi-gan-bwe)
on HugggingFace Spaces where you can upload audio clips and run the model. You
can also run the model on Colab with this
[notebook](https://colab.research.google.com/drive/1dlw9SipnWZ0xqTquJ_-YfFqUdGD40b0a?usp=sharing).

### Running with pipx
The HiFi-GAN+ [library](https://pypi.org/project/hifi-gan-bwe/) can be run
directly from PyPI if you have the [pipx](https://github.com/pypa/pipx)
application installed. The following script uses a hosted pretrained model
to upsample an MP3 file to 48kHz. The input audio can be in any format
supported by the [audioread](https://pypi.org/project/audioread) library, and
the output can be in any format supported by
[soundfile](https://pypi.org/project/SoundFile).

```shell
pipx run --python=python3.9 hifi-gan-bwe \
  hifi-gan-bwe-10-42890e3-vctk-48kHz \
  input.mp3 \
  output.wav
```

### Running in a Virtual Environment
If you have a Python 3.9 virtual environment installed, you can install
the HiFi-GAN+ library into it and run synthesis, training, etc. using it.

```shell
pip install hifi-gan-bwe

hifi-synth hifi-gan-bwe-10-42890e3-vctk-48kHz input.mp3 output.wav
```

## Pretrained Models
The following models can be loaded with `BandwidthExtender.from_pretrained`
and used for audio upsampling. You can also download the model file from
the link and use it offline.

|Name|Sample Rate|Parameters|Wandb Metrics|Notes|
|-|-|-|-|-|
|[hifi-gan-bwe-10-42890e3-vctk-48kHz](https://cdn.brentspell.com/models/hifi-gan-bwe/hifi-gan-bwe-10-42890e3-vctk-48kHz.pt)|48kHz|1M|[bwe-10-42890e3](https://wandb.ai/brentspell/hifi-gan-bwe/runs/bwe-10-42890e3?workspace=user-brentspell)|Same as bwe-05, but uses bandlimited interpolation for upsampling, for reduced noise and aliasing. Uses the same parameters as resampy's [kaiser_best](https://github.com/bmcfee/resampy/blob/5f46888e8b52402f2c62f374b39b93e0743543ad/resampy/filters.py#L9) mode.|
|[hifi-gan-bwe-11-d5f542d-vctk-8kHz-48kHz](https://cdn.brentspell.com/models/hifi-gan-bwe/hifi-gan-bwe-11-d5f542d-vctk-8kHz-48kHz.pt)|48kHz|1M|[bwe-11-d5f542d](https://wandb.ai/brentspell/hifi-gan-bwe/runs/bwe-11-d5f542d?workspace=user-brentspell)|Same as bwe-10, but trained only on 8kHz sources, for specialized upsampling.|
|[hifi-gan-bwe-12-b086d8b-vctk-16kHz-48kHz](https://cdn.brentspell.com/models/hifi-gan-bwe/hifi-gan-bwe-12-b086d8b-vctk-16kHz-48kHz.pt)|48kHz|1M|[bwe-12-b086d8b](https://wandb.ai/brentspell/hifi-gan-bwe/runs/bwe-12-b086d8b?workspace=user-brentspell)|Same as bwe-10, but trained only on 16kHz sources, for specialized upsampling.|
|[hifi-gan-bwe-13-59f00ca-vctk-24kHz-48kHz](https://cdn.brentspell.com/models/hifi-gan-bwe/hifi-gan-bwe-13-59f00ca-vctk-24kHz-48kHz.pt)|48kHz|1M|[bwe-13-59f00ca](https://wandb.ai/brentspell/hifi-gan-bwe/runs/bwe-13-59f00ca?workspace=user-brentspell)|Same as bwe-10, but trained only on 24kHz sources, for specialized upsampling.|
|[hifi-gan-bwe-05-cd9f4ca-vctk-48kHz](https://cdn.brentspell.com/models/hifi-gan-bwe/hifi-gan-bwe-05-cd9f4ca-vctk-48kHz.pt)|48kHz|1M|[bwe-05-cd9f4ca](https://wandb.ai/brentspell/hifi-gan-bwe/runs/bwe-05-cd9f4ca?workspace=user-brentspell)|Trained for 200K iterations on the VCTK speech dataset with noise agumentation from the DNS Challenge dataset.|

## Training
If you want to train your own model, you can use any of the methods above
to install/run the library or fork the repo and run the script commands
locally. The following commands are supported:

|Name|Description|
|-|-|
|[hifi-train](https://github.com/brentspell/hifi-gan-bwe/blob/main/hifi_gan_bwe/scripts/train.py)|Starts a new training run, pass in a name for the run.|
|[hifi-clone](https://github.com/brentspell/hifi-gan-bwe/blob/main/hifi_gan_bwe/scripts/clone.py)|Clone an existing training run at a given or the latest checkpoint.|
|[hifi-export](https://github.com/brentspell/hifi-gan-bwe/blob/main/hifi_gan_bwe/scripts/export.py)|Optimize a model for inference and export it to a PyTorch model file (.pt).|
|[hifi-synth](https://github.com/brentspell/hifi-gan-bwe/blob/main/hifi_gan_bwe/scripts/synth.py)|Run model inference using a trained model on a source audio file.|

For example, you might start a new training run called `bwe-01` with the
following command:

```bash
hifi-train 01
```

To train a model, you will first need to download the
[VCTK](https://datashare.ed.ac.uk/handle/10283/2950) and
[DNS Challenge](https://github.com/microsoft/DNS-Challenge)
datasets. By default, these datasets are assumed to be in the `./data/vctk`
and `./data/dns` directories. See `train.py` for how to specify your own
training data directories. If you want to use a custom training dataset,
you can implement a dataset wrapper in datasets.py.

The training scripts use [wandb.ai](https://wandb.ai/) for experiment tracking
and visualization. Wandb metrics can be disabled by passing `--no_wandb` to
the training script. All of my own experiment results are publicly available at
[wandb.ai/brentspell/hifi-gan-bwe](https://wandb.ai/brentspell/hifi-gan-bwe?workspace=user-brentspell).

Each training run is identified by a name and a git hash
(ex: `bwe-01-8abbca9`). The git hash is used for simple experiment tracking,
reproducibility, and model provenance. Using git to manage experiments also
makes it easy to change model hyperparameters by simply changing the code,
making a commit, and starting the training run. This is why there is no
hyperparameter configuration file in the project, since I often end up
having to change the code anyway to run interesting experiments.

## Development

### Setup
The following script creates a virtual environment using
[pyenv](https://github.com/pyenv/pyenv) for the project and installs
dependencies.

```bash
pyenv install 3.9.10
pyenv virtualenv 3.9.10 hifi-gan-bwe
pip install -r requirements.txt
```

If you want to run the `hifi-*` scripts described above in development,
you can install the package locally:

```bash
pip install -e .
```

You can then run tests, etc. follows:

```bash
pytest --cov=hifi_gan_bwe
black .
isort --profile=black .
flake8 .
mypy .
```

These checks are also included in the
[pre-commit](https://pypi.org/project/pre-commit/) configuration for the
project, so you can set them up to run automatically on commit by running

```bash
pre-commit install
```

## Acknowledgements
The original research on the HiFi-GAN+ model is not my own, and all credit
goes to the paper's authors. I also referred to kan-bayashi's excellent
[Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
implementation, specifically the WaveNet module. If you use this code, please
cite the original paper:

```bibtex
@inproceedings{su2021bandwidth,
  title={Bandwidth extension is all you need},
  author={Su, Jiaqi and Wang, Yunyun and Finkelstein, Adam and Jin, Zeyu},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={696--700},
  year={2021},
  organization={IEEE},
  url={https://doi.org/10.1109/ICASSP39728.2021.9413575},
}
```

## License
Copyright © 2022 Brent M. Spell

Licensed under the MIT License (the "License"). You may not use this
package except in compliance with the License. You may obtain a copy of the
License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
