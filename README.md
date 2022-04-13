# HiFi-GAN+
This package is an unoffical implementation of the HiFi-GAN+ model for
audio bandwidth extension

## Status
[![PyPI](https://badge.fury.io/py/hifi-gan-bwe.svg)](https://badge.fury.io/py/hifi-gan-bwe)
![Tests](https://github.com/brentspell/hifi-gan-bwe/actions/workflows/test.yml/badge.svg)
[![Coveralls](https://coveralls.io/repos/github/brentspell/hifi-gan-bwe/badge.svg?branch=main)](https://coveralls.io/github/brentspell/hifi-gan-bwe)

### Install with pip
```bash
pip install hifi-gan-bwe
```

## Usage


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

You can then run tests, etc. follows:

```bash
pytest --cov=hifi_gan_bwe
black .
isort --profile=black .
flake8 .
mypy .
```

These can also be used with the [pre-commiit](https://pypi.org/project/pre-commit/)
library to run all checks at commit time.

### Deployment
The project uses setup.py for installation and is deployed to
[PyPI](https://pypi.org/project/hifi-gan-bwe). The source distribution can be
built for deployment with the following command:

```bash
python setup.py clean --all
rm -r ./dist
python setup.py sdist
```

The distribution can then be uploaded to PyPI using twine.

```bash
twine upload --repository-url=https://upload.pypi.org/legacy/ dist/*
```

For deployment testing, the following command can be used to upload to the
PyPI test repository:

```bash
twine upload --repository-url=https://test.pypi.org/legacy/ dist/*
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
