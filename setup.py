#!/usr/bin/env python

from distutils.core import setup

import setuptools  # noqa, required for markdown manifest

setup(
    name="hifi-gan-bwe",
    version="0.1.14",
    description=(
        "Unofficial implementation of the HiFi-GAN+ model "
        "for audio bandwidth extension"
    ),
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/brentspell/hifi-gan-bwe/",
    author="Brent M. Spell",
    author_email="brent@brentspell.com",
    packages=["hifi_gan_bwe", "hifi_gan_bwe.scripts"],
    setup_requires=[],
    install_requires=[
        "audioread",
        "matplotlib",
        "numpy",
        "requests",
        "soundfile",
        "torch",
        "torchaudio",
        "tqdm",
        "wandb",
    ],
    entry_points={
        "console_scripts": [
            "hifi-gan-bwe = hifi_gan_bwe.scripts.synth:main",
            "hifi-clone = hifi_gan_bwe.scripts.clone:main",
            "hifi-export = hifi_gan_bwe.scripts.export:main",
            "hifi-synth = hifi_gan_bwe.scripts.synth:main",
            "hifi-train = hifi_gan_bwe.scripts.train:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
