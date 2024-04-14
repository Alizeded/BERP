#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    description="BERP: A Blind Estimator of Room acoustic and physical Parameters.",
    author="Lucianius L. Wang",
    author_email="lijun.wang@jaist.ac.jp",
    url="https://github.com/Alizeded/BERP",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
