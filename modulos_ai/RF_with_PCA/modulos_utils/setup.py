# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# coding: utf-8

"""

This package contains modulos_utils.

"""
import os

from setuptools import setup, find_packages

_NAME = "modulos-utils"
_VERSION = "0.4.6"

DIR = os.path.dirname(__file__)


def read_requirements(path):
    path = os.path.join(DIR, path)
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


_REQUIRES = read_requirements("requirements.txt")

setup(
    name=_NAME,
    version=_VERSION,
    description="Modulos utility library",
    author="modulos.ai",
    author_email="contact@modulos.ai",
    url="",
    keywords=["auto-ml modulos utils"],
    install_requires=_REQUIRES,
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={
        '': ['*.sh', '*.json', '*.md', '*.txt', '*.pdf', '*.yml']
    },
    entry_points={
        "console_scripts": [
            "ml-profiler = modulos_utils.scripts.profiler:"
            "_create_profiler_plots"
        ]
    },
    python_requires=">=3.7",
    license=("(C) Modulos AG (2019-2022). You may use and modify this code "
             "according to the Modulos AG Terms and Conditions. Please retain "
             "this header.")
)
