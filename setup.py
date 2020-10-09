"""Install sklearn-lvq."""

from __future__ import print_function
import sys
from setuptools import setup
from setuptools import find_packages

PROJECT_URL = "https://github.com/MrNuggelz/sklearn-lvq"
DOWNLOAD_URL = "https://github.com/MrNuggelz/sklearn-lvq/releases/tag/1.1.0"

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name="sklearn-lvq",
      version="1.1.1",
      description="Scikit-Learn compatible Generalized Learning Vector "
      "Quantization (GLVQ) and Robust Soft Learning Vector Quantization "
      "implementation.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Joris Jensun",
      author_email="jjensen@techfak.uni-bielefeld.de",
      url=PROJECT_URL,
      download_url=DOWNLOAD_URL,
      license="BSD-3-Clause",
      install_requires=[
          "numpy>=1.9.1",
          "scikit-learn>=0.17",
      ],
      extras_require={
          "examples": ["matplotlib>=2.0.2"],
          "tests": ["nose"],
          "docs": [
              "sphinx",
              "pillow",
              "sphinx-gallery",
              "sphinx_rtd_theme",
              "metric_learn",
              "matplotlib>=2.0.2",
              "numpydoc",
          ],
      },
      classifiers=[
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Software Development :: Libraries",
          "Topic :: Software Development :: Libraries :: Python Modules"
      ],
      packages=find_packages())
