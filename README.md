[![Build Status](https://travis-ci.org/MrNuggelz/sklearn-glvq.svg?branch=master)](https://travis-ci.org/MrNuggelz/sklearn-glvq)
[![Build status](https://ci.appveyor.com/api/projects/status/qiwkue1x5lgll382?svg=true)](https://ci.appveyor.com/project/MrNuggelz/sklearn-glvq)
[![CircleCI](https://circleci.com/gh/MrNuggelz/sklearn-glvq.svg?style=shield)](https://circleci.com/gh/MrNuggelz/sklearn-glvq)
[![Coverage Status](https://coveralls.io/repos/github/MrNuggelz/sklearn-glvq/badge.svg?branch=master)](https://coveralls.io/github/MrNuggelz/sklearn-glvq?branch=master)

# Generalized Learning Vector Quantization
Scikit-learn compatible implementation of GLVQ, GRLVQ, GLMVQ and LGMLVQ.

Compatible with Python2.7 and Python3.6

This implementation is based on the Matlab implementation
provided by Biehl, Schneider and Bunte (http://matlabserver.cs.rug.nl/gmlvqweb/web/)

## Important Links
- scikit-learn (http://scikit-learn.org/)
- documentation (https://mrnuggelz.github.io/sklearn-glvq/)

## Installation
Before you can install this module you need to install `numpy` and `scipy`:
```
pip install numpy scipy
```
To install this module run:
```
python setup.py install
```
or
```
pip install sklearn-glvq
```

## Examples
To run the examples `matplotlib` is needed
```
pip install matplotlib
```
The examples can be found in the examples directory.

## Testing
Requires installation of `nose` package.
```
pip install nose
```
Tests are located in the `glvq/tests` folder
and can be run with the `nosetests` command in the main directory.

## Documentation
To build the documentation locally, ensure that you have sphinx, sphinx-gallery, pillow, sphinx_rt_theme, metric_learn and matplotlib by executing:

```
pip install sphinx pillow sphinx-gallery sphinx_rtd_theme metric_learn
```
