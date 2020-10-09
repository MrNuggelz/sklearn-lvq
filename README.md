[![Build Status](https://travis-ci.org/MrNuggelz/sklearn-lvq.svg?branch=stable)](https://travis-ci.org/MrNuggelz/sklearn-lvq)
[![Build status](https://ci.appveyor.com/api/projects/status/qiwkue1x5lgll382?svg=true)](https://ci.appveyor.com/project/MrNuggelz/sklearn-glvq)
[![CircleCI](https://circleci.com/gh/MrNuggelz/sklearn-lvq.svg?style=shield)](https://circleci.com/gh/MrNuggelz/sklearn-lvq)
[![Coverage Status](https://coveralls.io/repos/github/MrNuggelz/sklearn-lvq/badge.svg)](https://coveralls.io/github/MrNuggelz/sklearn-lvq)
[![Coverage Status](https://readthedocs.org/projects/sklearn-lvq/badge/?version=latest)](https://sklearn-lvq.readthedocs.io/en/latest/?badge=latest)
# Warning

Repository and Package Name changed to sklearn-lvq!

# Generalized Learning Vector Quantization
Scikit-learn compatible implementation of GLVQ, GRLVQ, GMLVQ, LGMLVQ
RSLVQ, MRSLVQ and LMRSLVQ.

Compatible with Python2.7, Python3.6 and above.

This implementation is based on the Matlab implementation
provided by Biehl, Schneider and Bunte (http://matlabserver.cs.rug.nl/gmlvqweb/web/).

## Important Links
- scikit-learn (http://scikit-learn.org/)
- documentation (https://sklearn-lvq.readthedocs.io/en/latest/?badge=latest)

## Installation
To install this module run:
```
pip install .
```
or
```
pip install sklearn-lvq
```

To also install the extras, use
```bash
pip install .[docs,examples,tests]
```
or
```bash
pip install -U sklearn-lvq[docs,examples,tests]
```

## Examples
To run the examples:
```
pip install -U sklearn-lvq[examples]
```
The examples can be found in the examples directory.

## Testing
To run testss:
```
pip install -U sklearn-lvq[tests]
```
Tests are located in the `sklearn_lvq/tests` folder
and can be run with the `nosetests` command in the main directory.

## Documentation
To build the documentation locally, ensure that you have sphinx, sphinx-gallery,
pillow, sphinx_rt_theme, metric_learn and matplotlib by executing:

```
pip install -U sklearn-lvq[docs]
```
