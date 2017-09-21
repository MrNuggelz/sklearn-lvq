# Generalized Learning Vector Quantization
Scikit-learn compatible implementation of GLVQ, GRLVQ, GLMVQ and LGMLVQ.

Compatible with Python2.7 and Python3.6

This implementation is based on the Matlab implementation
provided by Biehl, Schneider and Bunte (http://matlabserver.cs.rug.nl/gmlvqweb/web/)

## Important Links
scikit-learn - http://scikit-learn.org/

## Installation
Before you can install this module you need to install `numpy` and `scipy`:
```
pip install numpy scipy
```
To install this module run:
```
python setup.py install
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