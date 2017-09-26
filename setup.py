from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='sklearn-glvq',
      version='1.0.1',
      description='sklearn compatible Generalized Learning Vector '
                  'Quantization implementation',
      author='Joris Jensen',
      packages=['glvq'],
      install_requires=INSTALL_REQUIRES,
      author_email='jjensen@techfak.uni-bielefeld.de',
      )
