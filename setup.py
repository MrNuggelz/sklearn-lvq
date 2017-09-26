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

CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Operating System :: POSIX',
    'Operating System :: MacOS',
    'Operating System :: Unix',
    'Operating System :: Microsoft :: Windows',
]

version='1.0.3'

setup(name='sklearn-glvq',
      version=version,
      description='sklearn compatible Generalized Learning Vector '
                  'Quantization implementation',
      author='Joris Jensen',
      url='https://github.com/MrNuggelz/sklearn-glvq',
      download_url='https://github.com/MrNuggelz/sklearn-glvq/releases/tag/{}'.format(version),
      tests_require=['nose'],
      platforms=['any'],
      license='BSD-3-Clause',
      packages=['glvq'],
      install_requires=INSTALL_REQUIRES,
      author_email='jjensen@techfak.uni-bielefeld.de',
      classifiers=CLASSIFIERS,
      )
