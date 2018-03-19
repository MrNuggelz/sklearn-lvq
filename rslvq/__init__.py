# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause
import matplotlib

matplotlib.use('Agg')

from .rslvq import RslvqModel
from .mrslvq import MrslvqModel
from .lmrslvq import LmrslvqModel
from plot_2d import plot2d
__all__ = ['RslvqModel','MrslvqModel','LmrslvqModel','plot2d']
__version__ = '1.0'
