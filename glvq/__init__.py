# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause
import matplotlib

matplotlib.use('Agg')

from .glvq import GlvqModel
from .grlvq import GrlvqModel
from .gmlvq import GmlvqModel
from .lgmlvq import LgmlvqModel
from .rslvq import RslvqModel
from .mrslvq import MrslvqModel
from .lmrslvq import LmrslvqModel
from .plot_2d import plot2d
__all__ = ['GlvqModel', 'GrlvqModel', 'GmlvqModel', 'LgmlvqModel','RslvqModel','MrslvqModel','LmrslvqModel','plot2d']
__version__ = '1.0'
