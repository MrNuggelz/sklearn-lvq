# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from .glvq import GlvqModel
from .grlvq import GrlvqModel
from .gmlvq import GmlvqModel
from .lgmlvq import LgmlvqModel

__all__ = ['GlvqModel', 'GrlvqModel', 'GmlvqModel', 'LgmlvqModel']
__version__ = '1.0'
