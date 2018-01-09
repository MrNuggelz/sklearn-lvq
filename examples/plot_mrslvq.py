"""
========================================
Generalized Learning Vector Quantization
========================================
This example shows the different glvq algorithms and how they project
different data sets. The data sets are chosen to show the strength of each
algorithm. Each plot shows for each data point which class it belongs to
(big circle) and which class it was classified to (smaller circle). It also
shows the prototypes (light blue circle). The projected data is shown in the
right plot.

"""
import numpy as np
import matplotlib.pyplot as plt

from glvq import plot2d
from glvq.mrslvq import MrslvqModel
from glvq.rslvq import RslvqModel

print(__doc__)

nb_ppc = 100
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

print('MRSLVQ:')
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.array([[5, 4], [4, 6]]),
                                  size=nb_ppc),
    np.random.multivariate_normal([9, 0], np.array([[5, 4], [4, 6]]),
                                  size=nb_ppc), axis=0)
mrslvq = MrslvqModel()
mrslvq.fit(toy_data, toy_label)
mrslvq.project(mrslvq.w_, 2)

print('classification accuracy:', mrslvq.score(toy_data, toy_label))
plot2d(mrslvq, toy_data, toy_label, 1, 'mrslvq')
plt.show()


