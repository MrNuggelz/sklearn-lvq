"""
========================================
Robust Soft Learning Vector Quantization
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

from sklearn_lvq import RslvqModel
from sklearn_lvq.utils import plot2d

print(__doc__)

nb_ppc = 100
print('RSLVQ:')
x = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=nb_ppc),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=nb_ppc), axis=0)
y = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

rslvq = RslvqModel(initial_prototypes=[[5,0,0],[0,0,1]])
rslvq.fit(x, y)
plot2d(rslvq, x, y, 1, 'rslvq')
print('classification accuracy:', rslvq.score(x, y))

plt.show()
