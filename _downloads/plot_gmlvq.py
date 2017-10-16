"""
===============================================
Generalized Matrix Learning Vector Quantization
===============================================
This example shows how GMLVQ projects and classifies.
The plot shows the target class of each data point
(big circle) and which class was predicted (smaller circle). It also
shows the prototypes (black diamond) and their labels (small point inside the diamond).
The projected data is shown in the right plot.

"""
import matplotlib.pyplot as plt
import numpy as np

from glvq import GmlvqModel, plot2d

print(__doc__)

nb_ppc = 100
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

print('GMLVQ:')
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.array([[5, 4], [4, 6]]),
                                  size=nb_ppc),
    np.random.multivariate_normal([9, 0], np.array([[5, 4], [4, 6]]),
                                  size=nb_ppc), axis=0)
gmlvq = GmlvqModel()
gmlvq.fit(toy_data, toy_label)
plot2d(gmlvq, toy_data, toy_label, 1, 'gmlvq')

print('classification accuracy:', gmlvq.score(toy_data, toy_label))
plt.show()
