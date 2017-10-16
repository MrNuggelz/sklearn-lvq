"""
==================================================
Generalized Relevance Learning Vector Quantization
==================================================
This example shows how GRLVQ projects and classifies.
The plot shows the target class of each data point
(big circle) and which class was predicted (smaller circle). It also
shows the prototypes (black diamond) and their labels (small point inside the diamond).
The projected data is shown in the right plot.

"""
import numpy as np
import matplotlib.pyplot as plt

from glvq import GrlvqModel,plot2d

print(__doc__)


nb_ppc = 100
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

print('GRLVQ:')
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.array([[0.3, 0], [0, 4]]),
                                  size=nb_ppc),
    np.random.multivariate_normal([4, 4], np.array([[0.3, 0], [0, 4]]),
                                  size=nb_ppc), axis=0)
grlvq = GrlvqModel()
grlvq.fit(toy_data, toy_label)
plot2d(grlvq, toy_data, toy_label, 1, 'grlvq')

print('classification accuracy:', grlvq.score(toy_data, toy_label))
plt.show()
