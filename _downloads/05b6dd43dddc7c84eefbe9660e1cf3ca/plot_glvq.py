"""
========================================
Generalized Learning Vector Quantization
========================================
This example shows how GLVQ classifies.
The plot shows the target class of each data point
(big circle) and which class was predicted (smaller circle). It also
shows the prototypes (black diamond) and their labels (small point inside the diamond).

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn_lvq import GlvqModel
from sklearn_lvq.utils import plot2d

print(__doc__)

nb_ppc = 100
print('GLVQ:')
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=nb_ppc),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=nb_ppc), axis=0)
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

glvq = GlvqModel()
glvq.fit(toy_data, toy_label)
plot2d(glvq, toy_data, toy_label, 1, 'glvq')

print('classification accuracy:', glvq.score(toy_data, toy_label))
plt.show()
