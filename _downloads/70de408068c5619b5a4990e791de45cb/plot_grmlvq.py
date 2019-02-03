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

from sklearn_lvq import GrmlvqModel
from sklearn_lvq.utils import plot2d

print(__doc__)

nb_ppc = 100
toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

print('GRMLVQ:')
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.array([[5, 4], [4, 6]]),
                                  size=nb_ppc),
    np.random.multivariate_normal([9, 0], np.array([[5, 4], [4, 6]]),
                                  size=nb_ppc), axis=0)
grmlvq = GrmlvqModel()
grmlvq.fit(toy_data, toy_label)
print(grmlvq.lambda_)
plot2d(grmlvq, toy_data, toy_label, 1, 'grmlvq')

print('classification accuracy:', grmlvq.score(toy_data, toy_label))
plt.show()
