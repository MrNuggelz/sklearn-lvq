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

from glvq import RslvqModel
from glvq.plot_2d import to_tango_colors, tango_color

print(__doc__)

nb_ppc = 100
print('RSLVQ:')
x = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=nb_ppc),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=nb_ppc), axis=0)
y = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

rslvq = RslvqModel(initial_prototypes=[[5,0,0],[0,0,1]])
rslvq.fit(x, y)
print('classification accuracy:', rslvq.score(x, y))

pred = rslvq.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=to_tango_colors(y), alpha=0.5)
plt.scatter(x[:, 0], x[:, 1], c=to_tango_colors(pred), marker='.')
plt.scatter(rslvq.w_[:, 0], rslvq.w_[:, 1],
           c=tango_color('aluminium', 5), marker='D')
plt.scatter(rslvq.w_[:, 0], rslvq.w_[:, 1],
           c=to_tango_colors(rslvq.c_w_, 0), marker='.')
plt.axis('equal')

plt.show()
