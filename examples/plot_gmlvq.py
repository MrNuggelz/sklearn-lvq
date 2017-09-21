"""
===============================================
Generalized Matrix Learning Vector Quantization
===============================================
This example shows the different glvq algorithms and how they project
different data sets. The data sets are chosen to show the strength of each
algorithm. Each plot shows for each data point which class it belongs to
(big circle) and which class it was classified to (smaller circle). It also
shows the prototypes (light blue circle). The projected data is shown in the
right plot.

"""
import numpy as np
import matplotlib.pyplot as plt

from glvq import GmlvqModel

print(__doc__)


def project_plot2d(model, x, y, figure, title=""):
    """
    Projects the input data to two dimensions and plots it. The projection is
    done using the relevances of the given glvq model.

    :param model: GlvqModel that has relevances
        (GrlvqModel,GmlvqModel,LgmlvqModel)
    :param x: Input data
    :param y: Input data target
    :param figure: the figure to plot on
    :param title: the title to use, optional
    :return: None
    """
    dim = 2
    f = plt.figure(figure)
    f.suptitle(title)
    pred = model.predict(x)

    if hasattr(model, 'omegas_'):
        nb_prototype = model.w_.shape[0]
        ax = f.add_subplot(1, nb_prototype + 1, 1)
        ax.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5)
        ax.scatter(x[:, 0], x[:, 1], c=pred, marker='.')
        ax.scatter(model.w_[:, 0], model.w_[:, 1])
        ax.axis('equal')
        for i in range(nb_prototype):
            x_p = model.project(x, i, dim, print_variance_covered=True)
            w_p = model.project(model.w_[i], i, dim)

            ax = f.add_subplot(1, nb_prototype + 1, i + 2)
            ax.scatter(x_p[:, 0], x_p[:, 1], c=y, alpha=0.2)
            # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
            ax.scatter(w_p[0], w_p[1], marker='D', s=20)
            ax.axis('equal')

    else:
        ax = f.add_subplot(121)
        ax.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5)
        ax.scatter(x[:, 0], x[:, 1], c=pred, marker='.')
        ax.scatter(model.w_[:, 0], model.w_[:, 1])
        ax.axis('equal')
        x_p = model.project(x, dim, print_variance_covered=True)
        w_p = model.project(model.w_, dim)

        ax = f.add_subplot(122)
        ax.scatter(x_p[:, 0], x_p[:, 1], c=y, alpha=0.5)
        # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
        ax.scatter(w_p[:, 0], w_p[:, 1], marker='D', s=20)
        ax.axis('equal')
    f.show()


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
project_plot2d(gmlvq, toy_data, toy_label, 1, 'gmlvq')

print('classification accuracy:', gmlvq.score(toy_data, toy_label))
plt.show()
