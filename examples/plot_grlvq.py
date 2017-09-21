"""
==================================================
Generalized Relevance Learning Vector Quantization
==================================================
This example shows the different glvq algorithms and how they project
different data sets. The data sets are chosen to show the strength of each
algorithm. Each plot shows for each datapoint which class it belongs to
(big circle) and which class it was classified to (smaller circle). It also
shows the prototypes (light blue circle). The projected data is shown in the
right plot.

"""
import numpy as np
import matplotlib.pyplot as plt

from glvq import GrlvqModel

print(__doc__)


def project_plot2d(model, X, y, figure, title=""):
    """
    Projects the input data to two dimensions and plots it. The projection is
    done using the relevances of the given glvq model.

    :param model: GlvqModel that has relevances
        (GrlvqModel,GmlvqModel,LgmlvqModel)
    :param X: Input data
    :param y: Input data target
    :param figure: the figure to plot on
    :param title: the title to use, optional
    :return: None
    """
    dim = 2
    f = plt.figure(figure)
    f.suptitle(title)
    pred = model.predict(X)

    if hasattr(model, 'omegas_'):
        nb_prototype = model.w_.shape[0]
        ax = f.add_subplot(1, nb_prototype + 1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
        ax.scatter(X[:, 0], X[:, 1], c=pred, marker='.')
        ax.scatter(model.w_[:, 0], model.w_[:, 1])
        ax.axis('equal')
        for i in range(nb_prototype):
            X_p = model.project(X, i, dim, print_variance_covered=True)
            w_p = model.project(model.w_[i], i, dim)

            ax = f.add_subplot(1, nb_prototype + 1, i + 2)
            ax.scatter(X_p[:, 0], X_p[:, 1], c=y, alpha=0.2)
            # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
            ax.scatter(w_p[0], w_p[1], marker='D', s=20)
            ax.axis('equal')

    else:
        ax = f.add_subplot(121)
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
        ax.scatter(X[:, 0], X[:, 1], c=pred, marker='.')
        ax.scatter(model.w_[:, 0], model.w_[:, 1])
        ax.axis('equal')
        X_p = model.project(X, dim, print_variance_covered=True)
        w_p = model.project(model.w_, dim)

        ax = f.add_subplot(122)
        ax.scatter(X_p[:, 0], X_p[:, 1], c=y, alpha=0.5)
        # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
        ax.scatter(w_p[:, 0], w_p[:, 1], marker='D', s=20)
        ax.axis('equal')
    f.show()


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
project_plot2d(grlvq, toy_data, toy_label, 1, 'grlvq')

print('classification accuracy:', grlvq.score(toy_data, toy_label))
plt.show()
