# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from glvq.rslvq import RslvqModel


class MrslvqModel(RslvqModel):
    """Generalized Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.

    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful termination
        of bfgs.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    See also
    --------
    GrlvqModel, GmlvqModel, LgmlvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1, learn_rate=0.1, initial_matrix=None,
                 regularization=0.0, dim=None, max_iter=2500, display=False,
                 random_state=None):
        super(MrslvqModel, self).__init__(prototypes_per_class,
                                          initial_prototypes, sigma,
                                          learn_rate,
                                          max_iter, display, random_state)
        self.initial_matrix = initial_matrix
        self.initialdim = dim

    def _optimize(self, x, y, random_state):
        nb_epochs = 500
        nb_samples, nb_features = x.shape

        nb_prototypes, nb_features = self.w_.shape
        if self.initialdim is None:
            self.dim_ = nb_features
        elif not isinstance(self.initialdim, int) or self.initialdim <= 0:
            raise ValueError("dim must be an positive int")
        else:
            self.dim_ = self.initialdim

        if self.initial_matrix is None:
            if self.dim_ == nb_features:
                self.omega_ = np.eye(nb_features)
            else:
                self.omega_ = random_state.rand(self.dim_, nb_features) * 2 - 1
        else:
            self.omega_ = validation.check_array(self.initial_matrix)
            if self.omega_.shape[1] != nb_features:
                raise ValueError(
                    "initial matrix has wrong number of features\n"
                    "found=%d\n"
                    "expected=%d" % (self.omega_.shape[1], nb_features))

        for epoch in range(nb_epochs):
            order = random_state.permutation(nb_samples)
            c = self.learn_rate / self.sigma
            s = 0
            for i in range(nb_samples):
                index = order[i]
                xi = x[index]
                c_xi = y[index]
                for j in range(self.w_.shape[0]):
                    oo = self.omega_.T.dot(self.omega_)
                    d = (xi - self.w_[j])[np.newaxis].T
                    if self.c_w_[j] == c_xi:
                        ps = self.p(j, xi, c_xi) - self.p(j, xi)
                        change = (c * ps * oo.dot(d)).T[0]
                        self.w_[j] += change
                        self.omega_ += ps / self.sigma * self.omega_.dot(
                            d).dot(d.T)
                    else:
                        change = (c * self.p(j, xi) * oo.dot(d)).T[0]
                        self.w_[j] -= change
                        self.omega_ -= self.p(j, xi) / self.sigma * \
                                       self.omega_.dot(d).dot(d.T)
                    self.omega_ /= np.sqrt(
                        np.sum(np.diag(self.omega_.T.dot(self.omega_))))
                    oo = self.omega_.T.dot(self.omega_)

        self.n_iter_ = nb_epochs

    def f(self, x, w):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(self.omega_.T).dot(self.omega_).dot(d)
        return -d / (2 * self.sigma)

    def p(self, j, e, y=None):
        if y is None:
            fs = [self.f(e, w) for w in self.w_]
        else:
            fs = [self.f(e, self.w_[i]) for i in range(self.w_.shape[0]) if
                  self.c_w_[i] == y]
        fs_max = max(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        if s == 0:
            print(s)
            print("booom")
        o = np.math.exp(self.f(e, self.w_[j]) - fs_max) / s
        return o

    def _compute_distance(self, x, w=None, omega=None):
        if w is None:
            w = self.w_
        if omega is None:
            omega = self.omega_
        nb_samples = x.shape[0]
        nb_prototypes = w.shape[0]
        distance = np.zeros([nb_prototypes, nb_samples])
        for i in range(nb_prototypes):
            distance[i] = np.sum((x - w[i]).dot(omega.T) ** 2, 1)
        return distance.T

    def project(self, x, dims, print_variance_covered=False):
        """Projects the data input data X using the relevance matrix of trained
        model to dimension dim

        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        dims : int
          dimension to project to
        print_variance_covered : boolean
          flag to print the covered variance of the projection

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        v, u = np.linalg.eig(self.omega_.conj().T.dot(self.omega_))
        idx = v.argsort()[::-1]
        v = v[idx][:dims]
        if print_variance_covered:
            print('variance coverd by projection:',
                  v.sum() / v.sum() * 100)
        v = np.where(np.logical_and(v < 0,v > -0.1), 0, v) #set negative eigenvalues to 0
        if np.any(v < 0):
            print("boom")
        if np.any(np.sqrt(v)):
            print("pewpew")
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v))))
