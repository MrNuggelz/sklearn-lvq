# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
from scipy.optimize import minimize
from sklearn.utils import validation
from .rslvq import RslvqModel


class MrslvqModel(RslvqModel):
    """Matrix Robust Soft Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different numbers
        per class.

    initial_prototypes : array-like,
     shape =  [n_prototypes, n_features + 1], optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype

    initial_matrix : array-like, shape = [dim, n_features], optional
        Relevance matrix to start with.
        If not given random initialization for rectangular matrix and unity
        for squared matrix.

    regularization : float, optional (default=0.0)
        Value between 0 and 1. Regularization is done by the log determinant
        of the relevance matrix. Without regularization relevances may
        degenerate to zero.

    initialdim : int, optional (default=nb_features)
        Maximum rank or projection dimensions

    sigma : float, optional (default=0.5)
        Variance for the distribution.

    max_iter : int, optional (default=500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful
        termination of l-bfgs-b.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional (default=None)
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

    dim_ : int
        Maximum rank or projection dimensions

    omega_ : array-like, shape = [dim, n_features]
        Relevance matrix

    See also
    --------
    RslvqModel, LmrslvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 initial_matrix=None, regularization=0.0, initialdim=None,
                 sigma=1, max_iter=1000, gtol=1e-5, display=False, random_state=None):
        super(MrslvqModel, self).__init__(sigma=sigma,
                                          random_state=random_state,
                                          prototypes_per_class=prototypes_per_class,
                                          initial_prototypes=initial_prototypes,
                                          gtol=gtol, display=display, max_iter=max_iter)
        self.regularization = regularization
        self.initial_matrix = initial_matrix
        self.initialdim = initialdim

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state, lr_relevances=0, lr_prototypes=1):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        variables = variables.reshape(variables.size // n_dim, n_dim)
        prototypes = variables[:nb_prototypes]
        omega = variables[nb_prototypes:]

        g = np.zeros(variables.shape)

        if lr_relevances > 0:
            gw = np.zeros([omega.shape[0], n_dim])

        oo = omega.T.dot(omega)
        c = 1 / self.sigma
        for i in range(n_data):
            xi = training_data[i]
            c_xi = label_equals_prototype[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])[np.newaxis].T
                p = self._p(j, xi, prototypes=prototypes, omega=omega)
                if self.c_w_[j] == c_xi:
                    pj = self._p(j, xi, prototypes=prototypes, y=c_xi, omega=omega)
                if lr_prototypes > 0:
                    if self.c_w_[j] == c_xi:
                        g[j] += (c * (pj - p) * oo.dot(d)).ravel()
                    else:
                        g[j] -= (c * p * oo.dot(d)).ravel()
                if lr_relevances > 0:
                    if self.c_w_[j] == c_xi:
                        gw -= (pj - p) / self.sigma * (omega.dot(d).dot(d.T))
                    else:
                        gw += p / self.sigma * (omega.dot(d).dot(d.T))
        f3 = 0
        if self.regularization:
            f3 = np.linalg.pinv(omega).conj().T
        if lr_relevances > 0:
            g[nb_prototypes:] = 2 / n_data \
                                * lr_relevances * gw - self.regularization * f3
        if lr_prototypes > 0:
            g[:nb_prototypes] = 1 / n_data * lr_prototypes \
                                * g[:nb_prototypes].dot(omega.T.dot(omega))
        g *= -(1 + 0.0001 * (random_state.rand(*g.shape) - 0.5))
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        variables = variables.reshape(variables.size // n_dim, n_dim)
        prototypes = variables[:nb_prototypes]
        omega = variables[nb_prototypes:]

        out = 0
        for i in range(n_data):
            xi = training_data[i]
            y = label_equals_prototype[i]
            fs = [self._costf(xi, w, omega=omega) for w in
                  prototypes]
            # fs = []
            # for w in prototypes:
            #     fs.append(self.costf(xi,w,self.sigma,omega=omega))
            fs_max = max(fs)
            s1 = sum([np.math.exp(fs[i] - fs_max) for i in range(len(fs))
                      if self.c_w_[i] == y])
            s2 = sum([np.math.exp(f - fs_max) for f in fs])
            s1 += 0.0000001
            s2 += 0.0000001
            out += np.math.log(s1 / s2)
        return -out

    def _optimize(self, x, y, random_state):
        if not isinstance(self.regularization,
                          float) or self.regularization < 0:
            raise ValueError("regularization must be a positive float ")
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

        variables = np.append(self.w_, self.omega_, axis=0)
        label_equals_prototype = y
        method = 'l-bfgs-b'
        method = 'bfgs'
        res = minimize(
            fun=lambda vs:
            self._optfun(vs, x, label_equals_prototype=y),
            jac=lambda vs:
            self._optgrad(vs, x, label_equals_prototype=y,
                          random_state=random_state,
                          lr_prototypes=1, lr_relevances=0),
            method=method, x0=variables,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        n_iter = res.nit
        res = minimize(
            fun=lambda vs:
            self._optfun(vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs:
            self._optgrad(vs, x, label_equals_prototype=label_equals_prototype,
                          random_state=random_state,
                          lr_prototypes=0, lr_relevances=1),
            method=method, x0=res.x,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        res = minimize(
            fun=lambda vs:
            self._optfun(vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs:
            self._optgrad(vs, x, label_equals_prototype=label_equals_prototype,
                          random_state=random_state,
                          lr_prototypes=1, lr_relevances=1),
            method=method, x0=res.x,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        out = res.x.reshape(res.x.size // nb_features, nb_features)
        self.w_ = out[:nb_prototypes]
        self.omega_ = out[nb_prototypes:]
        self.omega_ /= np.math.sqrt(
            np.sum(np.diag(self.omega_.T.dot(self.omega_))))
        self.n_iter_ = n_iter

    def _costf(self, x, w, **kwargs):
        if 'omega' in kwargs:
            omega = kwargs['omega']
        else:
            omega = self.omega_
        d = (x - w)[np.newaxis].T
        d = d.T.dot(omega.T).dot(omega).dot(d)
        return -d / (2 * self.sigma)

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
        v = np.where(np.logical_and(v < 0, v > -0.1), 0,
                     v)  # set negative eigenvalues to 0
        if np.any(v < 0):
            print("boom")
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v))))
