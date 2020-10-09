# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
from scipy.optimize import minimize
from sklearn.utils import validation

from .rslvq import RslvqModel


class LmrslvqModel(RslvqModel):
    """Localized Matrix Robust Soft Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.

    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.

    initial_matrices : list of array-like, optional
        Matrices to start with. If not given random initialization

    regularization : float or array-like, shape = [n_classes/n_prototypes],
     optional (default=0.0)
        Values between 0 and 1. Regularization is done by the log determinant
        of the relevance matrix. Without regularization relevances may
        degenerate to zero.

    initialdim : int, optional
        Maximum rank or projection dimensions

    classwise : boolean, optional
        If true, each class has one relevance matrix.
        If false, each prototype has one relevance matrix.

    sigma : float, optional (default=0.5)
        Variance for the distribution.

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

    omegas_ : list of array-like
        Relevance Matrices

    dim_ : list of int
        Maximum rank of projection

    regularization_ : array-like, shape = [n_classes/n_prototypes]
        Values between 0 and 1
    See also
    --------
    RslvqModel, MrslvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 initial_matrices=None, regularization=0.0, initialdim=None,
                 classwise=False, sigma=1, max_iter=2500, gtol=1e-5, display=False,
                 random_state=None):
        super(LmrslvqModel, self).__init__(sigma=sigma,
                                           random_state=random_state,
                                           prototypes_per_class=prototypes_per_class,
                                           initial_prototypes=initial_prototypes,
                                           gtol=gtol, display=display, max_iter=max_iter)
        self.regularization = regularization
        self.initial_matrices = initial_matrices
        self.classwise = classwise
        self.initialdim = initialdim

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state, lr_relevances=0, lr_prototypes=1):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        variables = variables.reshape(variables.size // n_dim, n_dim)
        prototypes = variables[:nb_prototypes]
        # dim to indices
        indices = []
        for i in range(len(self.dim_)):
            indices.append(sum(self.dim_[:i + 1]))
        omegas = np.split(variables[nb_prototypes:], indices[:-1])  # .conj().T

        g = np.zeros(variables.shape)

        if lr_relevances > 0:
            gw = []
            for i in range(len(omegas)):
                gw.append(np.zeros(omegas[i].shape))
        c = 1 / self.sigma
        for i in range(n_data):
            xi = training_data[i]
            c_xi = label_equals_prototype[i]
            for j in range(prototypes.shape[0]):
                if len(omegas) == nb_prototypes:
                    omega_index = j
                else:
                    omega_index = np.where(self.classes_ == self.c_w_[j])[0][0]
                oo = omegas[omega_index].T.dot(omegas[omega_index])
                d = (xi - prototypes[j])[np.newaxis].T
                p = self._p(j, xi, prototypes=prototypes, omega=omegas[omega_index])
                if self.c_w_[j] == c_xi:
                    pj = self._p(j, xi, prototypes=prototypes, y=c_xi,
                                 omega=omegas[omega_index])
                if lr_prototypes > 0:
                    if self.c_w_[j] == c_xi:
                        g[j] += (c * (pj - p) * oo.dot(d)).ravel()
                    else:
                        g[j] -= (c * p * oo.dot(d)).ravel()
                if lr_relevances > 0:
                    if self.c_w_[j] == c_xi:
                        gw -= (pj - p) / self.sigma * (
                            omegas[omega_index].dot(d).dot(d.T))
                    else:
                        gw += p / self.sigma * (omegas[omega_index].dot(d).dot(d.T))
        if lr_relevances > 0:
            if sum(self.regularization_) > 0:
                regmatrices = np.zeros([sum(self.dim_), n_dim])
                for i in range(len(omegas)):
                    regmatrices[sum(self.dim_[:i + 1]) - self.dim_[i]:sum(
                        self.dim_[:i + 1])] = \
                        self.regularization_[i] * np.linalg.pinv(omegas[i])
                g[nb_prototypes:] = 2 / n_data * lr_relevances * \
                                    np.concatenate(gw) - regmatrices
            else:
                g[nb_prototypes:] = 2 / n_data * lr_relevances * \
                                    np.concatenate(gw)
        if lr_prototypes > 0:
            g[:nb_prototypes] = 1 / n_data * \
                                lr_prototypes * g[:nb_prototypes]
        g *= -(1 + 0.0001 * (random_state.rand(*g.shape) - 0.5))
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        variables = variables.reshape(variables.size // n_dim, n_dim)
        prototypes = variables[:nb_prototypes]
        indices = []
        for i in range(len(self.dim_)):
            indices.append(sum(self.dim_[:i + 1]))
        omegas = np.split(variables[nb_prototypes:], indices[:-1])

        out = 0
        for i in range(n_data):
            xi = training_data[i]
            y = label_equals_prototype[i]
            if len(omegas) == nb_prototypes:
                fs = [self._costf(xi, prototypes[j], omega=omegas[j])
                      for j in range(nb_prototypes)]
            else:
                fs = [self._costf(xi, prototypes[j], omega=omegas[np.where(self.classes_ == self.c_w_[j])[0][0]])
                      for j in range(nb_prototypes)]
            fs_max = max(fs)
            s1 = sum([np.math.exp(fs[i] - fs_max) for i in range(len(fs))
                      if self.c_w_[i] == y])
            s2 = sum([np.math.exp(f - fs_max) for f in fs])
            s1 += 0.0000001
            s2 += 0.0000001
            out += np.math.log(s1 / s2)
        return -out

    def _optimize(self, x, y, random_state):
        nb_prototypes, nb_features = self.w_.shape
        nb_classes = len(self.classes_)
        if not isinstance(self.classwise, bool):
            raise ValueError("classwise must be a boolean")
        if self.initialdim is None:
            if self.classwise:
                self.dim_ = nb_features * np.ones(nb_classes, dtype=np.int)
            else:
                self.dim_ = nb_features * np.ones(nb_prototypes, dtype=np.int)
        else:
            self.dim_ = validation.column_or_1d(self.initialdim)
            if self.dim_.size == 1:
                if self.classwise:
                    self.dim_ = self.dim_[0] * np.ones(nb_classes,
                                                       dtype=np.int)
                else:
                    self.dim_ = self.dim_[0] * np.ones(nb_prototypes,
                                                       dtype=np.int)
            elif self.classwise and self.dim_.size != nb_classes:
                raise ValueError("dim length must be number of classes")
            elif self.dim_.size != nb_prototypes:
                raise ValueError("dim length must be number of prototypes")
            if self.dim_.min() <= 0:
                raise ValueError("dim must be a list of positive ints")

        # initialize psis (psis is list of arrays)
        if self.initial_matrices is None:
            self.omegas_ = []
            for d in self.dim_:
                self.omegas_.append(
                    random_state.rand(d, nb_features) * 2.0 - 1.0)
        else:
            if not isinstance(self.initial_matrices, list):
                raise ValueError("initial matrices must be a list")
            self.omegas_ = list(map(lambda v: validation.check_array(v),
                                    self.initial_matrices))
            if self.classwise:
                if len(self.omegas_) != nb_classes:
                    raise ValueError("length of matrices wrong\n"
                                     "found=%d\n"
                                     "expected=%d" % (
                                         len(self.omegas_), nb_classes))
                elif np.sum(map(lambda v: v.shape[1],
                                self.omegas_)) != nb_features * \
                        len(self.omegas_):
                    raise ValueError(
                        "each matrix should have %d columns" % nb_features)
            elif len(self.omegas_) != nb_prototypes:
                raise ValueError("length of matrices wrong\n"
                                 "found=%d\n"
                                 "expected=%d" % (
                                     len(self.omegas_), nb_classes))
            elif np.sum([v.shape[1] for v in self.omegas_]) != \
                    nb_features * len(self.omegas_):
                raise ValueError(
                    "each matrix should have %d columns" % nb_features)

        if isinstance(self.regularization, float):
            if self.regularization < 0:
                raise ValueError('regularization must be a positive float')
            self.regularization_ = np.repeat(self.regularization,
                                             len(self.omegas_))
        else:
            self.regularization_ = validation.column_or_1d(self.regularization)
            if self.classwise:
                if self.regularization_.size != nb_classes:
                    raise ValueError(
                        "length of regularization must be number of classes")
            else:
                if self.regularization_.size != self.w_.shape[0]:
                    raise ValueError(
                        "length of regularization "
                        "must be number of prototypes")

        variables = np.append(self.w_, np.concatenate(self.omegas_), axis=0)
        label_equals_prototype = y
        res = minimize(
            fun=lambda vs: self._optfun(
                vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                vs, x, label_equals_prototype=label_equals_prototype,
                lr_prototypes=0, lr_relevances=1, random_state=random_state),
            method='L-BFGS-B',
            x0=variables, options={'disp': self.display, 'gtol': self.gtol,
                                   'maxiter': self.max_iter})
        n_iter = res.nit
        res = minimize(
            fun=lambda vs: self._optfun(
                vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                vs, x, label_equals_prototype=label_equals_prototype,
                lr_prototypes=0, lr_relevances=1, random_state=random_state),
            method='L-BFGS-B',
            x0=res.x, options={'disp': self.display, 'gtol': self.gtol,
                               'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        res = minimize(
            fun=lambda vs: self._optfun(
                vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                vs, x, label_equals_prototype=label_equals_prototype,
                lr_prototypes=1, lr_relevances=1, random_state=random_state),
            method='L-BFGS-B',
            x0=res.x, options={'disp': self.display, 'gtol': self.gtol,
                               'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        out = res.x.reshape(res.x.size // nb_features, nb_features)
        self.w_ = out[:nb_prototypes]
        indices = []
        for i in range(len(self.dim_)):
            indices.append(sum(self.dim_[:i + 1]))
        self.omegas_ = np.split(out[nb_prototypes:], indices[:-1])  # .conj().T
        self.n_iter_ = n_iter

    def _f(self, x, i):
        d = (x - self.w_[i])[np.newaxis].T
        d = d.T.dot(self.omegas_[i].T).dot(self.omegas_[i]).dot(d)
        return -d / (2 * self.sigma)

    def _costf(self, x, w, **kwargs):
        if 'omega' in kwargs:
            omega = kwargs['omega']
        else:
            omega = self.omegas_[np.where(self.w_ == w)[0][0]]
        d = (x - w)[np.newaxis].T
        d = d.T.dot(omega.T).dot(omega).dot(d)
        return -d / (2 * self.sigma)

    def _compute_distance(self, x, w=None):
        if w is None:
            w = self.w_

        def foo(e):
            fun = np.vectorize(lambda w: self._costf(e, w),
                               signature='(n)->()')
            return fun(w)

        return np.vectorize(foo, signature='(n)->()')(x)

    def project(self, x, prototype_idx, dims, print_variance_covered=False):
        """Projects the data input data X using the relevance matrix of the
        prototype specified by prototype_idx to dimension dim

        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        prototype_idx : int
          index of the prototype
        dims : int
          dimension to project to
        print_variance_covered : boolean
          flag to print the covered variance of the projection

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        nb_prototypes = self.w_.shape[0]
        if len(self.omegas_) != nb_prototypes \
                or self.prototypes_per_class != 1:
            print('project only possible with classwise relevance matrix')
        # y = self.predict(X)
        v, u = np.linalg.eig(
            self.omegas_[prototype_idx].T.dot(self.omegas_[prototype_idx]))
        idx = v.argsort()[::-1]
        if print_variance_covered:
            print('variance coverd by projection:',
                  v[idx][:dims].sum() / v.sum() * 100)
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))
