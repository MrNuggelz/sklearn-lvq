import numpy as np

from sklearn.utils.testing import assert_greater, assert_raise_message

from sklearn import datasets
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import check_estimator

from .. import LmrslvqModel
from .. import MrslvqModel
from .. import RslvqModel

# also load the iris dataset
iris = datasets.load_iris()
rng = check_random_state(42)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

score = 0.9

def test_rslvq_iris():
    model = RslvqModel()
    check_estimator(model)
    model.fit(iris.data, iris.target)
    assert_greater(model.score(iris.data, iris.target), score)

    assert_raise_message(ValueError, 'display must be a boolean',
                         RslvqModel(display='true').fit, iris.data, iris.target)
    assert_raise_message(ValueError, 'gtol must be a positive float',
                         RslvqModel(gtol=-1.0).fit, iris.data, iris.target)
    assert_raise_message(ValueError, 'the initial prototypes have wrong shape',
                         RslvqModel(initial_prototypes=[[1, 1], [2, 2]]).fit,
                         iris.data, iris.target)
    assert_raise_message(ValueError,
                         'prototype labels and test data classes do not match',
                         RslvqModel(initial_prototypes=[[1, 1, 1, 1, 'a'],
                                                        [2, 2, 2, 2, 5],
                                                        [2, 2, 2, 2, -3]]).fit,
                         iris.data, iris.target)
    assert_raise_message(ValueError, 'max_iter must be an positive integer',
                         RslvqModel(max_iter='5').fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'max_iter must be an positive integer',
                         RslvqModel(max_iter=0).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'max_iter must be an positive integer',
                         RslvqModel(max_iter=-1).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError,
                         'values in prototypes_per_class must be positive',
                         RslvqModel(prototypes_per_class=np.zeros(
                             np.unique(iris.target).size) - 1).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError,
                         'length of prototypes per class'
                         ' does not fit the number of',
                         RslvqModel(prototypes_per_class=[1, 2]).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'X has wrong number of features',
                         model.predict, [[1, 2], [3, 4]])


def test_mrslvq_iris():
    model = MrslvqModel()
    check_estimator(model)
    model.fit(iris.data, iris.target)
    assert_greater(model.score(iris.data, iris.target), score)

    assert_raise_message(ValueError, 'regularization must be a positive float',
                         MrslvqModel(regularization=-1.0).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError,
                         'initial matrix has wrong number of features',
                         MrslvqModel(
                             initial_matrix=[[1, 2], [3, 4], [5, 6]]).fit,
                         iris.data, iris.target)
    assert_raise_message(ValueError, 'dim must be an positive int',
                         MrslvqModel(initialdim=0).fit, iris.data, iris.target)


def test_lmrslvq_iris():
    model = LmrslvqModel()
    check_estimator(model)
    model.fit(iris.data, iris.target)
    assert_greater(model.score(iris.data, iris.target), 0.85) #TODO: make more stable and increase to 0.94

    assert_raise_message(ValueError, 'regularization must be a positive float',
                         LmrslvqModel(regularization=-1.0).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError,
                         'length of regularization'
                         ' must be number of prototypes',
                         LmrslvqModel(regularization=[-1.0]).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError,
                         'length of regularization must be number of classes',
                         LmrslvqModel(regularization=[-1.0],
                                      classwise=True).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'initial matrices must be a list',
                         LmrslvqModel(initial_matrices=np.array(
                             [[1, 2], [3, 4], [5, 6]])).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'length of matrices wrong',
                         LmrslvqModel(
                             initial_matrices=[[[1, 2], [3, 4], [5, 6]]]).fit,
                         iris.data, iris.target)
    assert_raise_message(ValueError, 'each matrix should have',
                         LmrslvqModel(
                             initial_matrices=[[[1]], [[1]], [[1]]]).fit,
                         iris.data, iris.target)
    assert_raise_message(ValueError, 'length of matrices wrong',
                         LmrslvqModel(initial_matrices=[[[1, 2, 3]]],
                                      classwise=True).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'each matrix should have',
                         LmrslvqModel(initial_matrices=[[[1]], [[1]], [[1]]],
                                      classwise=True).fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'classwise must be a boolean',
                         LmrslvqModel(classwise="a").fit, iris.data,
                         iris.target)
    assert_raise_message(ValueError, 'dim must be a list of positive ints',
                         LmrslvqModel(initialdim=[-1]).fit, iris.data, iris.target)
    assert_raise_message(ValueError, 'dim length must be number of prototypes',
                         LmrslvqModel(initialdim=[1, 1]).fit, iris.data, iris.target)
    assert_raise_message(ValueError, 'dim length must be number of classes',
                         LmrslvqModel(initialdim=[1, 1], classwise=True).fit,
                         iris.data, iris.target)

    LmrslvqModel(classwise=True, initialdim=[1], prototypes_per_class=2).fit(
        iris.data, iris.target)

    model = LmrslvqModel(regularization=0.1)
    model.fit(iris.data, iris.target)
