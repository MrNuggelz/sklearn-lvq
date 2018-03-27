Learning Vector Quantization
============================

Learning Vector quantization (LVQ) [1]_ attempts to construct a highly
sparse model of the data by representing data classes by *prototypes*.
Prototypes are vectors in the data spaced which are placed such that
they achieve a good nearest-neighbor classification accuracy. More
formally, for a dataset :math:`\{(x_1, y_1), ..., (x_m, y_m)\}` LVQ attempts to
place K prototypes :math:`w_1, ..., w_K` with :math:`labels c_1, ..., c_K` in the data
space, such that as many data points as possible are correctly
classified by assigning the label of the closest prototype.
The number of prototypes K is a hyper-parameter to be specified by the
user. Per default, we use 1 prototype per class.

Contents:

    .. toctree::
       :maxdepth: 2
       
       modules/api
       auto_examples/index
       glvq
       rslvq

Dimensionality Reducation
=========================

.. currentmodule:: sklearn_lvq

The relevances learned by a :class:`GrlvqModel`,:class:`GmlvqModel`,:class:`LgmlvqModel`,:class:`MrslvqModel` and :class:`LmrslvqModel` can be applied for
dimensionality reduction by projecting the data on the eigenvectors of
the relevance matrix which correspond to the largest eigenvalues.

.. topic:: References:

    .. [1] `"Learning Vector Quantization"
     <https://doi.org/10.1007/978-3-642-97610-0_6>`_
     Kohonen, Teuvo - Self-Organizing Maps, pp. 175-189, 1995.
