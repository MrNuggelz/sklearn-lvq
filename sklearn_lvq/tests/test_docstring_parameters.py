# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import inspect
import sys
import warnings
import importlib

from pkgutil import walk_packages
from inspect import getsource, isabstract

from sklearn.base import signature
from sklearn.utils.testing import SkipTest, _get_args
from sklearn.utils.testing import _get_func_name
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.deprecation import _is_deprecated

import sklearn_lvq

PUBLIC_MODULES = [pckg[1] for pckg in walk_packages(prefix='sklearn_lvq.',
                                                    path=sklearn_lvq.__path__)
                  if not ("._" in pckg[1] or ".tests." in pckg[1])]
PUBLIC_MODULES = set(PUBLIC_MODULES)

# Methods where y param should be ignored if y=None by default
_METHODS_IGNORE_NONE_Y = [
    'fit',
    'score',
    'fit_predict',
    'fit_transform',
    'partial_fit',
    'predict'
]


def test_docstring_parameters():
    # Test module docstring formatting

    # Skip test if numpydoc is not found or if python version is < 3.5
    try:
        import numpydoc  # noqa
        assert sys.version_info >= (3, 5)
    except (ImportError, AssertionError):
        raise SkipTest("numpydoc is required to test the docstrings, "
                       "as well as python version >= 3.5")

    from numpydoc import docscrape

    incorrect = []
    for name in PUBLIC_MODULES:
        if name.startswith('_'):
            continue
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        # Exclude imported classes
        classes = [cls for cls in classes if cls[1].__module__ == name]
        for cname, cls in classes:
            this_incorrect = []
            if cname.startswith('_'):
                continue
            if isabstract(cls):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError('Error for __init__ of %s in %s:\n%s'
                                   % (cls, name, w[0]))

            cls_init = getattr(cls, '__init__', None)

            if _is_deprecated(cls_init):
                continue

            elif cls_init is not None:
                this_incorrect += check_docstring_parameters(
                    cls.__init__, cdoc, class_name=cname)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                if _is_deprecated(method):
                    continue
                param_ignore = None
                # Now skip docstring test for y when y is None
                # by default for API reason
                if method_name in _METHODS_IGNORE_NONE_Y:
                    sig = signature(method)
                    if ('y' in sig.parameters and
                            sig.parameters['y'].default is None):
                        param_ignore = ['y']  # ignore y for fit and score
                result = check_docstring_parameters(
                    method, ignore=param_ignore, class_name=cname)
                this_incorrect += result

            incorrect += this_incorrect

        functions = inspect.getmembers(module, inspect.isfunction)
        # Exclude imported functions
        functions = [fn for fn in functions if fn[1].__module__ == name]
        for fname, func in functions:
            # Don't test private methods / functions
            if fname.startswith('_'):
                continue
            if fname == "configuration" and name.endswith("setup"):
                continue
            name_ = _get_func_name(func)
            if not _is_deprecated(func):
                incorrect += check_docstring_parameters(func)
    msg = '\n' + '\n'.join(sorted(list(set(incorrect))))
    if len(incorrect) > 0:
        raise AssertionError("Docstring Error: " + msg)


@ignore_warnings(category=DeprecationWarning)
def test_tabs():
    # Test that there are no tabs in our source files
    for importer, modname, ispkg in walk_packages(sklearn_lvq.__path__,
                                                  prefix='sklearn_lvq.'):
        # because we don't import
        mod = importlib.import_module(modname)
        try:
            source = getsource(mod)
        except IOError:  # user probably should have run "make clean"
            continue
        assert '\t' not in source, ('"%s" has tabs, please remove them ',
                                    'or add it to theignore list'
                                    % modname)


def check_docstring_parameters(func, doc=None, ignore=None, class_name=None):
    """Helper to check docstring

    Parameters
    ----------
    func : callable
        The function object to test.
    doc : str, optional (default: None)
        Docstring if it is passed manually to the test.
    ignore : None | list
        Parameters to ignore.
    class_name : string, optional (default: None)
       If ``func`` is a class method and the class name is known specify
       class_name for the error message.

    Returns
    -------
    incorrect : list
        A list of string describing the incorrect results.
    """
    from numpydoc import docscrape
    incorrect = []
    ignore = [] if ignore is None else ignore

    func_name = _get_func_name(func, class_name=class_name)
    if not func_name.startswith('sklearn_lvq.'):
        return incorrect
    # Don't check docstring for property-functions
    if inspect.isdatadescriptor(func):
        return incorrect
    args = list(filter(lambda x: x not in ignore, _get_args(func)))
    # drop self
    if len(args) > 0 and args[0] == 'self':
        args.remove('self')

    if doc is None:
        with warnings.catch_warnings(record=True) as w:
            try:
                doc = docscrape.FunctionDoc(func)
            except Exception as exp:
                incorrect += [func_name + ' parsing error: ' + str(exp)]
                return incorrect
        if len(w):
            raise RuntimeError('Error for %s:\n%s' % (func_name, w[0]))

    param_names = []
    for name, type_definition, param_doc in doc['Parameters']:
        if (type_definition.strip() == "" or
                type_definition.strip().startswith(':')):

            param_name = name.lstrip()

            # If there was no space between name and the colon
            # "verbose:" -> len(["verbose", ""][0]) -> 7
            # If "verbose:"[7] == ":", then there was no space
            if param_name[len(param_name.split(':')[0].strip())] == ':':
                incorrect += [func_name +
                              ' There was no space between the param name and '
                              'colon ("%s")' % name]
            else:
                incorrect += [func_name + ' Incorrect type definition for '
                                          'param: "%s" (type definition was "%s")'
                              % (name.split(':')[0], type_definition)]
        if '*' not in name:
            param_names.append(name.split(':')[0].strip('` '))

    param_names = list(filter(lambda x: x not in ignore, param_names))

    if len(param_names) != len(args):
        bad = str(sorted(list(set(param_names) ^ set(args))))
        incorrect += [func_name + ' arg mismatch: ' + bad]
    else:
        for n1, n2 in zip(param_names, args):
            if n1 != n2:
                incorrect += [func_name + ' ' + n1 + ' != ' + n2]
    return incorrect
