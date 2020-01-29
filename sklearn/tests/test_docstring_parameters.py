# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import inspect
import warnings
import importlib

from pkgutil import walk_packages
from inspect import signature

import numpy as np

import sklearn
from sklearn.utils import IS_PYPY
from sklearn.utils._testing import SkipTest
from sklearn.utils._testing import check_docstring_parameters
from sklearn.utils._testing import _get_func_name
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import all_estimators
from sklearn.utils.deprecation import _is_deprecated
from sklearn.externals._pep562 import Pep562
from sklearn.datasets import make_classification, make_regression
from sklearn.base import is_classifier, is_regressor

import pytest


# walk_packages() ignores DeprecationWarnings, now we need to ignore
# FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    PUBLIC_MODULES = set([
        pckg[1] for pckg in walk_packages(prefix='sklearn.',
                                          path=sklearn.__path__)
        if not ("._" in pckg[1] or ".tests." in pckg[1])
    ])

# functions to ignore args / docstring of
_DOCSTRING_IGNORES = [
    'sklearn.utils.deprecation.load_mlcomp',
    'sklearn.pipeline.make_pipeline',
    'sklearn.pipeline.make_union',
    'sklearn.utils.extmath.safe_sparse_dot',
    'sklearn.utils._joblib'
]

# Methods where y param should be ignored if y=None by default
_METHODS_IGNORE_NONE_Y = [
    'fit',
    'score',
    'fit_predict',
    'fit_transform',
    'partial_fit',
    'predict'
]


# numpydoc 0.8.0's docscrape tool raises because of collections.abc under
# Python 3.7
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.skipif(IS_PYPY, reason='test segfaults on PyPy')
def test_docstring_parameters():
    # Test module docstring formatting

    # Skip test if numpydoc is not found
    try:
        import numpydoc  # noqa
    except ImportError:
        raise SkipTest("numpydoc is required to test the docstrings")

    from numpydoc import docscrape

    incorrect = []
    for name in PUBLIC_MODULES:
        if name == 'sklearn.utils.fixes':
            # We cannot always control these docstrings
            continue
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        # Exclude imported classes
        classes = [cls for cls in classes if cls[1].__module__ == name]
        for cname, cls in classes:
            this_incorrect = []
            if cname in _DOCSTRING_IGNORES or cname.startswith('_'):
                continue
            if inspect.isabstract(cls):
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
                    cls.__init__, cdoc)

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
                    method, ignore=param_ignore)
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
            if (not any(d in name_ for d in _DOCSTRING_IGNORES) and
                    not _is_deprecated(func)):
                incorrect += check_docstring_parameters(func)

    msg = '\n'.join(incorrect)
    if len(incorrect) > 0:
        raise AssertionError("Docstring Error:\n" + msg)


@ignore_warnings(category=FutureWarning)
def test_tabs():
    # Test that there are no tabs in our source files
    for importer, modname, ispkg in walk_packages(sklearn.__path__,
                                                  prefix='sklearn.'):

        if IS_PYPY and ('_svmlight_format_io' in modname or
                        'feature_extraction._hashing_fast' in modname):
            continue

        # because we don't import
        mod = importlib.import_module(modname)

        # TODO: Remove when minimum python version is 3.7
        # unwrap to get module because Pep562 backport wraps the original
        # module
        if isinstance(mod, Pep562):
            mod = mod._module

        try:
            source = inspect.getsource(mod)
        except IOError:  # user probably should have run "make clean"
            continue
        assert '\t' not in source, ('"%s" has tabs, please remove them ',
                                    'or add it to theignore list'
                                    % modname)


@pytest.mark.parametrize('name, Estimator',
                         all_estimators())
def test_fit_docstring_attributes(name, Estimator):
    pytest.importorskip('numpydoc')
    from numpydoc import docscrape

    doc = docscrape.ClassDoc(Estimator)
    attributes = doc['Attributes']

    IGNORED = ['ClassifierChain', 'ColumnTransformer', 'CountVectorizer',
               'DictVectorizer', 'FeatureUnion', 'GaussianRandomProjection',
               'GridSearchCV', 'MultiOutputClassifier', 'MultiOutputRegressor',
               'NoSampleWeightWrapper', 'OneVsOneClassifier',
               'OneVsRestClassifier', 'OutputCodeClassifier', 'Pipeline',
               'RFE', 'RFECV', 'RandomizedSearchCV', 'RegressorChain',
               'SelectFromModel', 'SparseCoder', 'SparseRandomProjection',
               'SpectralBiclustering', 'StackingClassifier',
               'StackingRegressor', 'TfidfVectorizer', 'VotingClassifier',
               'VotingRegressor']
    if Estimator.__name__ in IGNORED or Estimator.__name__.startswith('_'):
        pytest.xfail(
            reason="Classifier cannot be fit easily to test fit attributes")

    est = Estimator()

    if Estimator.__name__ in ['SelectKBest']:
        est.k = 2

    X_classif, y_classif = \
        make_classification(n_samples=20, n_features=3,
                            n_redundant=0, n_classes=2)

    X_reg, y_reg = \
        make_regression(n_samples=10, n_features=2)

    # Make sure features are positive as some models need it
    X_classif -= X_classif.min()
    X_reg -= X_reg.min()

    if is_classifier(est):
        X, y = X_classif, y_classif
    elif is_regressor(est):
        X, y = X_reg, y_reg
    else:
        X, y = X_classif, y_classif

    tags = est._get_tags()

    if '1darray' in tags['X_types']:
        X = X[:, 0]

    if tags['multioutput_only']:
        y = np.c_[y, y]

    if getattr(est, '_pairwise', None):
        X = X.dot(X.T)

    if '1dlabels' in tags['X_types']:
        est.fit(y)
    elif '2dlabels' in tags['X_types']:
        est.fit(np.c_[y, y])
    else:
        est.fit(X, y)

    for attr in attributes:
        desc = ' '.join(attr.desc).lower()
        if 'only ' not in desc:
            assert hasattr(est, attr.name)

    IGNORED = ['HistGradientBoostingClassifier', 'HistGradientBoostingRegressor',
               'MiniBatchKMeans']
    if Estimator.__name__ in IGNORED:
        pytest.xfail(
            reason="Classifier has too many undocumented attributes.")

    fit_attr = [k for k in est.__dict__.keys() if k.endswith('_')]
    fit_attr_names = [attr.name for attr in attributes]
    for attr in fit_attr:
        if attr in ['X_offset_', 'X_scale_', 'fit_', 'partial_fit_', 'x_mean_',
                    'y_mean_', 'x_std_', 'y_std_', 'dual_gap_',
                    'base_estimator_', 'n_classes_', 'n_estimators_',
                    'classes_', 'n_features_', 'loss_', 'do_early_stopping_',
                    'n_samples_fit_', 'effective_metric_params_',
                    'effective_metric_', 'tree_', 'active_', 'alphas_',
                    'random_state_', 'exp_dirichlet_component_',
                    'dissimilarity_matrix_', 'n_iter_', 't_',
                    'loss_curve_', 'best_loss_', 'eps_',
                    'class_weight_', 'fit_status_', 'shape_fit_', 'location_',
                    'n_nonzero_coefs_', 'loss_function_', 'random_weights_',
                    'random_offset_', 'outlier_label_', 'n_outputs_',
                    'one_hot_encoder_']:
            continue
        if attr.startswith('_'):
            continue
        assert attr in fit_attr_names
