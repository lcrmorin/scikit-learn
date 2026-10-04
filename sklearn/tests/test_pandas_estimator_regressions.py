"""Regression coverage from the broad pandas/reference estimator audit."""

from decimal import Decimal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sklearn.base import clone
from sklearn.impute import MissingIndicator
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM


def _data():
    X = np.random.default_rng(42).integers(0, 4, (36, 4))
    return X, X[:, 0] % 2


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "estimator,method",
    [
        (LogisticRegression(), "predict_proba"),
        (GaussianNB(), "predict_proba"),
        (AdditiveChi2Sampler(), "transform"),
    ],
)
def test_numeric_categorical_estimator_inference(ordered, estimator, method):
    pd = pytest.importorskip("pandas")
    X, y = _data()
    frame = pd.DataFrame(
        {i: pd.Categorical(X[:, i], ordered=ordered) for i in range(4)}
    )
    reference = clone(estimator).fit(X, y)
    expected = getattr(reference, method)(X)
    candidate = clone(estimator).fit(frame, y)
    assert_allclose(getattr(candidate, method)(frame), expected)


@pytest.mark.parametrize("estimator", [LogisticRegression(), GaussianNB()])
def test_arrow_decimal_estimator_inference(estimator):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    X, y = _data()
    frame = pd.DataFrame(
        {
            i: pd.Series(
                [Decimal(int(v)) / 10 for v in X[:, i]],
                dtype=pd.ArrowDtype(pa.decimal128(12, 2)),
            )
            for i in range(4)
        }
    )
    reference = clone(estimator).fit(frame.to_numpy(dtype=object), y)
    expected = reference.predict_proba(frame.to_numpy(dtype=object))
    candidate = clone(estimator).fit(frame, y)
    assert_allclose(candidate.predict_proba(frame), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "estimator",
    [
        SVC(),
        SVR(),
        NuSVC(),
        NuSVR(),
        OneClassSVM(),
        KNeighborsClassifier(metric="precomputed"),
        KNeighborsRegressor(metric="precomputed"),
    ],
)
def test_sparse_pandas_estimator_matches_scipy(estimator, dtype):
    pd = pytest.importorskip("pandas")
    X, y = _data()
    if isinstance(estimator, (KNeighborsClassifier, KNeighborsRegressor)):
        X = pairwise_distances(X)
    X = X.astype(dtype)
    reference_input = sparse.csr_matrix(X)
    frame = pd.DataFrame(X).astype(pd.SparseDtype(dtype, 0))
    reference = clone(estimator).fit(reference_input, y)
    expected = reference.predict(reference_input)
    candidate = clone(estimator).fit(frame, y)
    assert_allclose(candidate.predict(frame), expected)


@pytest.mark.parametrize(
    "estimator,method",
    [
        (AdditiveChi2Sampler(), "transform"),
        (GaussianMixture(random_state=0), "score_samples"),
        (BayesianGaussianMixture(random_state=0), "score_samples"),
    ],
)
def test_numpy_boolean_estimator_matches_float(estimator, method):
    # This is an estimator/NumPy Boolean issue, not a nullable-pandas regression.
    X, y = _data()
    X = X.astype(bool)
    reference = clone(estimator).fit(X.astype(float), y)
    candidate = clone(estimator).fit(X, y)
    assert_allclose(
        getattr(candidate, method)(X), getattr(reference, method)(X.astype(float))
    )


@pytest.mark.parametrize("dtype", [object, "string[python]", "string[pyarrow]"])
@pytest.mark.parametrize("missing_values", ["pd.NA", "np.nan"])
def test_missing_indicator_pd_na(dtype, missing_values):
    pd = pytest.importorskip("pandas")
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    X = pd.DataFrame({"x": ["blue", pd.NA, "red", "nan"]}, dtype=dtype)
    original = X.copy(deep=True)
    sentinel = pd.NA if missing_values == "pd.NA" else np.nan
    indicator = MissingIndicator(missing_values=sentinel)
    actual = indicator.fit_transform(X)
    assert_allclose(indicator.transform(X), actual)
    pd.testing.assert_frame_equal(X, original)
    assert_allclose(actual, [[False], [True], [False], [False]])
