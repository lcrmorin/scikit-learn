"""Integration coverage for pandas string extension arrays."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_array, column_or_1d


@pytest.fixture(params=["python", "pyarrow"])
def string_storage(request):
    if request.param == "pyarrow":
        pytest.importorskip("pyarrow")
    return request.param


@pytest.fixture(params=["nan", "NA"])
def string_dtype(request, string_storage):
    pd = pytest.importorskip("pandas")
    if request.param == "NA":
        return pd.StringDtype(storage=string_storage)
    if parse_version(pd.__version__) < parse_version("3.0"):
        pytest.skip("The explicit NaN string dtype requires pandas 3")
    na_value = np.nan if request.param == "nan" else pd.NA
    return pd.StringDtype(storage=string_storage, na_value=na_value)


@pytest.mark.parametrize("container", ["series", "dataframe"])
def test_pandas_string_validation(string_dtype, container):
    pd = pytest.importorskip("pandas")
    X = pd.Series(["1", "2", "3"], dtype=string_dtype)
    if container == "dataframe":
        X = X.to_frame()
    ensure_2d = container == "dataframe"
    result = check_array(X, dtype=None, ensure_2d=ensure_2d)
    assert result.dtype == np.dtype(object)
    assert_array_equal(result, X.to_numpy())
    assert_array_equal(
        check_array(X, dtype="numeric", ensure_2d=ensure_2d),
        check_array(X.astype(object), dtype="numeric", ensure_2d=ensure_2d),
    )
    with pytest.raises(ValueError):
        check_array(X.replace("1", "blue"), dtype="numeric", ensure_2d=ensure_2d)
    assert_array_equal(
        check_array(X, dtype=float, ensure_2d=ensure_2d),
        X.to_numpy(dtype=float),
    )
    assert_array_equal(column_or_1d(X), ["1", "2", "3"])


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_pandas_string_encoder_roundtrip(string_dtype, Encoder):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"colour": ["blue", "red", "blue"]}, dtype=string_dtype)
    encoder = Encoder()
    encoded = encoder.fit_transform(X)
    reference = Encoder().fit_transform(X.astype(object))
    if hasattr(encoded, "toarray"):
        encoded, reference = encoded.toarray(), reference.toarray()
    assert_array_equal(encoded, reference)
    assert_array_equal(encoder.inverse_transform(encoded), X.to_numpy())


@pytest.mark.parametrize("Encoder", [LabelEncoder, LabelBinarizer])
def test_pandas_string_target_roundtrip(string_dtype, Encoder):
    pd = pytest.importorskip("pandas")
    y = pd.Series(["red", "blue", "red", "green"], dtype=string_dtype)
    assert type_of_target(y) == "multiclass"
    assert_array_equal(unique_labels(y), ["blue", "green", "red"])
    encoder = Encoder()
    encoded = encoder.fit_transform(y)
    assert_array_equal(encoded, Encoder().fit_transform(y.astype(object)))
    assert_array_equal(encoder.inverse_transform(encoded), y.to_numpy())


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_pandas_string_encoder_missing(string_dtype, Encoder):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"colour": ["blue", None, "red"]}, dtype=string_dtype)
    encoder = Encoder()
    encoded = encoder.fit_transform(X)
    restored = encoder.inverse_transform(encoded)
    assert_array_equal(restored[[0, 2], 0], ["blue", "red"])
    assert pd.isna(restored[1, 0])


def test_pandas_string_pipeline_cross_validation(string_dtype):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "colour": pd.Series(["blue", "red"] * 12, dtype=string_dtype),
            "number": np.tile([0.0, 1.0], 12),
        }
    )
    y = pd.Series(["no", "yes"] * 12, dtype=string_dtype)
    selector = make_column_selector(dtype_include=["object", "string"])
    assert selector(X) == ["colour"]
    pipeline = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), selector),
            remainder="passthrough",
        ),
        LogisticRegression(),
    )
    scores = cross_val_score(pipeline, X, y, cv=3, error_score="raise")
    reference_scores = cross_val_score(
        pipeline,
        X.astype({"colour": object}),
        y.astype(object),
        cv=3,
        error_score="raise",
    )
    assert_allclose(scores, reference_scores)
    assert_array_equal(pipeline.fit(X, y).predict(X), y.to_numpy())


def test_pandas_string_imputation(string_dtype):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"colour": ["blue", None, "blue", "red"]}, dtype=string_dtype)
    imputer = SimpleImputer(
        missing_values=string_dtype.na_value, strategy="most_frequent"
    )
    result = imputer.fit_transform(X)
    assert_array_equal(result[:, 0], ["blue", "blue", "blue", "red"])


def test_pandas_nan_string_ordinal_missing_and_unknown(string_storage):
    pd = pytest.importorskip("pandas")
    if parse_version(pd.__version__) < parse_version("3.0"):
        pytest.skip("The explicit NaN string dtype requires pandas 3")
    dtype = pd.StringDtype(storage=string_storage, na_value=np.nan)
    X = pd.DataFrame({"colour": ["blue", None, "red"]}, dtype=dtype)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
    ).fit(X)
    probe = pd.DataFrame({"colour": ["red", None, "green"]}, dtype=dtype)
    assert_array_equal(encoder.transform(probe)[:, 0], [1, -2, -1])


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
@pytest.mark.parametrize("values", [[None, None], ["nan", "<NA>", "", None]])
def test_pandas_string_missing_matches_nan(string_dtype, Encoder, values):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"x": values}, dtype=string_dtype)
    original = X.copy(deep=True)
    reference = X.to_numpy(dtype=object, na_value=np.nan)
    encoder = Encoder()
    expected_encoder = Encoder()
    actual = encoder.fit_transform(X)
    expected = expected_encoder.fit_transform(reference)
    if hasattr(actual, "toarray"):
        actual, expected = actual.toarray(), expected.toarray()
    assert_allclose(actual, expected)
    restored = encoder.inverse_transform(actual)
    expected_restored = expected_encoder.inverse_transform(expected)
    pd.testing.assert_frame_equal(
        pd.DataFrame(restored), pd.DataFrame(expected_restored)
    )
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_pandas_string_prediction_only_missing(string_dtype, Encoder):
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame({"x": ["blue", "red"]}, dtype=string_dtype)
    probe = pd.DataFrame({"x": [None, "green"]}, dtype=string_dtype)
    with pytest.raises(ValueError, match="unknown categories"):
        Encoder().fit(train).transform(probe)
    kwargs = (
        {"handle_unknown": "ignore"}
        if Encoder is OneHotEncoder
        else {"handle_unknown": "use_encoded_value", "unknown_value": -1}
    )
    encoder = Encoder(**kwargs).fit(train)
    result = encoder.transform(probe)
    if Encoder is OneHotEncoder:
        assert_array_equal(result.toarray(), np.zeros((2, 2)))
    else:
        assert_array_equal(result, [[-1], [-1]])


@pytest.mark.parametrize(
    "dtype,labels",
    [
        ("Int64", [0, 1]),
        ("UInt64", [0, 1]),
        ("boolean", [False, True]),
        ("Int64", [2**53, 2**53 + 1]),
        ("UInt64", [2**53, 2**53 + 1]),
        ("int64[pyarrow]", [2**53, 2**53 + 1]),
        ("uint64[pyarrow]", [2**53, 2**53 + 1]),
        ("bool[pyarrow]", [False, True]),
    ],
)
def test_nullable_target_pipeline(dtype, labels):
    pd = pytest.importorskip("pandas")
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    X = np.tile([[0.0, 1.0], [1.0, 0.0]], (20, 1))
    y = pd.Series(labels * 20, dtype=dtype)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    pipeline.fit(X, y)
    # Python integer comparisons do not mask precision loss at 2**53.
    assert [int(v) for v in pipeline[-1].classes_] == [int(v) for v in labels]
    assert [int(v) for v in pipeline.predict(X)] == [int(v) for v in y]
    assert pipeline.score(X, y) == 1.0
    assert_array_equal(
        cross_val_score(pipeline, X, y, cv=2, error_score="raise"), [1.0, 1.0]
    )
    y_missing = y.copy()
    y_missing.iloc[0] = pd.NA
    with pytest.raises(ValueError, match="NaN|NA|missing"):
        pipeline.fit(X, y_missing)


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Object columns containing pd.NA still mix NAType and strings",
)
def test_object_string_pd_na_encoding(Encoder):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"x": ["blue", pd.NA, "red"]}, dtype=object)
    encoder = Encoder()
    result = encoder.inverse_transform(encoder.fit_transform(X))
    assert_array_equal(result[[0, 2], 0], ["blue", "red"])
    assert pd.isna(result[1, 0])
