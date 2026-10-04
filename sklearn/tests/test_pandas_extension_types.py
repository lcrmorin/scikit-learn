"""Integration checks for pandas extension types at sklearn boundaries."""

from decimal import Decimal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_array


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_categorical_encoder_values(ordered, missing, Encoder):
    pd = pytest.importorskip("pandas")
    dtype = pd.CategoricalDtype(["red", "blue", "unused"], ordered=ordered)
    X = pd.DataFrame(
        {"x": pd.Series(["blue", None if missing else "red", "blue"], dtype=dtype)}
    )
    original = X.copy(deep=True)
    encoder = Encoder()
    encoded = encoder.fit_transform(X)
    restored = encoder.inverse_transform(encoded)
    reference = X.astype(object).to_numpy()
    assert_array_equal(pd.isna(restored), pd.isna(reference))
    observed = ~pd.isna(reference)
    assert_array_equal(restored[observed], reference[observed])
    assert "unused" not in encoder.categories_[0]
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
@pytest.mark.parametrize("missing", [False, True])
def test_categorical_large_integer_values(Encoder, missing, request):
    pd = pytest.importorskip("pandas")
    if missing:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="Missing categorical data triggers lossy float64 conversion",
            )
        )
    values = [2**53, 2**53 + 1, None if missing else 2**53]
    X = pd.DataFrame({"x": pd.Categorical(values)})
    encoder = Encoder()
    restored = encoder.inverse_transform(encoder.fit_transform(X))
    observed = [v for v in values if v is not None]
    assert [int(v) for v in restored.ravel() if not pd.isna(v)] == observed
    assert_array_equal(pd.isna(restored.ravel()), pd.isna(values))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("missing", [False, True])
def test_sparse_dataframe_stays_sparse(dtype, missing):
    pd = pytest.importorskip("pandas")
    values = np.array([[0, 1], [2, 0], [0, np.nan if missing else 3]], dtype=dtype)
    X = pd.DataFrame(values).astype(pd.SparseDtype(dtype, fill_value=0))
    result = check_array(
        X, accept_sparse="csr", dtype=None, ensure_all_finite="allow-nan"
    )
    assert sparse.issparse(result)
    assert result.dtype == dtype
    assert result.nnz == 3
    assert_allclose(result.toarray(), values)
    if missing:
        with pytest.raises(ValueError, match="NaN"):
            check_array(X, accept_sparse=True)


def test_sparse_pipeline_preserves_sparse_output():
    pd = pytest.importorskip("pandas")
    values = np.tile([[0.0, 1.0], [1.0, 0.0]], (10, 1)).astype(np.float32)
    X = pd.DataFrame(values).astype(pd.SparseDtype("float32", 0))
    y = np.tile([0, 1], 10)
    pipeline = make_pipeline(StandardScaler(with_mean=False), LogisticRegression())
    pipeline.fit(X, y)
    transformed = pipeline[:-1].transform(X)
    assert sparse.issparse(transformed)
    assert_array_equal(pipeline.predict(X), y)
    assert transformed.dtype == np.float32


@pytest.mark.parametrize(
    "kind", ["datetime", "timezone", "timedelta", "period", "interval"]
)
def test_temporal_and_interval_indexing_preserves_dtype(kind):
    pd = pytest.importorskip("pandas")
    constructors = {
        "datetime": lambda: pd.Series(
            pd.to_datetime(["2020-01-01", None, "2020-01-03"])
        ),
        "timezone": lambda: pd.Series(
            pd.to_datetime(["2020-01-01", None, "2020-01-03"], utc=True)
        ),
        "timedelta": lambda: pd.Series(pd.to_timedelta([1, None, 3], unit="D")),
        "period": lambda: pd.Series(
            pd.PeriodIndex(["2020-01", None, "2020-03"], freq="M")
        ),
        "interval": lambda: pd.Series(
            pd.arrays.IntervalArray.from_tuples([(0, 1), None, (2, 3)])
        ),
    }
    X = constructors[kind]().to_frame("x")
    selected = _safe_indexing(X, [2, 1])
    pd.testing.assert_frame_equal(selected, X.iloc[[2, 1]])


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_datetime_explicit_feature_extraction(unit):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=4).as_unit(unit)})
    # The caller chooses the meaning and unit of the numeric feature.
    features = X.assign(time=X["time"].dt.day.astype(np.float32))
    result = StandardScaler().fit_transform(features)
    assert result.dtype == np.float32
    assert_allclose(
        result.ravel(), (np.arange(1, 5) - 2.5) / np.std(np.arange(1, 5)), rtol=1e-6
    )


@pytest.mark.parametrize("kind", ["period", "interval"])
def test_non_numeric_extensions_reject_float_conversion(kind):
    pd = pytest.importorskip("pandas")
    values = (
        pd.period_range("2020-01", periods=3, freq="M")
        if kind == "period"
        else pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)])
    )
    with pytest.raises((ValueError, TypeError)):
        check_array(pd.DataFrame({"x": values}), dtype=np.float64)


@pytest.mark.parametrize("missing", [False, True])
def test_arrow_decimal_explicit_float_conversion(missing):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    X = pd.DataFrame(
        {
            "x": pd.Series(
                [
                    Decimal("1.25"),
                    None if missing else Decimal("2.50"),
                    Decimal("3.75"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            )
        }
    )
    result = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
    assert result.dtype == np.float64
    assert_allclose(result.ravel(), [1.25, np.nan if missing else 2.5, 3.75])


@pytest.mark.parametrize("kind", ["list", "struct"])
def test_arrow_nested_numeric_rejection(kind):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    arrow_dtype, values = (
        (pa.list_(pa.int64()), [[1, 2], [3, 4]])
        if kind == "list"
        else (pa.struct([("a", pa.int64())]), [{"a": 1}, {"a": 2}])
    )
    X = pd.DataFrame({"x": pd.Series(values, dtype=pd.ArrowDtype(arrow_dtype))})
    with pytest.raises((TypeError, ValueError)):
        check_array(X, dtype=np.float64)


@pytest.mark.parametrize("fill", [1.0, np.nan])
def test_sparse_nonzero_fill_preserved_or_rejected(fill):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {"x": pd.Series([fill, 2.0, fill], dtype=pd.SparseDtype("float32", fill))}
    )
    try:
        result = check_array(X, accept_sparse=True, ensure_all_finite="allow-nan")
    except ValueError as exc:
        assert "fill value must be 0" in str(exc)
    else:
        result = result.toarray() if sparse.issparse(result) else result
        assert_allclose(result.ravel(), [fill, 2.0, fill])


@pytest.mark.parametrize("kind", ["timestamp", "timestamp_tz", "duration"])
def test_arrow_temporal_selection(kind):
    from datetime import timedelta

    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    if kind == "duration":
        dtype, values = pa.duration("us"), [timedelta(days=1), None, timedelta(days=3)]
    else:
        tz = "UTC" if kind == "timestamp_tz" else None
        dtype = pa.timestamp("us", tz=tz)
        values = [
            pd.Timestamp("2020-01-01", tz=tz),
            None,
            pd.Timestamp("2020-01-03", tz=tz),
        ]
    X = pd.DataFrame({"x": pd.Series(values, dtype=pd.ArrowDtype(dtype))})
    pd.testing.assert_frame_equal(_safe_indexing(X, [2, 1]), X.iloc[[2, 1]])
