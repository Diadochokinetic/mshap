import re
import pandas as pd
import numpy as np
from mshap import Mshap
import pytest

shap1 = pd.DataFrame(
    {
        "age": np.random.uniform(-5, 5, 1000),
        "income": np.random.uniform(-5, 5, 1000),
        "married": np.random.uniform(-5, 5, 1000),
        "sex": np.random.uniform(-5, 5, 1000),
    }
)

shap2 = [
    pd.DataFrame(
        {
            "age": np.random.uniform(-5, 5, 1000),
            "income": np.random.uniform(-5, 5, 1000),
            "married": np.random.uniform(-5, 5, 1000),
            "sex": np.random.uniform(-5, 5, 1000),
        }
    ),
    pd.DataFrame(
        {
            "age": np.random.uniform(-5, 5, 1000),
            "income": np.random.uniform(-5, 5, 1000),
            "married": np.random.uniform(-5, 5, 1000),
            "sex": np.random.uniform(-5, 5, 1000),
        }
    ),
    pd.DataFrame(
        {
            "age": np.random.uniform(-5, 5, 1000),
            "income": np.random.uniform(-5, 5, 1000),
            "married": np.random.uniform(-5, 5, 1000),
            "sex": np.random.uniform(-5, 5, 1000),
        }
    ),
]

ex1 = np.float64(3)
ex2 = np.array([4, 5, 6], dtype=np.float64)

real_ex1 = np.mean((np.sum(shap1, axis=1) + ex1) * (np.sum(shap2[0], axis=1) + ex2[0]))


def test_basic_case():
    m = Mshap(shap1, shap2[0], ex1, ex2[0])
    res = m.shap_values()
    assert isinstance(res["expected_value"], float)
    assert isinstance(res["shap_vals"], pd.DataFrame)
    assert res["expected_value"] == real_ex1


def test_multi_case():
    m = Mshap(shap1, shap2, ex1, ex2)
    res = m.shap_values()
    m2 = Mshap(shap2, shap1, ex2, ex1)
    res2 = m2.shap_values()
    assert isinstance(res[0], dict)
    assert len(res) == 3
    assert isinstance(res2[0], dict)
    assert len(res2) == 3


def test_different_names():
    m = Mshap(
        shap1,
        shap2,
        ex1,
        ex2,
        shap_1_names=["Age", "Income", "Married", "Sex"],
        shap_2_names=["Age", "Income", "Children", "American"],
    )
    res = m.shap_values()
    assert res[1]["shap_vals"].shape[1] == 6


def test_matrices():
    m = Mshap(shap1.values, shap2[0].values, ex1, ex2[0])
    res = m.shap_values()
    assert isinstance(res["expected_value"], float)
    assert isinstance(res["shap_vals"], pd.DataFrame)
    assert res["expected_value"] == real_ex1


@pytest.mark.parametrize("ex_list_like", [[0, 1], np.array([0, 1]), pd.Series([0, 1])])
def test_warnings(ex_list_like):
    warning_message = "`ex2` has a length greater than 1, only using first element"
    with pytest.warns(UserWarning, match=warning_message):
        m = Mshap(
            shap1,
            shap2[0],
            ex1,
            ex_list_like,
            shap_1_names=["Age", "Income", "Married", "Sex"],
            shap_2_names=["Age", "Income", "Children", "American"],
        )
        _ = m.shap_values()
    warning_message = "`ex1` has a length greater than 1, only using first element"
    with pytest.warns(UserWarning, match=warning_message):
        m = Mshap(
            shap1,
            shap2[0],
            ex_list_like,
            ex1,
            shap_1_names=["Age", "Income", "Married", "Sex"],
            shap_2_names=["Age", "Income", "Children", "American"],
        )
        _ = m.shap_values()


def test_multiple_matrices():
    error_message = re.escape(
        "`shap_values` is not currently set up to handle multiple matrices in each"
        " `shap_*` argument. Did you accidentally wrap a matrix in a `list()`?"
    )
    with pytest.raises(ValueError, match=error_message):
        m = Mshap(shap2, shap2, ex1, ex2)
        _ = m.shap_values()


def test_same_number_rows():
    error_message = re.escape(
        "`shap_1` and `shap_2` (or their elements) must have the same number of rows"
    )
    with pytest.raises(ValueError, match=error_message):
        m = Mshap(shap1.iloc[1:], shap2, ex1, ex2)
        _ = m.shap_values()


def test_only_both_names():
    error_message = re.escape(
        "You cannot specify only one of `shap_1_names` and `shap_2_names`. Please"
        " specify none or both."
    )
    with pytest.raises(ValueError, match=error_message):
        m = Mshap(
            shap1, shap2, ex1, ex2, shap_1_names=["Age", "Income", "Married", "Sex"]
        )
        _ = m.shap_values()


numerical_dtypes = [
    int,
    float,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
]


@pytest.mark.parametrize(
    "shap1, shap2",
    [(shap1.values.astype(i), shap2[0].values.astype(i)) for i in numerical_dtypes],
)
def test_numerical_shap_values(shap1, shap2):
    m = Mshap(shap1, shap2, ex1, ex2[0])
    _ = m.shap_values()


@pytest.mark.parametrize("shap_values", [shap1, shap2[0]])
def test_wrong_dtype_shap_values(shap_values):
    error_message = re.escape(
        "`shap1` and `shap2` must be only composed of numerical values"
    )
    with pytest.raises(ValueError, match=error_message):
        shap_values_wrong_dtype = shap_values.copy()
        shap_values_wrong_dtype["age"] = shap_values_wrong_dtype["age"].astype(str)
        m = Mshap(shap_values_wrong_dtype, shap2, ex1, ex2)
        _ = m.shap_values()


@pytest.mark.parametrize("ex1, ex2", [(i(ex1), i(ex2[0])) for i in numerical_dtypes])
def test_numerical_expected_values(ex1, ex2):
    m = Mshap(shap1, shap2[0], ex1, ex2)
    _ = m.shap_values()


@pytest.mark.parametrize(
    "ex1, ex2",
    [(np.array([0]).astype(i), np.array([0]).astype(i)) for i in numerical_dtypes],
)
def test_numerical_expected_values_list_like(ex1, ex2):
    m = Mshap(shap1, shap2[0], ex1, ex2)
    _ = m.shap_values()


@pytest.mark.parametrize("ex", [ex1, ex2])
def test_wrong_dtype_expected_values(ex):
    error_message = re.escape("`ex_1` and `ex_2` must be numeric")
    with pytest.raises(ValueError, match=error_message):
        ex_wrong_dtype = ex.copy()
        ex_wrong_dtype = ex_wrong_dtype.astype(str)
        m = Mshap(shap1, shap2, ex_wrong_dtype, ex2)
        _ = m.shap_values()


def test_same_dimensions():
    error_message = re.escape(
        "`shap1` and `shap2` must have the same dimensions, or you must supply"
        " `shap_1_names` and `shap_2_names`"
    )
    with pytest.raises(ValueError, match=error_message):
        shap1_wrong_shape = shap1.copy()
        shap1_wrong_shape["newvar"] = range(1, shap1.shape[0] + 1)
        m = Mshap(shap1_wrong_shape, shap2, ex1, ex2)
        _ = m.shap_values()
