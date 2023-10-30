import copy
import pandas as pd
import numpy as np
import warnings


class Mshap:
    """
    Class for computing the multiplied SHAP values for two sets of SHAP values and
    expected values.

    Parameters
    ----------
    shap_1 : pandas.DataFrame or numpy.ndarray
        The SHAP values for the first set of examples.
    shap_2 : pandas.DataFrame or numpy.ndarray
        The SHAP values for the second set of examples.
    ex_1 : float or numpy.ndarray
        The expected value(s) for the first set of examples.
    ex_2 : float or numpy.ndarray
        The expected value(s) for the second set of examples.
    shap_1_names : list, optional
        The names of the features for the first set of SHAP values.
    shap_2_names : list, optional
        The names of the features for the second set of SHAP values.

    Attributes
    ----------
    shap_1 : pandas.DataFrame or numpy.ndarray
        The SHAP values for the first set of examples.
    shap_2 : pandas.DataFrame or numpy.ndarray
        The SHAP values for the second set of examples.
    ex_1 : float or numpy.ndarray
        The expected value(s) for the first set of examples.
    ex_2 : float or numpy.ndarray
        The expected value(s) for the second set of examples.
    shap_1_names : list or None
        The names of the features for the first set of SHAP values.
    shap_2_names : list or None
        The names of the features for the second set of SHAP values.

    Methods
    -------
    shap_values()
        Computes the multiplied SHAP values for two sets of SHAP values and
        expected values.

    Notes
    -----
    This implementation is based on the paper "mSHAP: SHAP Values for Two-Part Models"
    by Matthews and Hartman (2021).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mshap import Mshap
    >>> shap_1 = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e']) # noqa
    >>> shap_2 = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e']) # noqa
    >>> ex_1 = np.random.randn(100)
    >>> ex_2 = np.random.randn(100)
    >>> m = Mshap(shap_1, shap_2, ex_1, ex_2)
    >>> m.shap_values()
    {'shap_vals':           a         b         c         d         e
    0  0.000000  0.000000  0.000000  0.000000  0.000000
    1  0.000000  0.000000  0.000000  0.000000  0.000000
    2  0.000000  0.000000  0.000000  0.000000  0.000000
    3  0.000000  0.000000  0.000000  0.000000  0.000000
    4  0.000000  0.000000  0.000000  0.000000  0.000000
    ..      ...       ...       ...       ...       ...
    95  0.000000  0.000000  0.000000  0.000000  0.000000
    96  0.000000  0.000000  0.000000  0.000000  0.000000
    97  0.000000  0.000000  0.000000  0.000000  0.000000
    98  0.000000  0.000000  0.000000  0.000000  0.000000
    99  0.000000  0.000000  0.000000  0.000000  0.000000

    [100 rows x 5 columns],
     'expected_value': 0.0001179472617199479}
    """

    def __init__(
        self, shap_1, shap_2, ex_1, ex_2, shap_1_names=None, shap_2_names=None
    ):
        self.shap_1 = copy.deepcopy(shap_1)
        self.shap_2 = copy.deepcopy(shap_2)
        self.ex_1 = ex_1
        self.ex_2 = ex_2
        self.shap_1_names = shap_1_names
        self.shap_2_names = shap_2_names

    def shap_values(self):
        """
        Computes the modified SHAP values for the two sets of SHAP values and examples.

        Raises
        ------
        ValueError
            If `shap_1` and `shap_2` is a list of matrices.

        Returns
        -------
        dict
            A dictionary containing the multiplied SHAP values and the expected value of
            the product of the two sets of examples.
        """
        if isinstance(self.shap_1, list) and isinstance(self.shap_2, list):
            raise ValueError(
                "`shap_values` is not currently set up to handle multiple matrices in"
                " each `shap_*` argument. Did you accidentally wrap a matrix in a"
                " `list()`?"
            )
        elif isinstance(self.shap_1, list) or isinstance(self.shap_2, list):
            if isinstance(self.shap_1, list):
                main = self.shap_1
                main_ex = self.ex_1
                secondary = self.shap_2
                sec_ex = self.ex_2
            else:
                main = self.shap_2
                main_ex = self.ex_2
                secondary = self.shap_1
                sec_ex = self.ex_1

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                l = [
                    self._multiply_shap(
                        shap_1=m,
                        shap_2=secondary,
                        ex_1=main_ex[i],
                        ex_2=sec_ex,
                        shap_1_names=self.shap_1_names,
                        shap_2_names=self.shap_2_names,
                    )
                    for i, m in enumerate(main)
                ]
        else:
            l = self._multiply_shap(
                self.shap_1,
                self.shap_2,
                self.ex_1,
                self.ex_2,
                self.shap_1_names,
                self.shap_2_names,
            )

        return l

    def _multiply_shap(
        self, shap_1, shap_2, ex_1, ex_2, shap_1_names=None, shap_2_names=None
    ):
        """
        Computes the multiplied SHAP values for two sets of SHAP values and expected
        values.

        Parameters
        ----------
        shap_1 : pandas.DataFrame or numpy.ndarray
            The SHAP values for the first set of examples.
        shap_2 : pandas.DataFrame or numpy.ndarray
            The SHAP values for the second set of examples.
        ex_1 : float or numpy.ndarray
            The expected value(s) for the first set of examples.
        ex_2 : float or numpy.ndarray
            The expected value(s) for the second set of examples.
        shap_1_names : list, optional
            The names of the features for the first set of SHAP values.
        shap_2_names : list, optional
            The names of the features for the second set of SHAP values.

        Returns
        -------
        dict
            A dictionary containing the multiplied SHAP values and the expected value of
            the product of the two sets of examples.
        """
        # Error Checking
        l = self._validate_shap(shap_1, shap_2, ex_1, ex_2, shap_1_names, shap_2_names)

        # Assign variables with updated, error-free values
        shap_1 = l["shap_1"]
        shap_2 = l["shap_2"]
        ex_1 = l["ex_1"]
        ex_2 = l["ex_2"]

        d = pd.concat(
            [
                (shap_1.iloc[:, i] * ex_2)
                + (shap_2.iloc[:, i] * ex_1)
                + ((shap_1.iloc[:, i] * shap_2.sum(axis=1)) / 2)
                + ((shap_1.sum(axis=1) * shap_2.iloc[:, i]) / 2)
                for i in range(shap_1.shape[1])
            ],
            axis=1,
        )
        d.columns = shap_1.columns

        preds_1 = shap_1.sum(axis=1) + ex_1
        preds_2 = shap_2.sum(axis=1) + ex_2
        preds_3 = preds_1 * preds_2
        expected_value = preds_3.mean()

        tot_s = d.abs().sum(axis=1)
        shap_vals = pd.concat(
            [
                d.iloc[:, i]
                + ((d.iloc[:, i].abs() / tot_s) * (ex_1 * ex_2 - expected_value))
                for i in range(d.shape[1])
            ],
            axis=1,
        )
        shap_vals.columns = shap_1.columns

        # return a dictionary with what we want
        return {"shap_vals": shap_vals, "expected_value": expected_value}

    def _validate_shap(
        self, shap_1, shap_2, ex_1, ex_2, shap_1_names=None, shap_2_names=None
    ):
        """
        Validate the SHAP values passed to the function.

        A simple function that throws errors when certain conditions are not met for
        the SHAP values passed to the mSHAP function.

        Parameters
        ----------
        shap_1 : pandas.DataFrame or numpy.ndarray
            SHAP values to be multiplied.
        shap_2 : pandas.DataFrame or numpy.ndarray
            SHAP values to be multiplied.
        ex_1 : float
            The expected value of the model that corresponds to the shap values in `shap_1`. # noqa
        ex_2 : float
            The expected value of the model that corresponds to the shap values in `shap_2`. # noqa
        shap_1_names : list of str, optional
            The names of the variables in the data that was passed as shap values in `shap_1`. # noqa
        shap_2_names : list of str, optional
            The names of the variables in the data that was passed as shap values in `shap_2`. # noqa

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `shap_1` and `shap_2` (or their elements) do not have the same number of
            rows.
            If `shap_1` and `shap_2` must have the same dimensions, or you must supply
            `shap_1_names` and `shap_2_names`.
            If `shap_1` and `shap_2` must be only composed of numerical values.
            If `ex_1` and `ex_2` are not numeric.
            If you cannot specify only one of `shap_1_names` and `shap_2_names`. Please
            specify none or both.

        Warns
        -----
        UserWarning
            If `ex_1` has a length greater than 1, only using first element.
            If `ex_2` has a length greater than 1, only using first element.
        """
        # Error Checking
        if isinstance(shap_1, np.ndarray):
            shap_1 = pd.DataFrame(shap_1)
        if isinstance(shap_2, np.ndarray):
            shap_2 = pd.DataFrame(shap_2)

        # Check the column types of shap_1 and shap_2
        if not np.issubdtype(shap_1.values.dtype, np.number) or not np.issubdtype(
            shap_2.values.dtype, np.number
        ):
            raise ValueError(
                "`shap1` and `shap2` must be only composed of numerical values"
            )

        # Check the type of the input on the expected values
        if isinstance(ex_1, (list, np.ndarray, pd.Series)):
            if len(ex_1) > 1:
                warnings.warn(
                    "`ex1` has a length greater than 1, only using first element",
                    UserWarning,
                )
            ex_1 = ex_1[0]

        if isinstance(ex_2, (list, np.ndarray, pd.Series)):
            if len(ex_2) > 1:
                warnings.warn(
                    "`ex2` has a length greater than 1, only using first element",
                    UserWarning,
                )
            ex_2 = ex_2[0]

        numerical_dtypes = (
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
        )

        if not isinstance(ex_1, numerical_dtypes) or not isinstance(
            ex_2, numerical_dtypes
        ):
            raise ValueError("`ex_1` and `ex_2` must be numeric")

        if shap_1.shape[0] != shap_2.shape[0]:
            raise ValueError(
                "`shap_1` and `shap_2` (or their elements) must have the same number of"
                " rows"
            )
        if sum([shap_1_names is None, shap_2_names is None]) == 1:
            raise ValueError(
                "You cannot specify only one of `shap_1_names` and `shap_2_names`."
                " Please specify none or both."
            )

        if (shap_1_names is None or shap_2_names is None) and not np.all(
            shap_1.shape == shap_2.shape
        ):
            raise ValueError(
                "`shap1` and `shap2` must have the same dimensions, or you must supply"
                " `shap_1_names` and `shap_2_names`"
            )
        elif not np.all(shap_1.shape == shap_2.shape) or shap_1_names is not None:
            shap_2_missing_names = list(set(shap_1_names) - set(shap_2_names))
            shap_1_missing_names = list(set(shap_2_names) - set(shap_1_names))
            shap_1.columns = shap_1_names
            shap_2.columns = shap_2_names
            if len(shap_2_missing_names) > 0:
                shap_2_missing = pd.DataFrame(
                    np.zeros((shap_1.shape[0], len(shap_2_missing_names))),
                    columns=shap_2_missing_names,
                )
                shap_2 = pd.concat([shap_2, shap_2_missing], axis=1)
            if len(shap_1_missing_names) > 0:
                shap_1_missing = pd.DataFrame(
                    np.zeros((shap_2.shape[0], len(shap_1_missing_names))),
                    columns=shap_1_missing_names,
                )
                shap_1 = pd.concat([shap_1, shap_1_missing], axis=1)
            # Make sure the columns are in the same order
            shap_2 = shap_2[shap_1.columns]

        return {"shap_1": shap_1, "shap_2": shap_2, "ex_1": ex_1, "ex_2": ex_2}
