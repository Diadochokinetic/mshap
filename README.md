# mshap

<!-- badges: start -->
[![codecov](https://codecov.io/gh/Diadochokinetic/mshap/graph/badge.svg?token=I8WXLLYC4I)](https://codecov.io/gh/Diadochokinetic/mshap)
<!-- badges: end -->

This is a Python port of [srmatth/mshap](https://github.com/srmatth/mshap)

The goal of mshap is to allow SHAP values for two-part models to be
easily computed. A two-part model is one where the output from one model
is multiplied by the output from another model. These are often used in
the Actuarial industry, but have other use cases as well.

## Installation

Install mSHAP from pypi with the following code:

``` sh
pip install mshap
```

Or the development version from github with:

``` sh
pip install git+https://github.com/Diadochokinetic/mshap
```


## Basic Use

We will demonstrate a simple use case on simulated data. Suppose that we wish to be able to predict to total amount of money a consumer will spend on a subscription to a software product. We might simulate 4 explanatory variables that looks like the following:


```python
import numpy as np

age = np.random.uniform(18, 60, size=1000)
income = np.random.uniform(50000, 150000, size=1000)
married = np.random.randint(0, 2, size=1000)
sex = np.random.randint(0, 2, size=1000)
```

Now because this is a contrived example, we will knowingly set the
response variables as follows (suppose here that `cost_per_month` is
usage based, so as to be continuous):


```python
cost_per_month = (0.0006 * income - 0.2 * sex + 0.5 * married - 0.001 * age) + 10
num_months = 15 * (0.001 * income * 0.001 * sex * 0.5 * married - 0.05 * age) ** 2
```

Thus, we have our data. We will combine the covariates and target variables into a single
data frame for ease of use in python.


```python
import pandas as pd

data = pd.DataFrame(
    {
        "age": age,
        "income": income,
        "married": married,
        "sex": sex,
        "cost_per_month": cost_per_month,
        "num_months": num_months,
    }
)
```

The end goal of this exercise is to predict the total revenue from the given customer, which mathematically will be `cost_per_month * num_months`. Instead of multiplying these two vectors together initially, we will instead create two models: one to predict `cost_per_month` and the other to predict `num_months`. We can then multiply the output of the two models together to get our predictions.

We now create our two models and predict on the training sets:


```python
from sklearn.ensemble import RandomForestRegressor

X = data[["age", "income", "married", "sex"]]
y1 = data["cost_per_month"]
y2 = data["num_months"]

cpm_mod = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=2)
cpm_mod.fit(X, y1)
# > RandomForestRegressor(max_depth=10, max_features=2)
nm_mod = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=2)
nm_mod.fit(X, y2)
# > RandomForestRegressor(max_depth=10, max_features=2)
cpm_preds = cpm_mod.predict(X)
nm_preds = nm_mod.predict(X)

tot_rev = cpm_preds * nm_preds
```

We will now proceed to use TreeSHAP and subsequently mSHAP to explain the ultimate model predictions.


```python
import shap

cpm_ex = shap.Explainer(cpm_mod)
cpm_shap = cpm_ex.shap_values(X)
cpm_expected_value = cpm_ex.expected_value

nm_ex = shap.Explainer(nm_mod)
nm_shap = nm_ex.shap_values(X)
nm_expected_value = nm_ex.expected_value
```


```python
from mshap import Mshap

final_shap = Mshap(
    cpm_shap, nm_shap, cpm_expected_value, nm_expected_value
).shap_values()
final_shap
```




    {'shap_vals':                0            1          2          3
     0   -2876.216193   325.130506  13.474704 -26.475439
     1    1950.301864   200.312921 -11.558773 -64.926704
     2   -2092.259421  -734.279715   7.840975  15.369813
     3    2735.235840 -1642.421894 -11.395891 -63.590990
     4    1971.574419  -878.331239 -20.712473  36.722350
     ..           ...          ...        ...        ...
     995 -1261.220638  1439.860900   2.017464  48.838624
     996  1291.397944  -553.954467 -27.043572 -50.365440
     997  1320.930428  -492.378408 -20.519565 -50.760569
     998  1156.518243  -415.144837  20.484928  59.726275
     999 -3375.016633   732.381880 -33.174228 -86.247622
     
     [1000 rows x 4 columns],
     'expected_value': 4284.231240147299}



You can put the result into a shap Explanation object to use shap plot capabilities:



```python
final_shap_explanation = shap.Explanation(
    values=final_shap["shap_vals"].values,
    base_values=final_shap["expected_value"],
    data=X,
    feature_names=X.columns,
)
```


```python
shap.summary_plot(final_shap_explanation, X)
```


    
![png](README_files/README_15_0.png)
    


## Citations

-   For more information about SHAP values in general, you can visit the[SHAP github page](https://github.com/slundberg/shap)
-   If you use `{mshap}`, please cite [*mSHAP: SHAP Values for Two-Part Models*](https://arxiv.org/abs/2106.08990)


