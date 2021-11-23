from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sg_custom_catboost import utils
from sg_custom_catboost.config.core import config


class TargetDiffTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer which creates the target_dg variable.
    """

    def __init__(self, final_var: str, original_var: str):
        self.final_var = final_var
        self.original_var = original_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Y = X.copy()
        Y[config.model_config.target_dg] = Y[self.final_var] - Y[self.original_var]

        return Y


class SelectVariablesTransformers(BaseEstimator, TransformerMixin):
    """
    Transformer which drops unwanted variables.
    """

    def __init__(self, variables_list: List):
        self.variables_list = variables_list

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Y = X.copy()
        Y = Y.drop(self.variables_list, axis=1)

        return Y


class StandardScalerTransformers(BaseEstimator, TransformerMixin):
    """
    Transformer which creates the standard scaled dataframe.
    """

    def __init__(self, variables_list: List):
        self.variables_list = variables_list

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Y = X.copy()
        Y = utils.standard_scaler_dataframe(Y, X)
        Y = utils.fix_standard_scaler_variables(self.variables_list, Y, X)

        return Y
