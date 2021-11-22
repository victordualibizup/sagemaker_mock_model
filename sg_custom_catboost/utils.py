import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler

def fix_standard_scaler_variables(
    variables_list: List, 
    target_dataframe: pd.DataFrame,
    original_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Return proper variables without Standard Scaler 
    transformation.

    Args:
        variables_list (List): List of variables to avoid transformation.
        target_dataframe (pd.DataFrame): The transformed dataframe.
        original_dataframe (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: The fixed_dataframe.
    """
    
    fixed_dataframe = target_dataframe.copy()
    for var in variables_list:
        fixed_dataframe[var] = original_dataframe[var]
        
    return fixed_dataframe


def standard_scaler_dataframe(
    target_dataframe: pd.DataFrame,
    original_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Applies StandardScaler method and return a structured dataframe.
    
    Args:
        target_dataframe (pd.DataFrame): The transformed dataframe.
        original_dataframe (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: [description]
    """

    transformed_dataframe = StandardScaler().fit_transform(target_dataframe)
    transformed_dataframe = pd.DataFrame(
        transformed_dataframe, 
        index=original_dataframe.index,
        columns=original_dataframe.columns
        )
    
    return transformed_dataframe