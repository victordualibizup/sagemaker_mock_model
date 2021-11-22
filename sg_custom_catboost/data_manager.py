import typing as t
from pathlib import Path
import pickle
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sg_custom_catboost.config.core import RAW_DATASET_DIR, PROCESSED_DATASET_DIR, PIPELINE_DIR, TRAINED_MODEL_DIR, \
    config

# TODO: FIX VERSION VARIABLE
_version = "0.0.1"


def load_dataset(file_name: str, raw_data: bool = True) -> pd.DataFrame:
    data_dir = PROCESSED_DATASET_DIR
    if raw_data:
        data_dir = RAW_DATASET_DIR
        dataframe = pd.read_csv(Path(f"{data_dir}/{file_name}"))
    else:
        dataframe = pd.read_csv(Path(f"{data_dir}/{file_name}"))
    return dataframe


def load_pipeline(file_name: str):
    pipeline = pickle.load(open(Path(f"{PIPELINE_DIR}/{file_name}"), "rb"))
    return pipeline


def load_model(file_name: str) -> Pipeline:
    """

    Parameters
    ----------
    file_name

    Returns
    -------

    """

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def save_model(file_name: str) -> None:
    """

    Parameters
    ----------
    file_name

    Returns
    -------

    """

    # Prepare versioned save file name
    save_file_name = f"{config.model_config.model_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    joblib.dump(file_name, save_path)


def remove_old_pipelines(files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
