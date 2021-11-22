from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import sg_custom_catboost

PACKAGE_ROOT = Path(sg_custom_catboost.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "processed"
CONFIG_FILE_PATH = ROOT / "config.yml"
TRAINED_MODEL_DIR = ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Configuration not relevant for model.
    """
    author: str
    squad: str
    package_name: str
    training_data_file: str
    new_data_file: str
    pytest_df: str


class ModelConfig(BaseModel):
    """
    Configuration for model purposes.
    """
    model_name: str
    pipeline_name: str
    model_save_file: str
    target: str
    diff_create_var: List[str]
    target_dg: str
    filter_ph: str
    filter_srm: str
    filter_ebc: str
    filter_ibu: str
    ph_limit: int
    srm_limit: int
    ebc_limit: int
    ibu_limit: int
    imputer_variables: List[str]
    no_standard_scaler_variables: str
    to_drop_train: List[str]
    target_diff_transformer_name: str
    drop_variables_transformer_name: str
    standard_scaler_transformer_name: str
    catboost_params: Dict[str, float]
    random_state: int
    test_size: float


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
