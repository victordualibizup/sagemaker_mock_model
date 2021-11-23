import pytest
from sg_custom_catboost.config.core import config
from sg_custom_catboost.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.pytest_df)
