import pytest

from iac_exercise.config.core import config
from iac_exercise.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.pytest_df)
