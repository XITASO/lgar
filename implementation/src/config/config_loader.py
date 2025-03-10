import sys
import os
sys.path.append(os.getcwd())
import json
from typing import Dict, Any
from pydantic import ValidationError
from implementation.src.config.config import Config


class ConfigLoader:
    _config = None

    def __init__(self, config_path: str = "config.json") -> None:
        """
        Initialize the ConfigLoader with a configuration file.

        :param config_path: Path to the configuration file
        """
        if ConfigLoader._config is None:
            ConfigLoader._config = self.load_and_validate_config(config_path)
        self.config = ConfigLoader._config

    def load_and_validate_config(self, config_path: str) -> Config:
        """
        Load and validate the configuration file to provide its information to other classes.

        :param config_path: Path to the configuration file
        :return: Validated Config object
        """
        project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_full_path: str = os.path.join(project_root, config_path)

        print(f"Loading configuration from: {config_full_path}")
        with open(config_full_path, "r") as file:
            config: Dict[str, Any] = json.load(file)

        print("Validating configuration...")
        try:
            validated_config = Config(**config)
            print("Validation successful!")
            return validated_config
        except ValidationError as e:
            print("Validation error:", e)
            raise
