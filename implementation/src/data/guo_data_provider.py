import sys
import os
sys.path.append(os.getcwd())
from implementation.src.config.config import Config
import pandas as pd
import numpy as np

class GuoDataProvider:
    def __init__(self, config: Config) -> None:
        """
        Initialize the GuoDataProvider class with a configuration object.

        :param config: Configuration object.
        """
        self.config = config
        self.folder_path_slrs = self.config.folder_path_slrs

    def create_dataframe(self, slr_name: str) -> pd.DataFrame:
        """
        Create a dataframe for the given slr of the dataset.

        :param slr_name: Name of csv file
        :return: Dataframe of the SLR
        """
        df = pd.read_csv(f"{self.folder_path_slrs}{slr_name}")
        columns_to_keep = ['title', 'abstract', 'tag']
        df = df[columns_to_keep]
        df = df.rename(columns={'tag': 'label'})
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['label'] = df['label'].map({'Included': 1, 'Excluded': 0, 'included': 1, 'excluded': 0})
        df['id'] = np.arange(1, len(df) + 1)
        return df