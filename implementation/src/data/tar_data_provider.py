import sys
import os
sys.path.append(os.getcwd())
from implementation.src.config.config import Config
import pandas as pd

class TarDataProvider:
    def __init__(self, config: Config) -> None:
        """
        Initialize the TarDataProvider class with a configuration object.

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
        columns_to_keep = ['PMID', 'Title', 'Abstract', 'Label']
        df = df[columns_to_keep]
        df = df.rename(columns={'Label': 'label', 'Abstract': 'abstract', 'Title': 'title', 'PMID': 'id'})
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        return df
