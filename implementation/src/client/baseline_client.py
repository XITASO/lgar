from abc import ABC, abstractmethod

class BaselineClient(ABC):
    def __init__(self, model_path) -> None:
        """
        Initialize the BaselineClient.

        :param model_path: Path to model (locally stored)
        """
        self.model_path = model_path

    @abstractmethod
    def get_relevance_scores(self, query, documents):
        pass

    @abstractmethod
    def get_ranked_ids(self, relevance_scores, doc_ids):
        pass