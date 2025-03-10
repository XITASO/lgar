import sys
import os
sys.path.append(os.getcwd())
import bm25s
import pandas as pd
from implementation.src.client.baseline_client import BaselineClient

class BM25Client(BaselineClient):
    def __init__(self, model_path, ids) -> None:
        super().__init__(model_path)
        self.ids = ids

    def get_relevance_scores(self, query, documents, batch_size=None):
        """
        Compute the relevance scores of the documents with respect to the query.

        :param query: The query string.
        :param documents: A list of document strings.
        :param batch_size: Not used here; only for compatibility
        :return: A list of relevance scores.
        """
        corpus_json = []
        for i in range(len(documents)):
            corpus_json.append({"text": documents[i], "id": self.ids[i]})
        corpus_text = [doc["text"] for doc in corpus_json]
        corpus_tokens = bm25s.tokenize(corpus_text)
        retriever = bm25s.BM25(corpus=corpus_json)
        retriever.index(corpus_tokens)

        results, _ = retriever.retrieve(bm25s.tokenize(query), k=len(documents))
        return results

    def get_ranked_ids(self, relevance_scores, doc_ids):
        """
        Create a ranked list of document IDs based on the relevance scores.

        :param relevance_scores: A list of relevance scores.
        :param doc_ids: A list of document IDs.
        """
        results = pd.DataFrame(relevance_scores[0].tolist())
        return results["id"].astype(int).tolist()