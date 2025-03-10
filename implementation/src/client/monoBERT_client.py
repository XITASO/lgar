import sys
import os
import torch
sys.path.append(os.getcwd())
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from implementation.src.client.baseline_client import BaselineClient

class MonoBERTClient(BaselineClient):
    def __init__(self, model_path) -> None:
        """
        Initialize the MonoBERTClient with the model path

        :param model_path: Path to the model directory
        """
        super().__init__(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def get_relevance_scores(self, query, documents, batch_size=32):
        """
        Get relevance scores for the given query and documents

        :param query: Query string
        :param documents: List of document strings
        :param batch_size: The batch size for processing documents.
        :return: List of relevance scores
        """
        relevance_scores = []

        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i + batch_size]
            
            inputs = self.tokenizer([query] * len(batch_documents), batch_documents, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            batch_scores = logits.tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            relevance_scores.extend(batch_scores)

        return relevance_scores

    def get_ranked_ids(self, relevance_scores, doc_ids):
        """
        Get ranked document IDs based on the relevance scores

        :param relevance_scores: List of relevance scores
        :param doc_ids: List of document IDs
        :return: List of ranked document IDs
        """
        ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=False)
        result_ids = [doc_ids[i] for i in ranked_indices]
        return result_ids