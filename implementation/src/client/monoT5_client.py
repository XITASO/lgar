import sys
import os
sys.path.append(os.getcwd())
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from implementation.src.client.baseline_client import BaselineClient

class MonoT5Client(BaselineClient):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, legacy=False)

    def get_relevance_scores(self, query, documents, batch_size=32):
        """
        Get relevance scores for each document in the list of documents

        :param query: The query string
        :param documents: A list of document strings
        :param batch_size: The size of each batch
        :return: A list of relevance scores for each document
        """
        relevance_scores = []
        for doc in documents:
            input_text = f"Query: {query} Document: {doc} Relevant:"
            inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=2, return_dict_in_generate=True, output_scores=True)

            logits = outputs.scores[0].squeeze()

            true_token_id = self.tokenizer.convert_tokens_to_ids("▁true")
            false_token_id = self.tokenizer.convert_tokens_to_ids("▁false")
            
            true_logit = logits[true_token_id].item()
            false_logit = logits[false_token_id].item()
            
            relevance_score = true_logit - false_logit
            relevance_scores.append(relevance_score)
        
        return relevance_scores


    def get_ranked_ids(self, relevance_scores, doc_ids):
        """
        Get the ranked list of document IDs based on the relevance scores

        :param relevance_scores: A list of relevance scores for each document
        :param doc_ids: A list of document IDs
        :return: A list of document IDs sorted by relevance score
        """
        ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
        result_ids = [doc_ids[i] for i in ranked_indices]
        return result_ids