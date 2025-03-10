import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time
from transformers import AutoTokenizer, AutoModel
from implementation.src.client.baseline_client import BaselineClient

MAX_LENGTH = 512

class ColBERTClient(BaselineClient):
    def __init__(self, model_path, dataframe: pd.DataFrame = None, query: str = None) -> None:
        """
        Initialize the ColBERTClient.
        """
        super().__init__(model_path)
        self.dataframe = dataframe
        self.query = query
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def colbert_late_interaction(self, query_embedding, document_embeddings):
        """
        Compute the relevance scores between the query and the documents using the ColBERT model.

        :param query_embedding: The query embedding.
        :param document_embeddings: The document embeddings.
        """
        scores = []
        query_embedding = query_embedding.squeeze(0)
        query_embedding_np = query_embedding.cpu().numpy()

        for doc_emb in document_embeddings:
            doc_emb_np = doc_emb.cpu().numpy()
            
            if query_embedding_np.ndim == 1:
                query_embedding_np = query_embedding_np.reshape(1, -1)
            if doc_emb_np.ndim == 1:
                doc_emb_np = doc_emb_np.reshape(1, -1)
            
            sim_matrix = cosine_similarity(query_embedding_np, doc_emb_np)
            
            max_sim = torch.tensor(sim_matrix.max(axis=1)).float()
            
            scores.append(torch.sum(max_sim).item())
        
        return scores
    
    def create_ranked_dataframe(self):
        """
        Create a DataFrame of the ranked documents based on the relevance scores.

        :return: A DataFrame containing the ranked documents and the elapsed time.
        """
        start_time = time.time()

        doc_texts = [str(row["title"]) + " " + str(row["abstract"]) for _, row in self.dataframe.iterrows()]
        doc_ids = self.dataframe["id"].apply(str).to_list()

        query_embedding = self.encode_texts([self.query])
        document_embeddings = self.encode_texts(doc_texts)
        scores = self.colbert_late_interaction(query_embedding, document_embeddings)


        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        result_ids = [doc_ids[i] for i in ranked_indices]
        elapsed_time = time.time() - start_time
        result_df = pd.DataFrame({"id": result_ids})
        result_df["id"] = pd.to_numeric(result_df["id"])
        result_df = pd.merge(result_df, self.dataframe, how="inner", on="id")
        return result_df, elapsed_time
    
    def encode_texts(self, texts):
        """
        Encode a list of texts into embeddings.
        
        :param texts: A list of texts to encode.
        :return: A tensor of embeddings.
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    
    def get_relevance_scores(self, query, documents, batch_size=32):
        """
        Get the relevance scores between the query and the documents.

        :param query: The query (title + rqs or SLR).
        :param documents: The documents (title + abstract for each paper).
        :param batch_size: The batch size for encoding documents.
        :return: The relevance scores.
        """
        query_embedding = self.encode_texts([query])

        scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.encode_texts(batch)

            if batch_embeddings is None or len(batch_embeddings) == 0:
                continue
            
            batch_scores = self.colbert_late_interaction(query_embedding, batch_embeddings)
            scores.extend(batch_scores)

        return scores

    def get_ranked_ids(self, relevance_scores, doc_ids):
        """
        Create a list of document IDs ranked by relevance scores.

        :param relevance_scores: The relevance scores.
        :param doc_ids: The document IDs.
        :return: The ranked document IDs.
        """
        ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
        result_ids = [doc_ids[i] for i in ranked_indices]
        return result_ids