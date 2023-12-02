from typing import Any, Dict, List, Optional
from langchain.callbacks.manager import Callbacks
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma : Chroma

    def get_relevant_documents(self, query):
        
        #Step 1: Calculate embeddings for the query string
        emb = self.embeddings.embed_query(query)
        #Step 2: Feed embeddings into max_marginal_relevance_search_by_vector

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding = emb,
            lambda_mult = 0.8
        )
    
    # for this implementation we are not concerned with the async version
    async def aget_relevant_documents(self):
        return []
     
