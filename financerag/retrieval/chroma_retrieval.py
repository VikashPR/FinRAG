import logging
from typing import Dict, List, Optional, Literal

import chromadb
from chromadb.config import Settings

from financerag.common.protocols import Encoder, Retrieval

logger = logging.getLogger(__name__)


class ChromaDBRetrieval(Retrieval):
    """
    ChromaDB-based dense retrieval that stores embeddings in ChromaDB and performs similarity search.
    
    This class uses ChromaDB as a vector database to store corpus embeddings and efficiently retrieve
    the top-k most relevant documents for each query using cosine similarity.
    """

    def __init__(
        self,
        model: Encoder,
        collection_name: str = "finance_corpus",
        persist_directory: str = "./chroma_db",
        batch_size: int = 64,
    ):
        """
        Initializes the ChromaDBRetrieval class.

        Args:
            model (`Encoder`):
                An encoder model implementing the `Encoder` protocol for encoding queries and corpus documents.
            collection_name (`str`, *optional*, defaults to `"finance_corpus"`):
                Name of the ChromaDB collection to use.
            persist_directory (`str`, *optional*, defaults to `"./chroma_db"`):
                Directory to persist ChromaDB data.
            batch_size (`int`, *optional*, defaults to `64`):
                Batch size for encoding queries and corpus documents.
        """
        self.model = model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.batch_size = batch_size
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
        queries: Dict[str, str],
        top_k: Optional[int] = None,
        score_function: Optional[str] = "cos_sim",
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Implements the abstract Retrieval.retrieve() by delegating to the internal search.

        Args:
            corpus: Mapping of document IDs to {title, text}.
            queries: Mapping of query IDs to query text.
            top_k: Number of results per query; defaults to full corpus if None.
            score_function: Kept for interface compatibility (cosine used internally).
            return_sorted: Whether to sort results by score desc.
            **kwargs: Additional args (unused).

        Returns:
            Dict mapping query_id -> {doc_id: score}.
        """
        if top_k is None:
            top_k = len(corpus)
        return self.search(
            corpus=corpus,
            queries=queries,
            top_k=top_k,
            score_function=score_function or "cos_sim",
            return_sorted=return_sorted,
            **kwargs,
        )

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "cos_sim",
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Searches the corpus using ChromaDB for the given queries and returns the top-k results.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                Dictionary mapping document IDs to documents with 'title' and 'text' fields.
            queries (`Dict[str, str]`):
                Dictionary mapping query IDs to query text.
            top_k (`int`):
                Number of top results to retrieve per query.
            score_function (`str`, *optional*, defaults to `"cos_sim"`):
                Similarity function (ChromaDB uses cosine similarity by default).
            return_sorted (`bool`, *optional*, defaults to `False`):
                Whether to sort results by score in descending order.
            **kwargs:
                Additional arguments.

        Returns:
            `Dict[str, Dict[str, float]]`:
                Nested dictionary with query IDs as keys, and document IDs mapped to similarity scores.
        """
        logger.info("Starting ChromaDB-based retrieval...")
        
        # Step 1: Create or get collection and index corpus
        self._index_corpus(corpus)
        
        # Step 2: Encode queries and search
        results = {}
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        logger.info(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self.model.encode_queries(
            query_texts,
            batch_size=self.batch_size,
        )
        
        # Search for each query
        for i, query_id in enumerate(query_ids):
            query_embedding = query_embeddings[i].tolist()
            
            # Query ChromaDB
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(corpus)),
                include=["distances"],
            )
            
            # Format results
            doc_ids = chroma_results['ids'][0]
            distances = chroma_results.get('distances', [[]])[0]
            
            # Convert distances to similarity scores (ChromaDB returns L2 distances)
            # For cosine similarity: similarity = 1 - distance
            scores = {doc_id: 1.0 - dist for doc_id, dist in zip(doc_ids, distances)}
            
            if return_sorted:
                scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            
            results[query_id] = scores
        
        logger.info(f"Retrieval complete for {len(results)} queries.")
        return results

    def _index_corpus(self, corpus: Dict[str, Dict[str, str]]):
        """
        Indexes the corpus into ChromaDB.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                Dictionary mapping document IDs to documents with 'title' and 'text' fields.
        """
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            existing_count = self.collection.count()
            
            # If collection exists and has the same number of documents, skip indexing
            if existing_count == len(corpus):
                logger.info(f"Collection '{self.collection_name}' already exists with {existing_count} documents. Skipping indexing.")
                return
            else:
                logger.info(f"Collection exists but has different document count ({existing_count} vs {len(corpus)}). Recreating...")
                self.client.delete_collection(name=self.collection_name)
        except Exception:
            logger.info(f"Creating new collection '{self.collection_name}'...")
        
        # Create collection with cosine similarity
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare corpus documents
        doc_ids = list(corpus.keys())
        doc_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in doc_ids
        ]
        
        logger.info(f"Encoding {len(doc_texts)} documents...")
        
        # Encode corpus in batches (Encoder expects list[{'title','text'}])
        all_embeddings = []
        for i in range(0, len(doc_ids), self.batch_size):
            batch_ids = doc_ids[i:i + self.batch_size]
            batch_docs = [
                {
                    "title": corpus[doc_id].get("title", ""),
                    "text": corpus[doc_id].get("text", ""),
                }
                for doc_id in batch_ids
            ]
            batch_embeddings = self.model.encode_corpus(batch_docs, batch_size=self.batch_size)
            all_embeddings.extend([emb.tolist() for emb in batch_embeddings])

            if (i + self.batch_size) % 1000 == 0 or (i + self.batch_size) >= len(doc_ids):
                logger.info(f"Encoded {min(i + self.batch_size, len(doc_ids))}/{len(doc_ids)} documents")
        
        # Add to ChromaDB in batches
        logger.info("Adding documents to ChromaDB...")
        batch_size_insert = 1000  # ChromaDB works well with larger batches for insertion
        
        for i in range(0, len(doc_ids), batch_size_insert):
            end_idx = min(i + batch_size_insert, len(doc_ids))
            self.collection.add(
                ids=doc_ids[i:end_idx],
                embeddings=all_embeddings[i:end_idx],
                documents=doc_texts[i:end_idx]
            )
            logger.info(f"Added {end_idx}/{len(doc_ids)} documents to ChromaDB")
        
        logger.info(f"Indexed {len(doc_ids)} documents into ChromaDB collection '{self.collection_name}'")
