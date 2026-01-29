import logging
import time
from typing import Dict, List, Optional, Literal
import pathlib

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from financerag.common.protocols import Encoder, Retrieval

logger = logging.getLogger(__name__)


class MilvusRetrieval(Retrieval):
    """
    Milvus-based dense retrieval that stores embeddings in Milvus and performs similarity search.
    
    This class uses Milvus as a vector database to store corpus embeddings and efficiently retrieve
    the top-k most relevant documents for each query using cosine similarity.
    """

    def __init__(
        self,
        model: Encoder,
        collection_name: str = "finance_corpus",
        uri: str = "./milvus_db/milvus.db",
        batch_size: int = 64,
        embedding_dim: int = 1024,
        index_type: str = "AUTOINDEX",
        index_params: Optional[Dict] = None,
        search_params: Optional[Dict] = None,
    ):
        """
        Initializes the MilvusRetrieval class.

        Args:
            model (`Encoder`):
                An encoder model implementing the `Encoder` protocol for encoding queries and corpus documents.
            collection_name (`str`, *optional*, defaults to `"finance_corpus"`):
                Name of the Milvus collection to use.
            uri (`str`, *optional*, defaults to `"./milvus_db/milvus.db"`):
                URI for Milvus connection. Use local path for Milvus Lite or server address for Milvus server.
            batch_size (`int`, *optional*, defaults to `64`):
                Batch size for encoding queries and corpus documents.
            embedding_dim (`int`, *optional*, defaults to `1024`):
                Dimension of the embeddings (e.g., 1024 for e5-large-v2).
            index_type (`str`, *optional*, defaults to `"AUTOINDEX"`):
                Index type for Milvus. Options: FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, DISKANN, etc.
            index_params (`Dict`, *optional*):
                Additional parameters for index creation (e.g., nlist for IVF, M and efConstruction for HNSW).
            search_params (`Dict`, *optional*):
                Search-time parameters (e.g., nprobe for IVF indices, ef for HNSW).
        """
        self.model = model
        self.collection_name = collection_name
        self.uri = uri
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_params = index_params or {}
        self.search_params = search_params or {}
        
        # Create parent directory for Milvus Lite if it doesn't exist
        if uri.endswith(".db"):
            parent_dir = pathlib.Path(uri).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created Milvus directory: {parent_dir}")
        
        # Initialize Milvus client (using MilvusClient for simplicity - Milvus Lite)
        self.client = MilvusClient(uri=uri)
        
        logger.info(f"Initialized Milvus client with URI: {uri}, Index Type: {index_type}")
        self.timing_metrics: Dict[str, float] = {}

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
        Searches the corpus using Milvus for the given queries and returns the top-k results.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                Dictionary mapping document IDs to documents with 'title' and 'text' fields.
            queries (`Dict[str, str]`):
                Dictionary mapping query IDs to query text.
            top_k (`int`):
                Number of top results to retrieve per query.
            score_function (`str`, *optional*, defaults to `"cos_sim"`):
                Similarity function (Milvus uses cosine similarity).
            return_sorted (`bool`, *optional*, defaults to `False`):
                Whether to sort results by score in descending order.
            **kwargs:
                Additional arguments.

        Returns:
            `Dict[str, Dict[str, float]]`:
                Nested dictionary with query IDs as keys, and document IDs mapped to similarity scores.
        """
        total_start = time.perf_counter()
        self.timing_metrics = {}
        logger.info("Starting Milvus-based retrieval...")
        
        # Step 1: Create collection and index corpus
        indexing_time = self._index_corpus(corpus)
        self.timing_metrics["indexing_time"] = indexing_time
        
        # Step 2: Encode queries and search
        results = {}
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        logger.info(f"Encoding {len(query_texts)} queries...")
        qenc_start = time.perf_counter()
        query_embeddings = self.model.encode_queries(
            query_texts,
            batch_size=self.batch_size,
        )
        query_encoding_time = time.perf_counter() - qenc_start
        self.timing_metrics["query_encoding_time"] = query_encoding_time
        
        # Search for each query
        search_start = time.perf_counter()
        for i, query_id in enumerate(query_ids):
            query_embedding = query_embeddings[i].tolist()
            
            # Prepare search parameters
            search_params_dict = {
                "metric_type": "COSINE",
                "params": self.search_params
            }
            
            # Query Milvus with search parameters
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=min(top_k, len(corpus)),
                output_fields=["doc_id"],
                search_params=search_params_dict,
            )
            
            # Format results (search_results is a list of lists)
            scores = {}
            for hit in search_results[0]:
                doc_id = hit.get("entity", {}).get("doc_id") or hit.get("id")
                # Milvus returns distance, convert to similarity
                # For cosine metric: distance is already similarity (1 - cosine_distance)
                similarity = hit.get("distance", 0.0)
                scores[doc_id] = float(similarity)
            
            if return_sorted:
                scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            
            results[query_id] = scores

        retrieval_time = time.perf_counter() - search_start
        self.timing_metrics["retrieval_time"] = retrieval_time
        if len(queries) > 0:
            self.timing_metrics["avg_query_time"] = retrieval_time / len(queries)
        self.timing_metrics["total_time"] = time.perf_counter() - total_start

        logger.info(
            "MilvusRetrieval timings | "
            f"indexing={self.timing_metrics['indexing_time']:.2f}s, "
            f"query_encode={self.timing_metrics['query_encoding_time']:.2f}s, "
            f"retrieval={self.timing_metrics['retrieval_time']:.2f}s, "
            f"total={self.timing_metrics['total_time']:.2f}s"
        )
        logger.info(f"Retrieval complete for {len(results)} queries.")
        return results

    def _index_corpus(self, corpus: Dict[str, Dict[str, str]]) -> float:
        """
        Indexes the corpus into Milvus.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                Dictionary mapping document IDs to documents with 'title' and 'text' fields.
        """
        start = time.perf_counter()
        # Check if collection exists
        if self.client.has_collection(collection_name=self.collection_name):
            # Get collection stats
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            existing_count = stats.get("row_count", 0)
            
            # If collection exists and has the same number of documents, skip indexing
            if existing_count == len(corpus):
                logger.info(
                    f"Collection '{self.collection_name}' already exists with {existing_count} documents. Skipping indexing."
                )
                return time.perf_counter() - start
            else:
                logger.info(
                    f"Collection exists but has different document count ({existing_count} vs {len(corpus)}). Recreating..."
                )
                self.client.drop_collection(collection_name=self.collection_name)
        
        # Create collection with schema
        logger.info(f"Creating new collection '{self.collection_name}' with index type: {self.index_type}...")
        
        # Get index parameters based on index type
        index_params = self._get_index_params()
        
        # MilvusClient.create_collection with index parameters
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",
            auto_id=True,
            index_params=index_params,
        )
        
        # Prepare corpus documents
        doc_ids = list(corpus.keys())
        doc_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in doc_ids
        ]
        
        logger.info(f"Encoding {len(doc_texts)} documents...")
        
        # Encode corpus in batches
        all_embeddings = []
        for i in range(0, len(doc_ids), self.batch_size):
            batch_ids = doc_ids[i : i + self.batch_size]
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
        
        # Prepare data for insertion
        logger.info("Adding documents to Milvus...")
        batch_size_insert = 1000  # Insert in batches
        
        for i in range(0, len(doc_ids), batch_size_insert):
            end_idx = min(i + batch_size_insert, len(doc_ids))
            
            # Prepare batch data
            batch_data = [
                {
                    "doc_id": doc_ids[j],
                    "vector": all_embeddings[j],
                }
                for j in range(i, end_idx)
            ]
            
            # Insert into Milvus
            self.client.insert(
                collection_name=self.collection_name,
                data=batch_data,
            )
            
            logger.info(f"Added {end_idx}/{len(doc_ids)} documents to Milvus")
        
        logger.info(f"Indexed {len(doc_ids)} documents into Milvus collection '{self.collection_name}'")
        return time.perf_counter() - start

    def _get_index_params(self) -> Dict:
        """
        Get index parameters based on the index type.
        
        Returns:
            Dict containing index parameters for Milvus.
        """
        # Default parameters for each index type
        default_params = {
            "FLAT": {},
            "IVF_FLAT": {"nlist": 128},
            "IVF_SQ8": {"nlist": 128},
            "IVF_PQ": {"nlist": 128, "m": 8, "nbits": 8},
            "HNSW": {"M": 16, "efConstruction": 200},
            "DISKANN": {},
            "AUTOINDEX": {},
        }
        
        # Get base params for the index type
        base_params = default_params.get(self.index_type, {})
        
        # Merge with user-provided params (user params take precedence)
        params = {**base_params, **self.index_params}
        
        return {
            "index_type": self.index_type,
            "metric_type": "COSINE",
            "params": params
        }

    def get_timing_metrics(self) -> Dict[str, float]:
        """
        Get timing metrics from the last retrieval operation.

        Returns:
            Dictionary containing timing information:
            - indexing_time: Time to index corpus (may be near-zero if skipped)
            - query_encoding_time: Time to encode queries
            - retrieval_time: Time spent in Milvus searches
            - avg_query_time: Average retrieval time per query (search only)
            - total_time: Total time for search()
        """
        return self.timing_metrics
