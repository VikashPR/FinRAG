"""
PostgreSQL Vector Retrieval using pgvector extension.

This module provides a retrieval implementation that stores embeddings in PostgreSQL
using the pgvector extension for efficient similarity search.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from financerag.common.protocols import Encoder, Retrieval

logger = logging.getLogger(__name__)


class PostgresVectorRetrieval(Retrieval):
    """
    PostgreSQL-based dense retrieval using pgvector extension.
    
    This class stores document embeddings in PostgreSQL using pgvector and performs
    similarity-based search using cosine similarity or inner product distance.
    """

    def __init__(
        self,
        model: Encoder,
        connection_string: str,
        table_name: str = "document_embeddings",
        embedding_dim: int = 1024,
        batch_size: int = 64,
        index_type: str = "ivfflat",  # 'ivfflat', 'hnsw', or 'none'
        recreate_table: bool = False,
    ):
        """
        Initialize PostgreSQL Vector Retrieval.

        Args:
            model: Encoder model for generating embeddings
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:password@localhost:5432/dbname"
            table_name: Name of the table to store embeddings
            embedding_dim: Dimension of embeddings (e.g., 1024 for e5-large-v2)
            batch_size: Batch size for encoding
            index_type: Type of index to create ('ivfflat', 'hnsw', or 'none')
            recreate_table: If True, drop and recreate the table
        """
        self.model = model
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.index_type = index_type
        self.recreate_table = recreate_table
        self.results: Dict = {}
        
        # Timing metrics
        self.timing_metrics: Dict[str, float] = {}
        
        # Initialize database connection and setup
        self._setup_database()

    def _get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)

    def _setup_database(self):
        """Set up the database with pgvector extension and create table."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                if self.recreate_table:
                    cur.execute(f"DROP TABLE IF EXISTS {self.table_name};")
                    logger.info(f"Dropped existing table: {self.table_name}")
                
                # Create table for embeddings
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        doc_id TEXT UNIQUE NOT NULL,
                        title TEXT,
                        content TEXT,
                        embedding vector({self.embedding_dim})
                    );
                """)
                
                conn.commit()
                logger.info(f"Database setup complete. Table: {self.table_name}")
        finally:
            conn.close()

    def _create_index(self, num_docs: int):
        """Create vector index for efficient similarity search."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Drop existing index if any
                cur.execute(f"DROP INDEX IF EXISTS {self.table_name}_embedding_idx;")
                
                if self.index_type == "ivfflat":
                    # IVF-Flat index - good balance of speed and recall
                    # Number of lists should be sqrt(num_docs) for optimal performance
                    num_lists = max(1, int(np.sqrt(num_docs)))
                    cur.execute(f"""
                        CREATE INDEX {self.table_name}_embedding_idx 
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = {num_lists});
                    """)
                    logger.info(f"Created IVF-Flat index with {num_lists} lists")
                    
                elif self.index_type == "hnsw":
                    # HNSW index - faster search but slower build
                    cur.execute(f"""
                        CREATE INDEX {self.table_name}_embedding_idx 
                        ON {self.table_name} 
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64);
                    """)
                    logger.info("Created HNSW index")
                else:
                    logger.info("No index created (exact search)")
                
                conn.commit()
        finally:
            conn.close()

    def _check_corpus_exists(self) -> int:
        """Check if corpus is already indexed and return count."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                count = cur.fetchone()[0]
                return count
        finally:
            conn.close()

    def _store_embeddings(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
    ) -> float:
        """
        Store corpus embeddings in PostgreSQL.
        
        Returns:
            Time taken for storing embeddings in seconds.
        """
        start_time = time.time()
        
        # Check if data already exists
        existing_count = self._check_corpus_exists()
        if existing_count > 0 and not self.recreate_table:
            logger.info(f"Found {existing_count} existing documents. Skipping indexing.")
            return time.time() - start_time
        
        logger.info(f"Encoding and storing {len(corpus)} documents...")
        
        # Sort corpus by document length for efficient batching
        sorted_corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        
        corpus_list = [corpus[cid] for cid in sorted_corpus_ids]
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Process in batches
                for batch_start in range(0, len(corpus_list), self.batch_size * 10):
                    batch_end = min(batch_start + self.batch_size * 10, len(corpus_list))
                    batch_corpus = corpus_list[batch_start:batch_end]
                    batch_ids = sorted_corpus_ids[batch_start:batch_end]
                    
                    # Encode batch
                    embeddings = self.model.encode_corpus(
                        batch_corpus, batch_size=self.batch_size
                    )
                    
                    if hasattr(embeddings, 'cpu'):
                        embeddings = embeddings.cpu().numpy()
                    
                    # Prepare data for insertion
                    data = []
                    for i, (doc_id, doc, emb) in enumerate(zip(batch_ids, batch_corpus, embeddings)):
                        title = doc.get("title", "")
                        text = doc.get("text", "")
                        # Convert embedding to list for pgvector
                        emb_list = emb.tolist()
                        data.append((doc_id, title, text, emb_list))
                    
                    # Batch insert using execute_values
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {self.table_name} (doc_id, title, content, embedding)
                        VALUES %s
                        ON CONFLICT (doc_id) DO NOTHING;
                        """,
                        data,
                        template="(%s, %s, %s, %s::vector)"
                    )
                    
                    conn.commit()
                    logger.info(f"Stored batch {batch_start // (self.batch_size * 10) + 1}")
                
            # Create index after all data is inserted
            self._create_index(len(corpus))
            
        finally:
            conn.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Stored {len(corpus)} documents in {elapsed:.2f} seconds")
        return elapsed

    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
        queries: Dict[str, str],
        top_k: Optional[int] = None,
        score_function: Literal["cos_sim", "dot"] | None = "cos_sim",
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve top-k documents from PostgreSQL for each query.

        Args:
            corpus: Dictionary of document ID to document content
            queries: Dictionary of query ID to query text
            top_k: Number of top documents to retrieve
            score_function: Scoring function ('cos_sim' for cosine, 'dot' for inner product)
            return_sorted: Whether to return sorted results (always sorted for pgvector)
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping query IDs to dictionaries of doc IDs and scores
        """
        if top_k is None:
            top_k = 100
            
        # Reset timing metrics
        self.timing_metrics = {}
        
        # Store corpus embeddings (will skip if already indexed)
        indexing_time = self._store_embeddings(corpus)
        self.timing_metrics['indexing_time'] = indexing_time
        
        # Encode queries
        logger.info(f"Encoding {len(queries)} queries...")
        query_start = time.time()
        
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, **kwargs
        )
        
        if hasattr(query_embeddings, 'cpu'):
            query_embeddings = query_embeddings.cpu().numpy()
        
        query_encoding_time = time.time() - query_start
        self.timing_metrics['query_encoding_time'] = query_encoding_time
        logger.info(f"Query encoding took {query_encoding_time:.2f} seconds")
        
        # Perform retrieval
        logger.info(f"Retrieving top-{top_k} documents for {len(queries)} queries...")
        retrieval_start = time.time()
        
        # Select distance operator based on score function
        if score_function == "cos_sim":
            distance_op = "<=>"  # Cosine distance
        else:
            distance_op = "<#>"  # Inner product (negative)
        
        results = {qid: {} for qid in query_ids}
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Set probes for IVF-Flat (higher = better recall, slower)
                if self.index_type == "ivfflat":
                    cur.execute("SET ivfflat.probes = 10;")
                elif self.index_type == "hnsw":
                    cur.execute("SET hnsw.ef_search = 100;")
                
                for i, (qid, q_emb) in enumerate(zip(query_ids, query_embeddings)):
                    # Convert query embedding to string format for pgvector
                    q_emb_str = '[' + ','.join(map(str, q_emb.tolist())) + ']'
                    
                    # Query for similar documents
                    if score_function == "cos_sim":
                        # Cosine similarity = 1 - cosine distance
                        cur.execute(f"""
                            SELECT doc_id, 1 - (embedding {distance_op} %s::vector) as score
                            FROM {self.table_name}
                            WHERE doc_id != %s
                            ORDER BY embedding {distance_op} %s::vector
                            LIMIT %s;
                        """, (q_emb_str, qid, q_emb_str, top_k))
                    else:
                        # Inner product (pgvector returns negative, so negate)
                        cur.execute(f"""
                            SELECT doc_id, -(embedding {distance_op} %s::vector) as score
                            FROM {self.table_name}
                            WHERE doc_id != %s
                            ORDER BY embedding {distance_op} %s::vector
                            LIMIT %s;
                        """, (q_emb_str, qid, q_emb_str, top_k))
                    
                    rows = cur.fetchall()
                    for doc_id, score in rows:
                        results[qid][doc_id] = float(score)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(query_ids)} queries")
                        
        finally:
            conn.close()
        
        retrieval_time = time.time() - retrieval_start
        self.timing_metrics['retrieval_time'] = retrieval_time
        self.timing_metrics['avg_query_time'] = retrieval_time / len(queries)
        self.timing_metrics['total_time'] = indexing_time + query_encoding_time + retrieval_time
        
        logger.info(f"Retrieval took {retrieval_time:.2f} seconds")
        logger.info(f"Average time per query: {self.timing_metrics['avg_query_time']*1000:.2f} ms")
        
        self.results = results
        return results

    def get_timing_metrics(self) -> Dict[str, float]:
        """
        Get timing metrics from the last retrieval operation.
        
        Returns:
            Dictionary containing timing information:
            - indexing_time: Time to encode and store corpus
            - query_encoding_time: Time to encode queries
            - retrieval_time: Time to retrieve from database
            - avg_query_time: Average time per query
            - total_time: Total operation time
        """
        return self.timing_metrics

    def clear_index(self):
        """Clear all data from the embeddings table."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
                conn.commit()
                logger.info(f"Cleared all data from {self.table_name}")
        finally:
            conn.close()
