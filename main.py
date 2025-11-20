import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# Paths
CORPUS_PATH = "/Users/vikashpr/Dev/Python/FinanceRAG/icaif-24-finance-rag-challenge/finder_corpus.jsonl/corpus.jsonl"
QUERY_PATH = "/Users/vikashpr/Dev/Python/FinanceRAG/icaif-24-finance-rag-challenge/finder_queries.jsonl/queries.jsonl"
QRELS_PATH = "/Users/vikashpr/Dev/Python/FinanceRAG/icaif-24-finance-rag-challenge/FinDER_qrels.tsv"

# Load data
def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

# Load corpus and queries
corpus = {
    doc["_id"]: {"title": doc.get("title", ""), "text": doc.get("text", "")}
    for doc in load_jsonl(CORPUS_PATH)
}
queries = {q["_id"]: q["text"] for q in load_jsonl(QUERY_PATH)}

# Load ground truth (qrels)
df = pd.read_csv(QRELS_PATH, sep="\t")
qrels = df.groupby("query_id").apply(lambda g: dict(zip(g["corpus_id"], g["score"]))).to_dict()

print(f"Loaded {len(corpus)} documents")
print(f"Loaded {len(queries)} queries")
print(f"Loaded qrels for {len(qrels)} queries")

# Initialize ChromaDB
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\nInitializing ChromaDB with model: {MODEL_NAME}")

client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)

# Get or create collection
collection_name = "finder_corpus"
try:
    collection = client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    print(f"✓ Loaded existing collection with {collection.count()} documents")
except:
    print("Creating new collection and adding documents...")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents to ChromaDB
    corpus_ids = list(corpus.keys())
    corpus_texts = [f"{corpus[doc_id]['title']} {corpus[doc_id]['text']}" for doc_id in corpus_ids]
    
    # Add in batches
    batch_size = 100
    for i in range(0, len(corpus_ids), batch_size):
        batch_ids = corpus_ids[i:i+batch_size]
        batch_texts = corpus_texts[i:i+batch_size]
        collection.add(
            ids=batch_ids,
            documents=batch_texts
        )
        print(f"Added {min(i+batch_size, len(corpus_ids))}/{len(corpus_ids)} documents")
    
    print(f"✓ Collection created with {collection.count()} documents")

# Retrieve: Query ChromaDB for each query
print("\nQuerying ChromaDB for each query...")
top_k = 100
results = {}

query_ids = list(queries.keys())
query_texts = [queries[q_id] for q_id in query_ids]

for query_id, query_text in zip(query_ids, query_texts):
    result = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    # Store results as {doc_id: score}
    # ChromaDB returns distances, convert to similarity scores (1 - distance for cosine)
    results[query_id] = {
        doc_id: 1 - distance 
        for doc_id, distance in zip(result['ids'][0], result['distances'][0])
    }

print(f"✓ Retrieved top-{top_k} documents for each query")

# Evaluate: Compare with ground truth
def calculate_metrics(qrels, results, k_values=[1, 5, 10]):
    """Calculate NDCG, MAP, Recall, and Precision"""
    from collections import defaultdict
    
    ndcg_scores = defaultdict(list)
    map_scores = defaultdict(list)
    recall_scores = defaultdict(list)
    precision_scores = defaultdict(list)
    
    for query_id in qrels:
        if query_id not in results:
            continue
            
        # Get relevant documents from qrels
        relevant_docs = set(doc_id for doc_id, score in qrels[query_id].items() if score > 0)
        
        # Get retrieved documents (sorted by score)
        retrieved_docs = list(results[query_id].keys())
        
        for k in k_values:
            retrieved_at_k = retrieved_docs[:k]
            
            # Precision@k
            num_relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant_docs])
            precision = num_relevant_retrieved / k if k > 0 else 0
            precision_scores[k].append(precision)
            
            # Recall@k
            recall = num_relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
            recall_scores[k].append(recall)
            
            # NDCG@k
            dcg = sum([1 / np.log2(i + 2) if retrieved_at_k[i] in relevant_docs else 0 
                      for i in range(len(retrieved_at_k))])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_docs), k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores[k].append(ndcg)
            
            # MAP is typically calculated once (not per k), but we'll compute it for consistency
            relevant_retrieved = [(i, doc) for i, doc in enumerate(retrieved_at_k) if doc in relevant_docs]
            if relevant_retrieved:
                ap = sum([(j + 1) / (i + 1) for j, (i, _) in enumerate(relevant_retrieved)]) / len(relevant_docs)
            else:
                ap = 0
            map_scores[k].append(ap)
    
    # Average across all queries
    ndcg = {k: np.mean(scores) for k, scores in ndcg_scores.items()}
    map_metric = {k: np.mean(scores) for k, scores in map_scores.items()}
    recall = {k: np.mean(scores) for k, scores in recall_scores.items()}
    precision = {k: np.mean(scores) for k, scores in precision_scores.items()}
    
    return ndcg, map_metric, recall, precision

# Calculate metrics
print("\nEvaluating against ground truth...")
ndcg, map_metric, recall, precision = calculate_metrics(qrels, results, k_values=[1, 5, 10])

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"NDCG@k:     {ndcg}")
print(f"MAP@k:      {map_metric}")
print(f"Recall@k:   {recall}")
print(f"Precision@k: {precision}")
print("="*50)