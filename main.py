import json
import logging
import pandas as pd

from financerag.tasks import FinDER
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from financerag.tasks.BaseTask import BaseTask  # optional if you prefer static evaluate

logging.basicConfig(level=logging.INFO)

CORPUS_PATH = "/Users/vikashpr/Dev/Python/FinanceRAG/icaif-24-finance-rag-challenge/finder_corpus.jsonl/corpus.jsonl"
QUERY_PATH = "/Users/vikashpr/Dev/Python/FinanceRAG/icaif-24-finance-rag-challenge/finder_queries.jsonl/queries.jsonl"
QRELS_PATH = "/Users/vikashpr/Dev/Python/FinanceRAG/icaif-24-finance-rag-challenge/FinDER_qrels.tsv"

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

class LocalFinDER(FinDER):
    def load_data(self):
        # override BaseTask.load_data so it doesn't try HF
        self.queries = {}
        self.corpus = {}

corpus = {
    doc["_id"]: {"title": doc.get("title", ""), "text": doc.get("text", "")}
    for doc in load_jsonl(CORPUS_PATH)
}
queries = {q["_id"]: q["text"] for q in load_jsonl(QUERY_PATH)}

finder_task = LocalFinDER()
finder_task.corpus = corpus
finder_task.queries = queries

df = pd.read_csv(QRELS_PATH, sep="\t")
qrels = df.groupby("query_id").apply(lambda g: dict(zip(g["corpus_id"], g["score"]))).to_dict()

print("Querry Result", qrels)

encoder = SentenceTransformerEncoder(
    model_name_or_path="intfloat/e5-large-v2",
    query_prompt="query: ",
    doc_prompt="passage: ",
)
retriever = DenseRetrieval(model=encoder)

import time

# Measure just the retrieval call time
start = time.perf_counter()
retrieval_result = finder_task.retrieve(retriever=retriever, top_k=100)
retrieval_time_s = time.perf_counter() - start

print(f"Retrieval time (DenseRetrieval): {retrieval_time_s:.2f}s")

ndcg, map_, recall, precision = finder_task.evaluate(
    qrels=qrels,
    results=retrieval_result,
    k_values=[1, 5, 10],
)
print("NDCG:", ndcg)
print("MAP:", map_)
print("Recall:", recall)
print("Precision:", precision)

finder_task.save_results(output_dir="./results")
