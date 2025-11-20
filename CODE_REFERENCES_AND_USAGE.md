# Code References and Usage Explanation

## 📚 Reference Code Sources

This codebase is adapted from several open-source projects:

### 1. **BaseTask.py** 
- **Reference**: Adapted from [MTEB (Massive Text Embedding Benchmark)](https://github.com/embeddings-benchmark/mteb)
- **Original Source**: `https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/AbsTask.py`
- **What it provides**: Base class structure for handling retrieval, reranking, and generation tasks

### 2. **Evaluation Method in BaseTask.py**
- **Reference**: Adapted from [BEIR (Benchmarking IR)](https://github.com/beir-cellar/beir)
- **Original Source**: `https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py`
- **What it provides**: Evaluation metrics (NDCG, MAP, Recall, Precision) using `pytrec_eval`

### 3. **DenseRetrieval (dense.py)**
- **Reference**: Adapted from BEIR
- **Original Source**: `https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/exact_search.py`
- **Utility Functions**: `https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/util.py`
- **What it provides**: Dense retrieval using cosine similarity and dot product scoring

### 4. **SentenceTransformerEncoder (sent_encoder.py)**
- **Reference**: Adopted from BEIR
- **Original Source**: `https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sentence_bert.py`
- **What it provides**: Wrapper for SentenceTransformer models to encode queries and documents

### 5. **CrossEncoderReranker (cross_encoder.py)**
- **Reference**: Adapted from BEIR
- **Original Source**: `https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py`
- **What it provides**: Cross-encoder based reranking for query-document pairs

---

## 🔄 How BaseTask.py and FinDERTask.py are Utilized

### Class Inheritance Hierarchy

```
BaseTask (abstract base class)
    ↓
FinDER (task-specific implementation)
    ↓
LocalFinDER (notebook-specific override)
```

### Step-by-Step Usage Flow in main.ipynb

#### **Step 1: Import and Class Definition**

```python
from financerag.tasks import FinDER
from financerag.tasks.BaseTask import BaseTask

class LocalFinDER(FinDER):
    def load_data(self):
        # Override BaseTask.load_data() to skip HuggingFace loading
        self.queries = {}
        self.corpus = {}
```

**What happens:**
- `FinDER` inherits from `BaseTask`
- `LocalFinDER` inherits from `FinDER`
- `LocalFinDER.load_data()` overrides the parent method to prevent loading from HuggingFace

#### **Step 2: Instantiation**

```python
finder_task = LocalFinDER()
finder_task.corpus = corpus
finder_task.queries = queries
```

**What happens internally:**
1. `LocalFinDER.__init__()` is called
2. It calls `FinDER.__init__()` which:
   - Creates `TaskMetadata` with FinDER dataset info
   - Calls `super().__init__(self.metadata)` → `BaseTask.__init__()`
   - `BaseTask.__init__()` calls `self.load_data()`
   - But `LocalFinDER.load_data()` overrides it, so it just initializes empty dicts
3. Then manually sets `corpus` and `queries` from local JSONL files

**Inherited attributes from BaseTask:**
- `self.metadata` (TaskMetadata object)
- `self.queries` (Dict[str, str])
- `self.corpus` (Dict[str, Dict[str, str]])
- `self.retrieve_results` (None initially)
- `self.rerank_results` (None initially)
- `self.generate_results` (None initially)

#### **Step 3: Retrieval**

```python
retrieval_result = finder_task.retrieve(retriever=retriever, top_k=200)
```

**Method call chain:**
1. `finder_task.retrieve()` → `BaseTask.retrieve()` (inherited method)
2. `BaseTask.retrieve()` validates:
   - Checks if `retriever` is a subclass of `Retrieval` protocol
   - Checks if `self.corpus` and `self.queries` are loaded
3. Calls `retriever.retrieve(queries=self.queries, corpus=self.corpus, top_k=200)`
4. Stores result in `self.retrieve_results`
5. Returns `Dict[str, Dict[str, float]]` (query_id → {doc_id: score})

**What BaseTask provides:**
- Input validation
- Error handling
- Result storage (`self.retrieve_results`)
- Consistent interface across all tasks

#### **Step 4: Reranking**

```python
reranking_result = finder_task.rerank(
    reranker=reranker,
    results=retrieval_result,
    top_k=100,
    batch_size=32,
)
```

**Method call chain:**
1. `finder_task.rerank()` → `BaseTask.rerank()` (inherited method)
2. `BaseTask.rerank()` validates:
   - Checks if `reranker` is a subclass of `Reranker` protocol
   - Checks if `self.corpus` and `self.queries` are loaded
   - If `results=None`, uses `self.retrieve_results` (fallback)
3. Calls `reranker.rerank(queries=self.queries, corpus=self.corpus, results=results, top_k=100, batch_size=32)`
4. Stores result in `self.rerank_results`
5. Returns `Dict[str, Dict[str, float]]` (reranked scores)

**What BaseTask provides:**
- Automatic fallback to `retrieve_results` if `results=None`
- Input validation
- Result storage (`self.rerank_results`)
- Consistent reranking interface

#### **Step 5: Evaluation**

```python
ndcg, map_, recall, precision = finder_task.evaluate(
    qrels=qrels,
    results=reranking_result,
    k_values=[1, 5, 10],
)
```

**Method call chain:**
1. `finder_task.evaluate()` → `BaseTask.evaluate()` (static method, inherited)
2. `BaseTask.evaluate()`:
   - Filters results to only queries present in `qrels`
   - Uses `pytrec_eval.RelevanceEvaluator` to compute metrics
   - Calculates NDCG, MAP, Recall, Precision at k=[1, 5, 10]
   - Returns averaged metrics across all queries

**What BaseTask provides:**
- Standard IR evaluation metrics
- Integration with `pytrec_eval` library
- Filtering and aggregation logic

#### **Step 6: Save Results**

```python
finder_task.save_results(output_dir="./results")
```

**Method call chain:**
1. `finder_task.save_results()` → `BaseTask.save_results()` (inherited method)
2. `BaseTask.save_results()`:
   - Creates output directory: `./results/FinDER/` (uses `self.metadata.name`)
   - Determines which results to save:
     - Prefers `self.rerank_results` if available
     - Falls back to `self.retrieve_results` if rerank_results is None
   - Saves top-k results to CSV: `results/FinDER/results.csv`
   - Format: `query_id,corpus_id` (one row per query-document pair)
   - If `self.generate_results` exists, saves to JSONL: `results/FinDER/output.jsonl`

**What BaseTask provides:**
- Automatic directory creation
- Result selection logic (rerank > retrieve)
- CSV and JSONL export formats
- Uses `self.metadata.name` for task-specific folder naming

---

## 📊 Complete Method Flow Diagram

```
main.ipynb
│
├─ LocalFinDER.__init__()
│  └─ FinDER.__init__()
│     └─ BaseTask.__init__()
│        └─ LocalFinDER.load_data() [OVERRIDDEN]
│
├─ finder_task.retrieve()
│  └─ BaseTask.retrieve() [INHERITED]
│     ├─ Validates retriever type
│     ├─ Validates data loaded
│     └─ Calls retriever.retrieve()
│        └─ DenseRetrieval.retrieve()
│           ├─ Encodes queries
│           ├─ Encodes corpus (batched)
│           ├─ Computes cosine similarity
│           └─ Returns top-k results
│
├─ finder_task.rerank()
│  └─ BaseTask.rerank() [INHERITED]
│     ├─ Validates reranker type
│     ├─ Validates data loaded
│     ├─ Falls back to retrieve_results if results=None
│     └─ Calls reranker.rerank()
│        └─ CrossEncoderReranker.rerank()
│           ├─ Prepares query-doc pairs
│           ├─ Scores pairs (batched)
│           └─ Returns reranked results
│
├─ finder_task.evaluate()
│  └─ BaseTask.evaluate() [INHERITED - STATIC]
│     ├─ Filters results by qrels
│     ├─ Uses pytrec_eval
│     └─ Returns metrics dicts
│
└─ finder_task.save_results()
   └─ BaseTask.save_results() [INHERITED]
      ├─ Creates output directory
      ├─ Selects final_result (rerank > retrieve)
      ├─ Saves CSV (top-k results)
      └─ Saves JSONL (if generate_results exists)
```

---

## 🎯 Key Benefits of This Architecture

### 1. **Code Reusability**
- `BaseTask` provides common functionality for all tasks (FinDER, FinQA, TATQA, etc.)
- Each task only needs to define metadata

### 2. **Consistent Interface**
- All tasks use the same methods: `retrieve()`, `rerank()`, `evaluate()`, `save_results()`
- Easy to swap between different tasks

### 3. **Separation of Concerns**
- `BaseTask`: Task orchestration and data management
- `FinDER`: Task-specific metadata
- `LocalFinDER`: Custom data loading for local files

### 4. **Extensibility**
- Easy to add new tasks by inheriting from `BaseTask`
- Can override methods for custom behavior (like `load_data()`)

### 5. **Evaluation Standardization**
- All tasks use the same evaluation metrics
- Results are saved in consistent formats

---

## 📝 Summary

**BaseTask.py** provides:
- ✅ Data loading framework (can be overridden)
- ✅ Retrieval orchestration
- ✅ Reranking orchestration  
- ✅ Generation framework (not used in this run)
- ✅ Evaluation metrics (NDCG, MAP, Recall, Precision)
- ✅ Result saving (CSV + JSONL)

**FinDERTask.py** provides:
- ✅ Task-specific metadata (name, dataset path, etc.)
- ✅ Inherits all BaseTask functionality

**main.ipynb** uses:
- ✅ `LocalFinDER` to override data loading for local files
- ✅ Inherited methods: `retrieve()`, `rerank()`, `evaluate()`, `save_results()`
- ✅ All functionality comes from BaseTask inheritance

