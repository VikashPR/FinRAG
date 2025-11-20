# Corpus to Vector Conversion: Dictionary vs Vectors

## 🎯 Quick Answer

**The corpus is stored as a dictionary, but converted to vectors ON-THE-FLY during retrieval. The vectors are NOT stored - they're computed, used, and then discarded.**

---

## 📊 Complete Flow

### **Step 1: Initial Storage (Dictionary Format)**

```python
# In main.ipynb
corpus = {
    doc["_id"]: {"title": doc.get("title", ""), "text": doc.get("text", "")}
    for doc in load_jsonl(CORPUS_PATH)
}
```

**Format:** Dictionary
```python
{
    "ADBE20230004": {
        "title": "ADBE OVERVIEW",
        "text": "Adobe is a global technology company..."
    },
    "MSFT20230014": {
        "title": "...",
        "text": "..."
    },
    # ... 13,867 documents
}
```

**Storage:** In memory as Python dictionary
**No vectors yet!** ✅

---

### **Step 2: Retrieval Call**

```python
retrieval_result = finder_task.retrieve(retriever=retriever, top_k=200)
```

**Input:** Dictionary format corpus
**What happens:** Vectors are computed on-the-fly

---

### **Step 3: Vector Conversion (Inside `DenseRetrieval.retrieve()`)**

Looking at `dense.py` lines 114-228:

```python
def retrieve(self, corpus, queries, top_k=200, ...):
    # corpus is STILL a dictionary at this point
    # corpus: Dict[str, Dict[Literal["title", "text"], str]]
    
    # Step 3a: Encode queries ONCE (stored temporarily)
    query_embeddings = self.model.encode_queries(query_texts, ...)
    # query_embeddings: torch.Tensor shape [216, 1024]
    # Stored in memory temporarily
    
    # Step 3b: Convert corpus dictionary to list
    corpus_list = [corpus[cid] for cid in sorted_corpus_ids]
    # corpus_list: List[Dict] = [
    #     {"title": "...", "text": "..."},
    #     {"title": "...", "text": "..."},
    #     ...
    # ]
    
    # Step 3c: Process corpus in CHUNKS (to save memory)
    for batch_num, start_idx in enumerate(range(0, len(corpus), self.corpus_chunk_size)):
        # Process 50,000 documents at a time
        
        # Convert chunk to vectors ON-THE-FLY
        sub_corpus_embeddings = self.model.encode_corpus(
            corpus_list[start_idx:end_idx],  # Dictionary format input
            batch_size=self.batch_size
        )
        # sub_corpus_embeddings: torch.Tensor shape [chunk_size, 1024]
        # Created in memory, used immediately, then can be garbage collected
        
        # Step 3d: Compute similarity scores
        cos_scores = self.score_functions[score_function](
            query_embeddings,      # [216, 1024]
            sub_corpus_embeddings  # [chunk_size, 1024]
        )
        # cos_scores: torch.Tensor shape [216, chunk_size]
        
        # Step 3e: Get top-k and store SCORES (not vectors!)
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(...)
        
        # Step 3f: Store results as SCORES in dictionary
        # Vectors are discarded after this chunk
        for query_itr in range(len(query_embeddings)):
            # Store: query_id -> {doc_id: score}
            self.results[qid][corpus_id] = score
    
    # After loop: sub_corpus_embeddings is discarded
    # Only self.results (scores) are kept
    
    return self.results  # Returns scores, NOT vectors
```

---

## 🔍 Key Points

### **1. Corpus Format Throughout Process**

| Stage | Format | Location |
|-------|--------|----------|
| **Initial Load** | Dictionary | `corpus = {...}` |
| **Passed to retrieve()** | Dictionary | `corpus: Dict[str, Dict[...]]` |
| **During Encoding** | Dictionary → Vectors (temporary) | `sub_corpus_embeddings` |
| **After Retrieval** | Dictionary (unchanged) | `finder_task.corpus` |
| **Results** | Dictionary of scores | `retrieval_result` |

### **2. Vector Lifecycle**

```
Dictionary Corpus
    ↓
[Chunk 1: 50k docs] → Encode → Vectors → Compute Similarity → Scores → Discard Vectors
[Chunk 2: 50k docs] → Encode → Vectors → Compute Similarity → Scores → Discard Vectors
[Chunk 3: 37k docs] → Encode → Vectors → Compute Similarity → Scores → Discard Vectors
    ↓
Final Results: Dictionary of scores (NO VECTORS STORED)
```

### **3. Memory Management**

**Stored in Memory:**
- ✅ Original corpus dictionary (text format)
- ✅ Query embeddings (216 × 1024 = ~0.9 MB)
- ✅ Current chunk embeddings (50k × 1024 = ~200 MB per chunk)
- ✅ Final results (scores only, very small)

**NOT Stored:**
- ❌ Full corpus embeddings (would be 13,867 × 1024 = ~55 MB)
- ❌ Previous chunk embeddings (discarded after use)

**Why?** Memory efficiency! Only one chunk of vectors exists at a time.

---

## 📝 Code Evidence

### **Evidence 1: Dictionary Input**

```python
# dense.py line 116
def retrieve(
    self,
    corpus: Dict[str, Dict[Literal["title", "text"], str]],  # Dictionary!
    queries: Dict[str, str],
    ...
):
```

### **Evidence 2: Temporary Vector Creation**

```python
# dense.py lines 185-187
# Encode chunk of corpus
sub_corpus_embeddings = self.model.encode_corpus(
    corpus_list[start_idx:end_idx], batch_size=self.batch_size, **kwargs
)
# sub_corpus_embeddings is a temporary tensor
# It's used immediately and can be garbage collected
```

### **Evidence 3: Only Scores Stored**

```python
# dense.py lines 224-226
for qid in result_heaps:
    for score, corpus_id in result_heaps[qid]:
        self.results[qid][corpus_id] = score  # Only score, not vector!
```

### **Evidence 4: No Vector Storage in Class**

```python
# dense.py lines 106-112
def __init__(self, ...):
    self.model: Encoder = model
    self.batch_size: int = batch_size
    self.score_functions: Dict = ...
    self.corpus_chunk_size: int = corpus_chunk_size
    self.results: Dict = {}  # Only stores scores, NOT vectors!
    # No self.corpus_embeddings or similar!
```

---

## 🎯 Summary

| Question | Answer |
|----------|--------|
| **Is corpus stored as dictionary?** | ✅ Yes, always |
| **Is corpus converted to vectors?** | ✅ Yes, but only temporarily |
| **When are vectors created?** | During `retrieve()` call |
| **Are vectors stored?** | ❌ No, they're computed and discarded |
| **What is stored?** | Dictionary format corpus + final scores |
| **Why not store vectors?** | Memory efficiency (would need ~55 MB) |
| **Can vectors be reused?** | ❌ No, recomputed each retrieval call |

---

## 💡 Why This Design?

### **Advantages:**
1. **Memory Efficient**: Only one chunk of vectors in memory at a time
2. **Flexible**: Can change encoder model without re-indexing
3. **Simple**: No vector database needed
4. **Fresh**: Always uses latest encoder model

### **Disadvantages:**
1. **Slower**: Must recompute vectors for each retrieval call
2. **No Caching**: Can't reuse embeddings across runs

### **Trade-off:**
This is a **research/evaluation** setup, not a production system. For production, you'd typically:
- Pre-compute and store corpus vectors in a vector database
- Cache query vectors
- Use approximate nearest neighbor search (FAISS, Pinecone, etc.)

---

## 🔄 Complete Data Flow Diagram

```
corpus.jsonl (text files)
    ↓
load_jsonl() → Parse JSON
    ↓
corpus = {"doc_id": {"title": "...", "text": "..."}}  [DICTIONARY]
    ↓
finder_task.corpus = corpus  [STILL DICTIONARY]
    ↓
retriever.retrieve(corpus, queries)  [DICTIONARY INPUT]
    ↓
┌─────────────────────────────────────┐
│ FOR EACH CHUNK (50k docs):          │
│   corpus_list[chunk]                │ [DICTIONARY]
│       ↓                              │
│   encode_corpus()                    │
│       ↓                              │
│   sub_corpus_embeddings              │ [VECTORS - TEMPORARY]
│       ↓                              │
│   Compute similarity                 │
│       ↓                              │
│   Extract top-k scores               │
│       ↓                              │
│   Store scores in results           │ [SCORES ONLY]
│       ↓                              │
│   Discard vectors (garbage collect) │ [VECTORS DELETED]
└─────────────────────────────────────┘
    ↓
retrieval_result = {"query_id": {"doc_id": score, ...}}  [SCORES ONLY]
    ↓
finder_task.corpus  [STILL DICTIONARY - UNCHANGED]
```

---

## ✅ Final Answer

**The corpus is stored as a dictionary throughout the entire process. Vectors are created on-the-fly during retrieval, used for similarity computation, and then discarded. Only the similarity scores are stored in the results.**

