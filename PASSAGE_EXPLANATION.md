# What is Passed as "Passage"?

## 📄 Yes, it comes from `corpus.jsonl`!

The `doc_prompt="passage: "` parameter refers to the **documents from `corpus.jsonl`**, but specifically the **combined title + text** of each document.

---

## 🔍 Step-by-Step Flow

### **Step 1: Loading from corpus.jsonl**

Each line in `corpus.jsonl` contains a JSON object like:

```json
{
  "_id": "ADBE20230004",
  "title": "ADBE OVERVIEW",
  "text": "Adobe is a global technology company with a mission to change the world through personalized digital experiences. For over four decades, Adobe's innovations have transformed how individuals, teams, businesses, enterprises, institutions, and governments engage and interact across all types of media..."
}
```

### **Step 2: Converting to Corpus Dictionary**

In `main.ipynb`, the corpus is loaded as:

```python
corpus = {
    doc["_id"]: {"title": doc.get("title", ""), "text": doc.get("text", "")}
    for doc in load_jsonl(CORPUS_PATH)
}
```

**Result structure:**
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
    # ... 13,867 documents total
}
```

### **Step 3: Encoding with "passage: " Prefix**

When `DenseRetrieval.retrieve()` is called, it calls `encoder.encode_corpus()`.

Looking at `sent_encoder.py` lines 38-67:

```python
def encode_corpus(self, corpus, batch_size=8, **kwargs):
    # corpus is a List[Dict] like:
    # [{"title": "ADBE OVERVIEW", "text": "Adobe is a global..."}, ...]
    
    sentences = [
        (doc["title"] + " " + doc["text"]).strip()  # Combine title + text
        if "title" in doc
        else doc["text"].strip()
        for doc in corpus
    ]
    
    # Add the "passage: " prefix
    if self.doc_prompt is not None:
        sentences = [self.doc_prompt + s for s in sentences]
    
    return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)
```

### **Step 4: What Actually Gets Encoded**

For the example document above, the encoder receives:

```
"passage: ADBE OVERVIEW Adobe is a global technology company with a mission to change the world through personalized digital experiences. For over four decades, Adobe's innovations have transformed how individuals, teams, businesses, enterprises, institutions, and governments engage and interact across all types of media..."
```

**Format:** `"passage: " + title + " " + text`

---

## 📊 Complete Example

### **Input (corpus.jsonl):**
```json
{
  "_id": "ADBE20230004",
  "title": "ADBE OVERVIEW",
  "text": "Adobe is a global technology company..."
}
```

### **After Loading:**
```python
corpus["ADBE20230004"] = {
    "title": "ADBE OVERVIEW",
    "text": "Adobe is a global technology company..."
}
```

### **What Gets Encoded:**
```
"passage: ADBE OVERVIEW Adobe is a global technology company..."
```

### **What Gets Encoded for Queries:**
```
"query: What is Adobe's mission?"
```

---

## 🎯 Why "passage: " Prefix?

The `e5-large-v2` model was trained with specific prefixes:
- **Queries**: `"query: "` prefix
- **Documents**: `"passage: "` prefix

This is required for optimal performance. See the [E5 paper](https://arxiv.org/abs/2212.03533) for details.

---

## 📝 Summary

| Question | Answer |
|----------|--------|
| **Is "passage" from corpus.jsonl?** | ✅ Yes! |
| **What exactly?** | The **combined `title + text`** from each document |
| **Format?** | `"passage: " + title + " " + text` |
| **How many?** | All 13,867 documents from `corpus.jsonl` |
| **When?** | During `retriever.retrieve()` → `encoder.encode_corpus()` |

---

## 🔄 Full Encoding Flow

```
corpus.jsonl (13,867 lines)
    ↓
load_jsonl() → Parse JSON objects
    ↓
corpus dict: {"doc_id": {"title": "...", "text": "..."}}
    ↓
DenseRetrieval.retrieve()
    ↓
encoder.encode_corpus(corpus_list)
    ↓
For each doc: "passage: " + title + " " + text
    ↓
e5-large-v2.encode() → 1024-dim vectors
    ↓
Compute cosine similarity with query vectors
```

---

## 💡 Key Points

1. **"passage" = document text** from `corpus.jsonl`
2. **Each document** = `title + " " + text` combined
3. **Prefix is required** for e5-large-v2 model performance
4. **All 13,867 documents** are encoded with this format
5. **Same format** is used for both retrieval and reranking

