# 🤖 RAG Pipeline Self-Tuning Optimization Report
**Optimizer Mode:** Sequential Greedy Coordinate Descent
**Optimization Metric Target:** `mean_adherence`
**Tuning Slice:** `50 records`
**Locked Run Evaluator:** `Full Dataset (510 records)`
**Tuning Dataset:** `cuad`
**Date:** 2026-08-03 04:55:11

---

## 🏆 Winning Configuration
The coordinate search locked in the following optimal parameters for RAG execution:

- **Embedder**: `minilm`
- **Retriever**: `dense` (Sparse = BM25)
- **Chunker**: `recursive`
- **Chunk Size**: `256` characters
- **Reranker**: `none` (Cross-Encoder: `ms-marco-MiniLM-L-6-v2`)

---

## 📈 Tuning Iterations History
The following details the sequential tuning process stage by stage (scores representing both tuning slice and locked full dataset runs):

### Stage: 1. Embedder Tuning
- **Options Tested (Tuning Slice)**: `{'bge-small': 0.98, 'bge-large': 0.964, 'minilm': 0.992}`
- **Selected Value**: `minilm`
- **Slice Score**: **0.9920**
- **Locked Full Dataset Score**: **0.9753**

### Stage: 2. Retriever Tuning
- **Options Tested (Tuning Slice)**: `{'dense': 0.992, 'sparse': 0.968, 'hybrid': 0.984}`
- **Selected Value**: `dense`
- **Slice Score**: **0.9920**
- **Locked Full Dataset Score**: **0.9753**

### Stage: 3. Chunker Tuning
- **Options Tested (Tuning Slice)**: `{'recursive': 0.992, 'semantic': 0.964}`
- **Selected Value**: `recursive`
- **Slice Score**: **0.9920**
- **Locked Full Dataset Score**: **0.9753**

### Stage: 4. Chunk Size Tuning
- **Options Tested (Tuning Slice)**: `{256: 0.992, 512: 0.9519999999999998, 1024: 0.84}`
- **Selected Value**: `256`
- **Slice Score**: **0.9920**
- **Locked Full Dataset Score**: **0.9753**

### Stage: 5. Reranker Tuning
- **Options Tested (Tuning Slice)**: `{'none': 0.992, 'cross-encoder': 0.976}`
- **Selected Value**: `none`
- **Slice Score**: **0.9920**
- **Locked Full Dataset Score**: **0.9753**


*Auto-tuning completed successfully. Total runs evaluated: 10 sweeps + 5 full dataset verification runs.*