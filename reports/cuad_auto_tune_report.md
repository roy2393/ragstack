# 🤖 RAG Pipeline Self-Tuning Optimization Report
**Optimizer Mode:** Sequential Greedy Coordinate Descent
**Optimization Metric Target:** `mean_adherence`
**Tuning Slice:** `2 records`
**Locked Run Evaluator:** `Full Dataset (510 records)`
**Tuning Dataset:** `cuad`
**Date:** 2026-08-03 01:20:31

---

## 🏆 Winning Configuration
The coordinate search locked in the following optimal parameters for RAG execution:

- **Embedder**: `mock`
- **Retriever**: `dense` (Sparse = BM25)
- **Chunker**: `recursive`
- **Chunk Size**: `256` characters
- **Reranker**: `none` (Cross-Encoder: `ms-marco-MiniLM-L-6-v2`)

---

## 📈 Tuning Iterations History
The following details the sequential tuning process stage by stage (scores representing both tuning slice and locked full dataset runs):

### Stage: 2. Retriever Tuning
- **Options Tested (Tuning Slice)**: `{'dense': 0.5, 'sparse': 0.5}`
- **Selected Value**: `dense`
- **Slice Score**: **0.5000**
- **Locked Full Dataset Score**: **0.5000**

### Stage: 3. Chunker Tuning
- **Options Tested (Tuning Slice)**: `{'recursive': 0.5, 'metadata': 0.5}`
- **Selected Value**: `recursive`
- **Slice Score**: **0.5000**
- **Locked Full Dataset Score**: **0.5000**

### Stage: 4. Chunk Size Tuning
- **Options Tested (Tuning Slice)**: `{256: 0.5, 512: 0.5}`
- **Selected Value**: `256`
- **Slice Score**: **0.5000**
- **Locked Full Dataset Score**: **0.5000**

### Stage: 5. Reranker Tuning
- **Options Tested (Tuning Slice)**: `{'none': 0.5, 'cross-encoder': 0.5}`
- **Selected Value**: `none`
- **Slice Score**: **0.5000**
- **Locked Full Dataset Score**: **0.5000**


*Auto-tuning completed successfully. Total runs evaluated: 6 sweeps + 4 full dataset verification runs.*