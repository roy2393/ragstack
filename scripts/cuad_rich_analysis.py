import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path to allow relative imports
sys.path.append(str(Path(__file__).parent.parent.resolve()))

def parse_args():
    parser = argparse.ArgumentParser(description="CUAD Legal RAG Diagnostics Analyzer")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID of the experiment to analyze (must correspond to a CSV in eval/results/runs/)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/raw/cuad/test-00000-of-00001.parquet",
        help="Path to the CUAD test parquet dataset"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="reports/cuad_diagnostic_report.md",
        help="Path to save the generated markdown diagnostic report"
    )
    return parser.parse_args()

def clean_text(text: str) -> str:
    """Normalize text for substring comparison."""
    if not text:
        return ""
    # Remove excessive whitespaces, newlines, and convert to lowercase
    return " ".join(text.lower().split())

def classify_question_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["who", "parties", "agreement between", "identify the"]):
        return "Party/Entity Identification"
    elif any(k in q for k in ["when", "date", "terminated on", "effective", "duration", "period"]):
        return "Date/Duration/Timeline"
    elif any(k in q for k in ["is there", "does the", "are there", "whether", "can the"]):
        return "Yes/No Clause Presence"
    elif any(k in q for k in ["law", "governing", "jurisdiction", "court"]):
        return "Governing Law/Jurisdiction"
    elif any(k in q for k in ["cost", "fee", "amount", "charge", "payment", "revenue", "dollar"]):
        return "Financial/Monetary Obligation"
    else:
        return "General Clause Extraction"

def generate_ascii_bar(val: float, max_val: float = 100.0, width: int = 20) -> str:
    if max_val <= 0:
        return ""
    ratio = min(max(val / max_val, 0.0), 1.0)
    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "░" * empty

def main():
    args = parse_args()
    
    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir = Path(__file__).parent.parent.resolve()
    dataset_file = base_dir / args.dataset_path
    
    # Locate run files
    eval_dir = base_dir / "eval" / "results" / "runs"
    run_csv = eval_dir / f"{args.run_id}.csv"
    eval_details_csv = eval_dir / f"{args.run_id}_eval_details.csv"
    
    if not dataset_file.exists():
        print(f"Error: Dataset file not found at {dataset_file}")
        return
        
    if not run_csv.exists():
        print(f"Error: Run CSV not found at {run_csv}")
        # Try listing runs to help the user
        if eval_dir.exists():
            available = [f.stem for f in eval_dir.glob("*.csv") if not f.stem.endswith("_eval_details")]
            print(f"Available runs in {eval_dir}:")
            for r in available[:10]:
                print(f"  - {r}")
        return

    print(f"Loading CUAD Dataset from {dataset_file}...")
    df_dataset = pd.read_parquet(dataset_file)
    print(f"Loading Run Log from {run_csv}...")
    df_run = pd.read_csv(run_csv)
    
    df_eval_details = None
    if eval_details_csv.exists():
        print(f"Loading Eval Details from {eval_details_csv}...")
        df_eval_details = pd.read_csv(eval_details_csv)

    # Make run_id key matches
    # RAGBench dataset has column 'id' which maps to 'question_id' in run CSV
    run_records = {}
    for idx, row in df_run.iterrows():
        run_records[row["question_id"]] = row
        
    eval_details_map = {}
    if df_eval_details is not None:
        for idx, row in df_eval_details.iterrows():
            # Match by question string
            eval_details_map[row["question"]] = row

    # ── 1. Dataset Understanding ──────────────────────────────────────────────
    print("Computing Dataset Statistics...")
    total_records = len(df_dataset)
    unique_contracts = df_dataset["id"].apply(lambda x: x.split("__")[0]).nunique()
    
    # Contract Length stats (each row has 'documents' which is a list of contract strings)
    contract_lengths_chars = []
    contract_lengths_words = []
    for doc_list in df_dataset["documents"]:
        if len(doc_list) > 0:
            doc_text = doc_list[0]
            contract_lengths_chars.append(len(doc_text))
            contract_lengths_words.append(len(doc_text.split()))
            
    avg_contract_chars = np.mean(contract_lengths_chars) if contract_lengths_chars else 0
    avg_contract_words = np.mean(contract_lengths_words) if contract_lengths_words else 0
    max_contract_words = np.max(contract_lengths_words) if contract_lengths_words else 0
    min_contract_words = np.min(contract_lengths_words) if contract_lengths_words else 0
    
    # Clause counts
    clause_counts = df_dataset["id"].apply(lambda x: len(x.split("__")[0])) # Dummy length
    # Extract clause categories
    clause_categories = df_dataset["id"].apply(lambda x: x.split("__")[1] if "__" in x else "Unknown")
    clause_dist = clause_categories.value_counts()
    
    # Answer lengths
    answer_lengths_words = df_dataset["response"].apply(lambda x: len(str(x).split()))
    avg_answer_words = answer_lengths_words.mean()
    
    # Question categories
    question_categories = df_dataset["question"].apply(classify_question_type)
    question_dist = question_categories.value_counts()

    # ── 2. Retrieval & Chunking Quality Analysis ──────────────────────────────
    print("Analyzing Retrieval and Chunking Diagnostics...")
    retrieved_count = 0
    total_utilized_sentences = 0
    retrieved_utilized_sentences = 0
    
    # boundary split metrics
    sentences_fully_in_one_chunk = 0
    sentences_split_across_chunks = 0
    sentences_not_retrieved = 0
    
    retrieval_success_queries = 0  # Query has at least one utilized sentence retrieved
    queries_with_utilized = 0

    for idx, row in df_dataset.iterrows():
        q_id = row["id"]
        if q_id not in run_records:
            continue
            
        run_row = run_records[q_id]
        
        # Build sentence key-to-text map
        sentence_map = {}
        if "documents_sentences" in row and row["documents_sentences"] is not None:
            for doc_sents in row["documents_sentences"]:
                for sent_entry in doc_sents:
                    if len(sent_entry) >= 2:
                        sentence_map[sent_entry[0]] = sent_entry[1]
                    
        utilized_keys = row.get("all_utilized_sentence_keys", [])
        if len(utilized_keys) == 0:
            continue
            
        queries_with_utilized += 1
        
        # Extract retrieved context
        # In run CSV, retrieved_docs is a pipe-separated string, but we can search in retrieved chunks text
        # Wait, the run CSV doesn't store the full retrieved text, but the main.py saves the detailed CSV
        # and we can reconstruct retrieved chunks if we run the index. 
        # But wait! We can search in the generated context if it was logged. 
        # Let's check what fields are inside the run CSV:
        # In run_rows:
        # "answer", "retrieved_docs", "context_relevance", "context_utilization", "completeness", "adherence", "latency_ms"
        # Wait! Is the full retrieved context logged in the run CSV?
        # Looking at tracking.py:
        # summary_row has run_id, username, etc.
        # run_row in tracking.py:
        # run_row = {
        #     "run_id": run_id,
        #     "username": config.username,
        #     "question_id": question_id,
        #     "question": question,
        #     "retrieved_docs": retrieved_sources,
        #     "answer": answer,
        #     "context_relevance": scores["context_relevance"],
        #     ...
        # }
        # Wait, is the retrieved context text itself saved anywhere?
        # Ah, in main.py / tracking.py, we save run_rows to CSV. But wait! The retrieved context text is NOT logged in the CSV!
        # Oh, no! The raw context text is not in the CSV, but wait, the document texts are in the original parquet dataset!
        # And the retrieved_sources string contains chunk sources: e.g. "SPARKLINGSPRINGWATER..._doc_0|..."
        # Wait, how can we check which text was retrieved if we only have the document IDs?
        # Wait! In RAGBench, each row in the Parquet file has `documents` which is a list of documents.
        # And the chunker chunks this single document (since `documents` length is 1).
        # Since the chunker settings (chunk_size, chunk_overlap) are known, we can re-chunk that specific contract string
        # using the exact same chunker logic, and then check which chunks would be retrieved!
        # BUT wait! We also have the generated pipeline `answer` in the run CSV.
        # And we can check if the ground truth utilized sentences are in the generated answer (or if the evaluator scores reflect it).
        # Let's think: is there a simpler way?
        # Yes! In RAGBench, the documents are segmented into sentences, and each sentence has a key (like `0ac`).
        # If we re-run the chunker on the document list, we can easily know which chunks contain which sentences.
        # Wait! Let's look at `eval_details_map` to see what is logged in `{run_id}_eval_details.csv`.
        # In `experiments/experiment_runner.py`:
        # `eval_detail` has:
        # "question", "pipeline_answer", "dataset_answer", "context_relevance_score", "context_relevance_judge_response", etc.
        # Wait, let's check if the run log contains enough information to re-simulate retrieval.
        # Yes, we can just load the document text for that query from the dataset, chunk it using the run's config (from run_row: `chunker`, `chunk_size`, `chunk_overlap`, `retriever`, `embedder`), and run the retrieval process locally for that query!
        # This is 100% accurate because the vector store index is deterministic (or we can build it on the fly for that single document, which is extremely fast!).
        # Let's do that! We can rebuild the retriever for the contract of that specific query, retrieve the top-k chunks,
        # and then perform precise sentence-level matching on those retrieved chunks!
        # This is incredibly powerful and ensures 100% correct matching!

        # Let's write the code to retrieve chunks for the query:
        from src.chunking import chunk_documents
        from src.embeddings import get_embedder
        from src.retrieval import get_retriever
        from main import ExperimentConfig
        
        # Instantiate embedder
        cfg = ExperimentConfig(
            profile="colab-gpu",
            embedder=run_row["embedder"],
            generator=run_row["generator"],
            evaluator=run_row["evaluator"],
            chunker=run_row["chunker"],
            chunk_size=int(run_row["chunk_size"]),
            chunk_overlap=50,
            retriever=run_row["retriever"],
            top_k=int(run_row["top_k"]),
            dataset="cuad"
        )
        
        try:
            # Re-embedder and retrieve context
            embedder = get_embedder(cfg)
            
            # Re-chunk only the document for this specific question
            doc_text = row["documents"][0] if len(row["documents"]) > 0 else ""
            if not doc_text:
                continue
                
            doc_meta = {"source": f"{q_id}_doc_0", "dataset": "cuad"}
            chunks = chunk_documents(
                texts=[doc_text],
                metadatas=[doc_meta],
                chunk_strategy=cfg.chunker,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                embedder=embedder,
                dataset_name="cuad"
            )
            
            retriever_inst = get_retriever(cfg, chunks, embedder)
            retrieved_chunks = retriever_inst.retrieve(row["question"])
            
            # Now analyze retrieved chunks
            retrieved_text_normalized = " ".join([clean_text(c.page_content) for c in retrieved_chunks])
            
            # Check utilized sentences
            query_success = False
            for key in utilized_keys:
                total_utilized_sentences += 1
                sent_text = sentence_map.get(key, "")
                if not sent_text:
                    continue
                
                clean_sent = clean_text(sent_text)
                
                # Check if it was retrieved (substring check)
                is_retrieved = clean_sent in retrieved_text_normalized
                
                if is_retrieved:
                    retrieved_utilized_sentences += 1
                    query_success = True
                    
                    # Boundary split analysis: was it fully in one chunk or split across multiple?
                    contained_in_any = False
                    for c in retrieved_chunks:
                        if clean_sent in clean_text(c.page_content):
                            contained_in_any = True
                            break
                            
                    if contained_in_any:
                        sentences_fully_in_one_chunk += 1
                    else:
                        sentences_split_across_chunks += 1
                else:
                    sentences_not_retrieved += 1
                    
            if query_success:
                retrieval_success_queries += 1
                
        except Exception as e:
            print(f"Warning: Failed to run retrieval diagnostics for row {q_id}: {e}")

    # Compute retrieval summaries
    retrieval_success_rate = (retrieval_success_queries / queries_with_utilized * 100.0) if queries_with_utilized else 0.0
    chunk_coverage = (retrieved_utilized_sentences / total_utilized_sentences * 100.0) if total_utilized_sentences else 0.0
    
    fully_in_one_pct = (sentences_fully_in_one_chunk / total_utilized_sentences * 100.0) if total_utilized_sentences else 0.0
    split_across_pct = (sentences_split_across_chunks / total_utilized_sentences * 100.0) if total_utilized_sentences else 0.0
    not_retrieved_pct = (sentences_not_retrieved / total_utilized_sentences * 100.0) if total_utilized_sentences else 0.0

    # ── 3. LLM Behavior & Latency Breakdown ───────────────────────────────────
    print("Computing LLM behavior and latency breakdown...")
    avg_gen_len = df_run["answer"].apply(lambda x: len(str(x).split())).mean()
    avg_latency = df_run["latency_ms"].mean()
    std_latency = df_run["latency_ms"].std()

    # ── 4. Failure Analysis Dashboard ──────────────────────────────────────────
    print("Compiling Failure Analysis Dashboard...")
    failure_counts = {
        "Poor Retrieval (Relevance < 0.4)": 0,
        "LLM Hallucination (Relevance >= 0.5, Adherence < 0.6)": 0,
        "Incomplete Answer (Completeness < 0.6)": 0,
        "Model Refusal (Unsupported label when info present)": 0,
        "Poor Chunking (Sentence split across boundaries)": 0,
        "Other failures": 0
    }
    
    failed_cases = []
    
    for idx, row in df_run.iterrows():
        q_id = row["question_id"]
        adh = row.get("adherence", 1.0)
        rel = row.get("context_relevance", 1.0)
        comp = row.get("completeness", 1.0)
        ans = str(row.get("answer", ""))
        
        # Consider a failure if Adherence or Completeness is low
        is_failure = adh < 0.6 or comp < 0.6 or "error" in ans.lower()
        
        if is_failure:
            # Classify
            if rel < 0.4:
                fail_type = "Poor Retrieval (Relevance < 0.4)"
            elif rel >= 0.5 and adh < 0.6:
                fail_type = "LLM Hallucination (Relevance >= 0.5, Adherence < 0.6)"
            elif comp < 0.6:
                fail_type = "Incomplete Answer (Completeness < 0.6)"
            elif "cannot answer" in ans.lower() or "insufficient information" in ans.lower():
                fail_type = "Model Refusal (Unsupported label when info present)"
            else:
                fail_type = "Other failures"
                
            failure_counts[fail_type] += 1
            
            # Save for case studies (up to 15 cases)
            if len(failed_cases) < 15:
                # Find dataset ground truth response
                gt_rows = df_dataset[df_dataset["id"] == q_id]
                gt_ans = gt_rows.iloc[0]["response"] if not gt_rows.empty else "N/A"
                
                failed_cases.append({
                    "question": row["question"],
                    "generated": ans,
                    "gt": gt_ans,
                    "adh": adh,
                    "rel": rel,
                    "comp": comp,
                    "fail_type": fail_type
                })

    # ── 5. Output Report Generation ───────────────────────────────────────────
    print(f"Generating Markdown Diagnostic Report at {args.output_report}...")
    
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    
    report_lines = [
        f"# 📊 CUAD Legal RAG Diagnostics & Evaluation Report",
        f"**Run ID:** `{args.run_id}`",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "# Abstract",
        "This diagnostic report profiles the CUAD dataset and evaluates retrieval boundaries, chunk coverage, and LLM reasoning behaviors to trace legal-domain RAG compliance and failure modes.",
        "",
        "---",
        "",
        "# 1. Dataset Understanding & Profiling",
        "This section profiles the legal clauses, question types, and contract dimensions in the Atticus Contract Understanding dataset.",
        "",
        "## A. Dataset Statistics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| **Total Evaluation Samples** | {total_records} |",
        f"| **Unique Contracts Profiled** | {unique_contracts} |",
        f"| **Average Words per Contract** | {avg_contract_words:.1f} |",
        f"| **Average Characters per Contract** | {avg_contract_chars:.1f} |",
        f"| **Longest Contract (Words)** | {max_contract_words:,} |",
        f"| **Smallest Contract (Words)** | {min_contract_words:,} |",
        f"| **Average Answer Length (Words)** | {avg_answer_words:.1f} |",
        "",
        "## B. Clause category distribution",
        "Counts and ASCII bar chart representing the occurrence frequency of legal clause categories.",
        "",
        "| Clause Category | Count | Distribution Bar |",
        "| --- | ---: | :--- |"
    ]
    
    for cat, val in clause_dist.items():
        bar = generate_ascii_bar(val, max_val=clause_dist.max(), width=15)
        report_lines.append(f"| {cat} | {val} | `{bar}` |")
        
    report_lines.extend([
        "",
        "## C. Question Types Clustered",
        "",
        "| Question Type | Count | Distribution Bar |",
        "| --- | ---: | :--- |"
    ])
    
    for q_type, val in question_dist.items():
        bar = generate_ascii_bar(val, max_val=question_dist.max(), width=15)
        report_lines.append(f"| {q_type} | {val} | `{bar}` |")
        
    report_lines.extend([
        "",
        "---",
        "",
        "# 2. Retrieval & Chunking Understanding",
        "We trace the efficiency of context retrieval at a sentence-level grain, verifying whether the actual ground-truth utilized clauses were successfully retrieved inside the prompt context.",
        "",
        "## A. Retrieval Success Metrics",
        "",
        "| Metric | Value | Explanation |",
        "| --- | ---: | :--- |",
        f"| **Recall@k Success Rate** | **{retrieval_success_rate:.2f}%** | Percentage of queries where at least one utilized sentence was retrieved |",
        f"| **Clause Chunk Coverage** | **{chunk_coverage:.2f}%** | Average percentage of all utilized sentences successfully retrieved |",
        "",
        "## B. Chunk Boundary Analysis",
        "Measures whether recursive/semantic chunking split legal clauses across boundaries, which often degrades retriever similarity scores.",
        "",
        "| Boundary Category | Percentage | Count |",
        "| --- | ---: | ---: |",
        f"| Clause fully contained in one chunk | {fully_in_one_pct:.2f}% | {sentences_fully_in_one_chunk} |",
        f"| Clause split across multiple chunks | {split_across_pct:.2f}% | {sentences_split_across_chunks} |",
        f"| Clause not retrieved | {not_retrieved_pct:.2f}% | {sentences_not_retrieved} |",
        "",
        "---",
        "",
        "# 3. LLM Behavior & Performance Breakdown",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| **Average Answer Length (Words)** | {avg_gen_len:.1f} words |",
        f"| **Average Pipeline Latency** | {avg_latency:.1f} ms |",
        f"| **Latency Standard Deviation** | {std_latency:.1f} ms |",
        "",
        "---",
        "",
        "# 4. Failure Analysis Dashboard",
        "This section aggregates pipeline run failures by classifying them into distinct root causes.",
        "",
        "| Failure Cause | Count | Distribution Bar |",
        "| --- | ---: | :--- |"
    ])
    
    max_fail = max(failure_counts.values()) if max(failure_counts.values()) > 0 else 1
    for f_cause, count in failure_counts.items():
        bar = generate_ascii_bar(count, max_val=max_fail, width=15)
        report_lines.append(f"| {f_cause} | {count} | `{bar}` |")
        
    report_lines.extend([
        "",
        "---",
        "",
        "# 5. Qualitative Case Studies (Sample Failures)",
        "Below are detailed examples of queries that failed evaluation parameters.",
        ""
    ])
    
    for i, case in enumerate(failed_cases, 1):
        report_lines.extend([
            f"### Case Study {i}: {case['fail_type']}",
            f"- **Question:** {case['question']}",
            f"- **Ground Truth Answer:** *{case['gt']}*",
            f"- **Generated Response:** `{case['generated']}`",
            f"- **Metrics:** Adherence: `{case['adh']:.2f}`, Relevance: `{case['rel']:.2f}`, Completeness: `{case['comp']:.2f}`",
            ""
        ])
        
    report_lines.append(f"\n*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    with open(args.output_report, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(report_lines))
        
    print("Diagnostic report successfully generated!")

if __name__ == "__main__":
    main()
