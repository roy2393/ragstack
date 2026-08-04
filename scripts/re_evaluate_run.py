import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from config import ExperimentConfig, DATA_DIR, EVAL_DIR
from src.chunking import chunk_documents
from src.embeddings import get_embedder
from src.retrieval import get_retriever
from src.generation import get_generator
from src.evaluation import get_evaluator
from src.tracking import log_experiment_run

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate historical runs using a constant 14B LLM judge.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a single detailed run CSV to evaluate.")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to a directory containing detailed run CSVs to batch evaluate.")
    parser.add_argument("--judge_model", type=str, default="qwen2.5:14b", help="Model to use as the constant judge.")
    return parser.parse_args()

def re_evaluate_single_file(file_path: Path, judge_model: str):
    print(f"\n========================================================")
    print(f"Re-evaluating run file: {file_path.name}")
    print(f"Constant Judge Model: {judge_model}")
    print(f"========================================================")

    # 1. Load the run CSV
    df_run = pd.read_csv(file_path)
    
    # Validate expected columns exist
    required_cols = ["question", "answer", "embedder", "chunker", "chunk_size", "retriever"]
    missing = [c for c in required_cols if c not in df_run.columns]
    if missing:
        print(f"Skipping {file_path.name} due to missing columns: {missing}")
        return

    # Extract configuration parameters from the first row
    first_row = df_run.iloc[0]
    original_run_id = first_row.get("run_id", file_path.stem)
    dataset = first_row.get("dataset", "cuad")
    embedder_name = first_row.get("embedder", "minilm")
    chunker_name = first_row.get("chunker", "recursive")
    chunk_size = int(first_row.get("chunk_size", 256))
    retriever_name = first_row.get("retriever", "dense")
    top_k = int(first_row.get("top_k", 3))

    # Clean judge model suffix for naming
    judge_suffix = judge_model.replace(":", "")

    # Define new run_id
    new_run_id = f"{original_run_id}_judged_{judge_suffix}"

    print(f"Configuration: Dataset={dataset}, Embedder={embedder_name}, Chunker={chunker_name}, "
          f"ChunkSize={chunk_size}, Retriever={retriever_name}, TopK={top_k}")

    # 2. Locate and load raw dataset
    dataset_dir = DATA_DIR / "raw" / dataset
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}. Skipping.")
        return

    parquet_files = glob.glob(str(dataset_dir / "*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {dataset_dir}. Skipping.")
        return

    # Select correct dataset split (matching original logic)
    test_files = [f for f in parquet_files if "test" in os.path.basename(f)]
    val_files = [f for f in parquet_files if "validation" in os.path.basename(f) or "val" in os.path.basename(f)]
    train_files = [f for f in parquet_files if "train" in os.path.basename(f)]

    file_to_load = test_files[0] if test_files else (val_files[0] if val_files else parquet_files[0])
    print(f"Loading raw dataset from {os.path.basename(file_to_load)}...")
    df_raw = pd.read_parquet(file_to_load)

    # 3. Rebuild Index & Retriever
    temp_config = ExperimentConfig(
        profile="colab-gpu",
        dataset=dataset,
        n_records=None, # build full corpus index
        embedder=embedder_name,
        chunker=chunker_name,
        chunk_size=chunk_size,
        retriever=retriever_name,
        top_k=top_k,
        generator=judge_model,
        evaluator="llm"
    )

    embedder = get_embedder(temp_config)
    
    print("Extracting and deduplicating document corpus...")
    all_docs = []
    seen_docs = set()
    doc_metadata = []
    for idx, row in df_raw.iterrows():
        row_docs = row.get("documents", [])
        if isinstance(row_docs, str):
            row_docs = [row_docs]
        row_id = row.get("id", f"q_{idx}")
        for doc_idx, doc_text in enumerate(row_docs):
            if not doc_text:
                continue
            if doc_text not in seen_docs:
                seen_docs.add(doc_text)
                all_docs.append(doc_text)
                doc_metadata.append({
                    "source": f"{row_id}_doc_{doc_idx}",
                    "dataset": dataset
                })

    print(f"Found {len(all_docs)} unique raw documents. Re-chunking...")
    chunks = chunk_documents(
        texts=all_docs,
        metadatas=doc_metadata,
        chunk_strategy=chunker_name,
        chunk_size=chunk_size,
        chunk_overlap=50,
        embedder=embedder,
        dataset_name=dataset
    )

    retriever = get_retriever(temp_config, chunks, embedder)

    # 4. Initialize 14B Constant Judge Evaluator
    print(f"Initializing constant judge evaluator model: {judge_model}...")
    judge_generator = get_generator(temp_config)
    evaluator = get_evaluator(temp_config, judge_generator)

    # 5. Evaluate Loop
    run_rows = []
    eval_details = []

    print(f"Evaluating {len(df_run)} records with judge model...")
    for idx, row in df_run.iterrows():
        question = row["question"]
        answer = row["answer"]
        question_id = row.get("question_id", f"q_{idx}")

        print(f"[{idx+1}/{len(df_run)}] Judging: '{question[:60]}...'")

        # Deterministic Context Retrieval
        try:
            retrieved_chunks = retriever.retrieve(question)
            context = "\n\n".join([c.page_content for c in retrieved_chunks])
            retrieved_sources = "|".join([c.metadata.get("source", "unknown") for c in retrieved_chunks])
        except Exception as e:
            print(f"Error during retrieval: {e}")
            context = ""
            retrieved_sources = "error"

        # Evaluate answer using the 14B judge
        try:
            scores = evaluator.score(question, context, answer)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            scores = {
                "context_relevance": 0.0,
                "context_utilization": 0.0,
                "completeness": 0.0,
                "adherence": 0.0
            }

        # Detailed Judge Response Formatting
        details = getattr(evaluator, "last_details", {})
        eval_detail = {
            "model": f"{first_row.get('generator', 'unknown')}_judged_{judge_suffix}",
            "question": question,
            "pipeline_answer": answer,
            "dataset_answer": row.get("dataset_answer", "N/A"),
            "context_relevance_score": scores["context_relevance"],
            "context_relevance_judge_response": details.get("context_relevance", {}).get("response", "N/A"),
            "context_utilization_score": scores["context_utilization"],
            "context_utilization_judge_response": details.get("context_utilization", {}).get("response", "N/A"),
            "completeness_score": scores["completeness"],
            "completeness_judge_response": details.get("completeness", {}).get("response", "N/A"),
            "adherence_score": scores["adherence"],
            "adherence_judge_response": details.get("adherence", {}).get("response", "N/A")
        }
        eval_details.append(eval_detail)

        # Output detailed row matching standard structure
        run_row = {
            "run_id": new_run_id,
            "username": row.get("username", "colab_user"),
            "question_id": question_id,
            "question": question,
            "retrieved_docs": retrieved_sources,
            "answer": answer,
            "context_relevance": scores["context_relevance"],
            "context_utilization": scores["context_utilization"],
            "completeness": scores["completeness"],
            "adherence": scores["adherence"],
            "latency_ms": row.get("latency_ms", 0),
            "embedder": embedder_name,
            "generator": f"{first_row.get('generator', 'unknown')}",
            "evaluator": f"llm_judge_{judge_suffix}",
            "chunker": chunker_name,
            "retriever": retriever_name,
            "chunk_size": chunk_size,
            "top_k": top_k
        }
        run_rows.append(run_row)

    # 6. Compute Aggregate Metrics
    mean_metrics = {
        "mean_context_relevance": float(np.mean([r["context_relevance"] for r in run_rows])),
        "mean_context_utilization": float(np.mean([r["context_utilization"] for r in run_rows])),
        "mean_completeness": float(np.mean([r["completeness"] for r in run_rows])),
        "mean_adherence": float(np.mean([r["adherence"] for r in run_rows])),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in run_rows]))
    }

    print("\n--- Re-Evaluation Summary Results ---")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")

    # Override config params for standard tracking logger
    log_config = ExperimentConfig(
        profile="colab-gpu",
        dataset=dataset,
        n_records=len(df_run),
        embedder=embedder_name,
        chunker=chunker_name,
        chunk_size=chunk_size,
        retriever=retriever_name,
        generator=f"{first_row.get('generator', 'unknown')}",
        evaluator=f"judge_{judge_suffix}"
    )

    log_experiment_run(
        config=log_config,
        run_rows=run_rows,
        mean_metrics=mean_metrics,
        eval_details=eval_details
    )
    print(f"Successfully re-evaluated and saved: {new_run_id}")

def main():
    args = parse_args()

    if args.input_file:
        file_path = Path(args.input_file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        re_evaluate_single_file(file_path, args.judge_model)

    elif args.input_dir:
        dir_path = Path(args.input_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)

        # Get all CSV files in directory
        all_csvs = glob.glob(str(dir_path / "*.csv"))
        
        # Filter files to process
        target_csvs = []
        for csv_path in all_csvs:
            name = os.path.basename(csv_path)
            # Skip judge details, summaries, or already evaluated files
            if "_eval_details" in name or "master_tracking" in name or "_judged_" in name:
                continue
            target_csvs.append(Path(csv_path))

        if not target_csvs:
            print(f"No valid run CSV files found in {dir_path}")
            return

        print(f"Found {len(target_csvs)} run CSV files to process.")
        for idx, csv_file in enumerate(target_csvs):
            print(f"\nProcessing file {idx+1}/{len(target_csvs)}...")
            try:
                re_evaluate_single_file(csv_file, args.judge_model)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
    else:
        print("Error: You must specify either --input_file or --input_dir.")
        sys.exit(1)

if __name__ == '__main__':
    main()
