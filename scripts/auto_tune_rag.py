import os
import sys
import json
import argparse
import time
import pandas as pd
from pathlib import Path

# Add project root to sys.path to allow relative imports
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from main import ExperimentConfig
from experiments.experiment_runner import run_experiment

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Self-Tuning RAG Hyperparameter Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cuad",
        help="Dataset to tune on (default: cuad)"
    )
    parser.add_argument(
        "--n_records_tune",
        type=int,
        default=50,
        help="Number of records to use for tuning sweeps (default: 50)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_adherence",
        choices=["mean_adherence", "mean_context_relevance", "mean_context_utilization", "mean_completeness"],
        help="Metric to optimize for (default: mean_adherence)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="colab-gpu",
        choices=["local-mock", "local-real", "colab-gpu"],
        help="Execution profile (default: colab-gpu)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default="colab_user",
        help="Username for logging (default: colab_user)"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="reports/cuad_auto_tune_report.md",
        help="Path to save the auto-tuning execution summary report"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume capability and force start a fresh tuning run"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="Override profile default generator and evaluator model (e.g. llama3:8b)"
    )
    return parser.parse_args()

def fetch_score_from_master(run_id: str, metric: str, base_dir: Path) -> float:
    """Reads the master_tracking.csv file and extracts the metric score for the given run_id."""
    time.sleep(0.5)  # Ensure file system flushes have finished
    master_path = base_dir / "master_tracking.csv"
    if not master_path.exists():
        print(f"Warning: master_tracking.csv not found at {master_path}. Returning default score.")
        return 0.0
    try:
        df = pd.read_csv(master_path)
        matching_rows = df[df["run_id"] == run_id]
        if not matching_rows.empty:
            return float(matching_rows.iloc[-1][metric])
    except Exception as e:
        print(f"Error reading metric '{metric}' from master file: {e}")
    return 0.0

def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                if not state.get("is_completed", False):
                    return state
        except Exception as e:
            print(f"Error reading state file: {e}")
    return None

def save_state(state_file: Path, state_data: dict):
    try:
        os.makedirs(state_file.parent, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
    except Exception as e:
        print(f"Error saving state file: {e}")

def main():
    args = parse_args()
    base_dir = Path(__file__).parent.parent.resolve()
    state_file = base_dir / "reports" / f"{args.dataset}_auto_tune_state.json"
    
    # Establish tuning parameters lists based on profile
    if args.profile == "local-mock":
        embedders = ["mock"]
        retrievers = ["dense", "sparse"]
        chunkers = ["recursive", "metadata"]
        chunk_sizes = [256, 512]
        rerankers = ["none", "cross-encoder"]
        generator_model = args.generator if args.generator else "mock"
        evaluator_model = "mock"
    elif args.profile == "local-real":
        embedders = ["minilm"]
        retrievers = ["dense", "sparse", "hybrid"]
        chunkers = ["recursive", "semantic"]
        chunk_sizes = [256, 512]
        rerankers = ["none", "cross-encoder"]
        generator_model = args.generator if args.generator else "mock"
        evaluator_model = "heuristic"
    else:  # colab-gpu
        embedders = ["bge-small", "bge-large", "minilm"]
        retrievers = ["dense", "sparse", "hybrid"]
        chunkers = ["recursive", "semantic"]
        chunk_sizes = [256, 512, 1024]
        rerankers = ["none", "cross-encoder"]
        generator_model = args.generator if args.generator else "hf-small"
        evaluator_model = "llm"

    # Initial Baseline / Default parameters
    locked_embedder = embedders[0]
    locked_retriever = retrievers[0]
    locked_chunker = chunkers[0]
    locked_chunk_size = chunk_sizes[0]
    locked_reranker = rerankers[0]
    
    current_stage = 1
    stage_runs_completed = {}
    tuning_history = []
    
    # Try to resume from state file
    resumed = False
    if not args.no_resume:
        state = load_state(state_file)
        if state:
            print("=" * 70)
            print(f"🔄 RESUMING AUTO-TUNING SESSION FROM STATE FILE: {state_file}")
            print(f"Target Metric: {state.get('metric')}")
            print(f"Resuming at Stage: {state.get('current_stage')}")
            print("=" * 70)
            
            locked_embedder = state.get("locked_embedder", locked_embedder)
            locked_retriever = state.get("locked_retriever", locked_retriever)
            locked_chunker = state.get("locked_chunker", locked_chunker)
            locked_chunk_size = state.get("locked_chunk_size", locked_chunk_size)
            locked_reranker = state.get("locked_reranker", locked_reranker)
            
            current_stage = state.get("current_stage", current_stage)
            stage_runs_completed = state.get("stage_runs_completed", {})
            tuning_history = state.get("tuning_history", [])
            resumed = True

    if not resumed:
        print("=" * 70)
        print(f"🤖 STARTING AUTOMATED RAG SELF-TUNING OPTIMIZER")
        print(f"Tuning metric: {args.metric}")
        print(f"Dataset: {args.dataset}")
        print(f"Tuning slice size: {args.n_records_tune} records")
        print(f"Winner verification: Full Dataset (510 records)")
        print(f"Profile: {args.profile}")
        print("=" * 70)
        
        # Save initial clean state
        initial_state = {
            "dataset": args.dataset,
            "metric": args.metric,
            "n_records_tune": args.n_records_tune,
            "profile": args.profile,
            "locked_embedder": locked_embedder,
            "locked_retriever": locked_retriever,
            "locked_chunker": locked_chunker,
            "locked_chunk_size": locked_chunk_size,
            "locked_reranker": locked_reranker,
            "current_stage": current_stage,
            "stage_runs_completed": stage_runs_completed,
            "tuning_history": tuning_history,
            "is_completed": False
        }
        save_state(state_file, initial_state)

    run_count = 0

    # Helper function to run winner's full verification run
    def run_winner_verification(stage_num: int, winner_val: str) -> float:
        print(f"\n⚡ Running Stage {stage_num} Winner Verification on FULL dataset...")
        run_prefix = f"autotune_s{stage_num}_winner"
        cfg = ExperimentConfig(
            profile=args.profile,
            embedder=locked_embedder,
            retriever=locked_retriever,
            chunker=locked_chunker,
            chunk_size=locked_chunk_size,
            reranker=locked_reranker,
            generator=generator_model,
            evaluator=evaluator_model,
            dataset=args.dataset,
            n_records=None, # None runs the full dataset (510 records)
            username=args.username,
            run_id_prefix=run_prefix
        )
        expected_run_id = f"{run_prefix}_{cfg.chunker}_{cfg.embedder}_{cfg.retriever}"
        print(f"  --> Executing Full Dataset Run (Run ID: {expected_run_id})")
        try:
            run_experiment(cfg)
            full_score = fetch_score_from_master(expected_run_id, args.metric, base_dir)
            print(f"      Full Run Result -> {args.metric}: {full_score:.4f}")
            return full_score
        except Exception as e:
            print(f"      Full run failed for winner {winner_val}: {e}")
            return 0.0

    # Helper function to persist state updates
    def update_and_persist_state(completed_stage: bool = False):
        state_data = {
            "dataset": args.dataset,
            "metric": args.metric,
            "n_records_tune": args.n_records_tune,
            "profile": args.profile,
            "locked_embedder": locked_embedder,
            "locked_retriever": locked_retriever,
            "locked_chunker": locked_chunker,
            "locked_chunk_size": locked_chunk_size,
            "locked_reranker": locked_reranker,
            "current_stage": current_stage,
            "stage_runs_completed": stage_runs_completed,
            "tuning_history": tuning_history,
            "is_completed": False
        }
        save_state(state_file, state_data)

    # ──── STAGE 1: Tune Embedder ─────────────────────────────────────────────
    if len(embedders) > 1:
        if current_stage <= 1:
            print(f"\n🚀 [STAGE 1] Tuning Embedders... (Testing: {embedders})")
            for emb in embedders:
                if emb in stage_runs_completed:
                    print(f"  --> Skip {emb} (Already completed with score {stage_runs_completed[emb]:.4f})")
                    continue
                
                run_count += 1
                run_prefix = f"autotune_s1_run{run_count}"
                cfg = ExperimentConfig(
                    profile=args.profile,
                    embedder=emb,
                    retriever=locked_retriever,
                    chunker=locked_chunker,
                    chunk_size=locked_chunk_size,
                    reranker=locked_reranker,
                    generator=generator_model,
                    evaluator=evaluator_model,
                    dataset=args.dataset,
                    n_records=args.n_records_tune,
                    username=args.username,
                    run_id_prefix=run_prefix
                )
                expected_run_id = f"{run_prefix}_{cfg.chunker}_{cfg.embedder}_{cfg.retriever}"
                print(f"  --> Running embedder sweep: {emb} on {args.n_records_tune} recs (Run ID: {expected_run_id})")
                
                try:
                    run_experiment(cfg)
                    score = fetch_score_from_master(expected_run_id, args.metric, base_dir)
                    stage_runs_completed[emb] = score
                    print(f"      Result -> {args.metric}: {score:.4f}")
                except Exception as e:
                    print(f"      Run failed for {emb}: {e}")
                    stage_runs_completed[emb] = 0.0
                
                # Persist progress after each sweep run
                update_and_persist_state()

            # Select best embedder
            best_emb = max(stage_runs_completed, key=stage_runs_completed.get)
            locked_embedder = best_emb
            print(f"✅ STAGE 1 Winner (Tuning Slice): '{locked_embedder}' (Score: {stage_runs_completed[best_emb]:.4f})")
            
            # Run winner verification on full dataset
            full_score = run_winner_verification(1, locked_embedder)
            
            tuning_history.append({
                "stage": "1. Embedder Tuning",
                "tested_options": str(stage_runs_completed),
                "selected": locked_embedder,
                "tune_score": stage_runs_completed[best_emb],
                "full_score": full_score
            })
            
            # Advance stage and clear temp stage variables
            current_stage = 2
            stage_runs_completed = {}
            update_and_persist_state()
    else:
        print(f"\n⏭️  Skipping Stage 1 (Only one option: '{locked_embedder}')")

    # ──── STAGE 2: Tune Retriever ────────────────────────────────────────────
    if len(retrievers) > 1:
        if current_stage <= 2:
            print(f"\n🚀 [STAGE 2] Tuning Retrievers... (Testing: {retrievers})")
            for ret in retrievers:
                if ret in stage_runs_completed:
                    print(f"  --> Skip {ret} (Already completed with score {stage_runs_completed[ret]:.4f})")
                    continue
                
                run_count += 1
                run_prefix = f"autotune_s2_run{run_count}"
                cfg = ExperimentConfig(
                    profile=args.profile,
                    embedder=locked_embedder,
                    retriever=ret,
                    chunker=locked_chunker,
                    chunk_size=locked_chunk_size,
                    reranker=locked_reranker,
                    generator=generator_model,
                    evaluator=evaluator_model,
                    dataset=args.dataset,
                    n_records=args.n_records_tune,
                    username=args.username,
                    run_id_prefix=run_prefix
                )
                expected_run_id = f"{run_prefix}_{cfg.chunker}_{cfg.embedder}_{cfg.retriever}"
                print(f"  --> Running retriever sweep: {ret} on {args.n_records_tune} recs (Run ID: {expected_run_id})")
                
                try:
                    run_experiment(cfg)
                    score = fetch_score_from_master(expected_run_id, args.metric, base_dir)
                    stage_runs_completed[ret] = score
                    print(f"      Result -> {args.metric}: {score:.4f}")
                except Exception as e:
                    print(f"      Run failed for {ret}: {e}")
                    stage_runs_completed[ret] = 0.0
                
                # Persist progress
                update_and_persist_state()

            best_ret = max(stage_runs_completed, key=stage_runs_completed.get)
            locked_retriever = best_ret
            print(f"✅ STAGE 2 Winner (Tuning Slice): '{locked_retriever}' (Score: {stage_runs_completed[best_ret]:.4f})")
            
            # Run winner verification on full dataset
            full_score = run_winner_verification(2, locked_retriever)
            
            tuning_history.append({
                "stage": "2. Retriever Tuning",
                "tested_options": str(stage_runs_completed),
                "selected": locked_retriever,
                "tune_score": stage_runs_completed[best_ret],
                "full_score": full_score
            })
            
            # Advance stage
            current_stage = 3
            stage_runs_completed = {}
            update_and_persist_state()
    else:
        print(f"\n⏭️  Skipping Stage 2 (Only one option: '{locked_retriever}')")

    # ──── STAGE 3: Tune Chunker ──────────────────────────────────────────────
    if len(chunkers) > 1:
        if current_stage <= 3:
            print(f"\n🚀 [STAGE 3] Tuning Chunkers... (Testing: {chunkers})")
            for chk in chunkers:
                if chk in stage_runs_completed:
                    print(f"  --> Skip {chk} (Already completed with score {stage_runs_completed[chk]:.4f})")
                    continue
                
                run_count += 1
                run_prefix = f"autotune_s3_run{run_count}"
                cfg = ExperimentConfig(
                    profile=args.profile,
                    embedder=locked_embedder,
                    retriever=locked_retriever,
                    chunker=chk,
                    chunk_size=locked_chunk_size,
                    reranker=locked_reranker,
                    generator=generator_model,
                    evaluator=evaluator_model,
                    dataset=args.dataset,
                    n_records=args.n_records_tune,
                    username=args.username,
                    run_id_prefix=run_prefix
                )
                expected_run_id = f"{run_prefix}_{cfg.chunker}_{cfg.embedder}_{cfg.retriever}"
                print(f"  --> Running chunker sweep: {chk} on {args.n_records_tune} recs (Run ID: {expected_run_id})")
                
                try:
                    run_experiment(cfg)
                    score = fetch_score_from_master(expected_run_id, args.metric, base_dir)
                    stage_runs_completed[chk] = score
                    print(f"      Result -> {args.metric}: {score:.4f}")
                except Exception as e:
                    print(f"      Run failed for {chk}: {e}")
                    stage_runs_completed[chk] = 0.0
                
                # Persist progress
                update_and_persist_state()

            best_chk = max(stage_runs_completed, key=stage_runs_completed.get)
            locked_chunker = best_chk
            print(f"✅ STAGE 3 Winner (Tuning Slice): '{locked_chunker}' (Score: {stage_runs_completed[best_chk]:.4f})")
            
            # Run winner verification on full dataset
            full_score = run_winner_verification(3, locked_chunker)
            
            tuning_history.append({
                "stage": "3. Chunker Tuning",
                "tested_options": str(stage_runs_completed),
                "selected": locked_chunker,
                "tune_score": stage_runs_completed[best_chk],
                "full_score": full_score
            })
            
            # Advance stage
            current_stage = 4
            stage_runs_completed = {}
            update_and_persist_state()
    else:
        print(f"\n⏭️  Skipping Stage 3 (Only one option: '{locked_chunker}')")

    # ──── STAGE 4: Tune Chunk Size ───────────────────────────────────────────
    if len(chunk_sizes) > 1:
        if current_stage <= 4:
            print(f"\n🚀 [STAGE 4] Tuning Chunk Sizes... (Testing: {chunk_sizes})")
            for sz in chunk_sizes:
                # Key must be string inside json dict keys
                str_sz = str(sz)
                if str_sz in stage_runs_completed or sz in stage_runs_completed:
                    existing_score = stage_runs_completed.get(str_sz) or stage_runs_completed.get(sz)
                    print(f"  --> Skip {sz} (Already completed with score {existing_score:.4f})")
                    continue
                
                run_count += 1
                run_prefix = f"autotune_s4_run{run_count}"
                cfg = ExperimentConfig(
                    profile=args.profile,
                    embedder=locked_embedder,
                    retriever=locked_retriever,
                    chunker=locked_chunker,
                    chunk_size=sz,
                    reranker=locked_reranker,
                    generator=generator_model,
                    evaluator=evaluator_model,
                    dataset=args.dataset,
                    n_records=args.n_records_tune,
                    username=args.username,
                    run_id_prefix=run_prefix
                )
                expected_run_id = f"{run_prefix}_{cfg.chunker}_{cfg.embedder}_{cfg.retriever}"
                print(f"  --> Running chunk size sweep: {sz} on {args.n_records_tune} recs (Run ID: {expected_run_id})")
                
                try:
                    run_experiment(cfg)
                    score = fetch_score_from_master(expected_run_id, args.metric, base_dir)
                    stage_runs_completed[sz] = score
                    print(f"      Result -> {args.metric}: {score:.4f}")
                except Exception as e:
                    print(f"      Run failed for chunk size {sz}: {e}")
                    stage_runs_completed[sz] = 0.0
                
                # Persist progress
                update_and_persist_state()

            # Normalize keys to integers
            normalized_scores = {int(k): v for k, v in stage_runs_completed.items()}
            best_sz = max(normalized_scores, key=normalized_scores.get)
            locked_chunk_size = best_sz
            print(f"✅ STAGE 4 Winner (Tuning Slice): '{locked_chunk_size}' (Score: {normalized_scores[best_sz]:.4f})")
            
            # Run winner verification on full dataset
            full_score = run_winner_verification(4, locked_chunk_size)
            
            tuning_history.append({
                "stage": "4. Chunk Size Tuning",
                "tested_options": str(normalized_scores),
                "selected": locked_chunk_size,
                "tune_score": normalized_scores[best_sz],
                "full_score": full_score
            })
            
            # Advance stage
            current_stage = 5
            stage_runs_completed = {}
            update_and_persist_state()
    else:
        print(f"\n⏭️  Skipping Stage 4 (Only one option: '{locked_chunk_size}')")

    # ──── STAGE 5: Tune Reranker ─────────────────────────────────────────────
    if len(rerankers) > 1:
        if current_stage <= 5:
            print(f"\n🚀 [STAGE 5] Tuning Rerankers... (Testing: {rerankers})")
            for rerank in rerankers:
                if rerank in stage_runs_completed:
                    print(f"  --> Skip {rerank} (Already completed with score {stage_runs_completed[rerank]:.4f})")
                    continue
                
                run_count += 1
                run_prefix = f"autotune_s5_run{run_count}"
                cfg = ExperimentConfig(
                    profile=args.profile,
                    embedder=locked_embedder,
                    retriever=locked_retriever,
                    chunker=locked_chunker,
                    chunk_size=locked_chunk_size,
                    reranker=rerank,
                    generator=generator_model,
                    evaluator=evaluator_model,
                    dataset=args.dataset,
                    n_records=args.n_records_tune,
                    username=args.username,
                    run_id_prefix=run_prefix
                )
                expected_run_id = f"{run_prefix}_{cfg.chunker}_{cfg.embedder}_{cfg.retriever}"
                print(f"  --> Running reranker sweep: {rerank} on {args.n_records_tune} recs (Run ID: {expected_run_id})")
                
                try:
                    run_experiment(cfg)
                    score = fetch_score_from_master(expected_run_id, args.metric, base_dir)
                    stage_runs_completed[rerank] = score
                    print(f"      Result -> {args.metric}: {score:.4f}")
                except Exception as e:
                    print(f"      Run failed for reranker {rerank}: {e}")
                    stage_runs_completed[rerank] = 0.0
                
                # Persist progress
                update_and_persist_state()

            best_rerank = max(stage_runs_completed, key=stage_runs_completed.get)
            locked_reranker = best_rerank
            print(f"✅ STAGE 5 Winner (Tuning Slice): '{locked_reranker}' (Score: {stage_runs_completed[best_rerank]:.4f})")
            
            # Run winner verification on full dataset
            full_score = run_winner_verification(5, locked_reranker)
            
            tuning_history.append({
                "stage": "5. Reranker Tuning",
                "tested_options": str(stage_runs_completed),
                "selected": locked_reranker,
                "tune_score": stage_runs_completed[best_rerank],
                "full_score": full_score
            })
            
            # End optimization
            current_stage = 6
            stage_runs_completed = {}
            # Save final completed state
            final_state = {
                "locked_embedder": locked_embedder,
                "locked_retriever": locked_retriever,
                "locked_chunker": locked_chunker,
                "locked_chunk_size": locked_chunk_size,
                "locked_reranker": locked_reranker,
                "current_stage": current_stage,
                "stage_runs_completed": {},
                "tuning_history": tuning_history,
                "is_completed": True
            }
            save_state(state_file, final_state)
    else:
        print(f"\n⏭️  Skipping Stage 5 (Only one option: '{locked_reranker}')")

    # ── Output Tuning Report ──────────────────────────────────────────────────
    print(f"\nSaving Auto-Tuning Execution report to {args.output_report}...")
    
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    
    report_lines = [
        f"# 🤖 RAG Pipeline Self-Tuning Optimization Report",
        f"**Optimizer Mode:** Sequential Greedy Coordinate Descent",
        f"**Optimization Metric Target:** `{args.metric}`",
        f"**Tuning Slice:** `{args.n_records_tune} records`",
        f"**Locked Run Evaluator:** `Full Dataset (510 records)`",
        f"**Tuning Dataset:** `{args.dataset}`",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 🏆 Winning Configuration",
        "The coordinate search locked in the following optimal parameters for RAG execution:",
        "",
        f"- **Embedder**: `{locked_embedder}`",
        f"- **Retriever**: `{locked_retriever}` (Sparse = BM25)",
        f"- **Chunker**: `{locked_chunker}`",
        f"- **Chunk Size**: `{locked_chunk_size}` characters",
        f"- **Reranker**: `{locked_reranker}` (Cross-Encoder: `ms-marco-MiniLM-L-6-v2`)",
        "",
        "---",
        "",
        "## 📈 Tuning Iterations History",
        "The following details the sequential tuning process stage by stage (scores representing both tuning slice and locked full dataset runs):",
        ""
    ]
    
    for h in tuning_history:
        report_lines.extend([
            f"### Stage: {h['stage']}",
            f"- **Options Tested (Tuning Slice)**: `{h['tested_options']}`",
            f"- **Selected Value**: `{h['selected']}`",
            f"- **Slice Score**: **{h['tune_score']:.4f}**",
            f"- **Locked Full Dataset Score**: **{h['full_score']:.4f}**",
            ""
        ])
        
    report_lines.append(f"\n*Auto-tuning completed successfully. Total runs evaluated: {run_count} sweeps + {len(tuning_history)} full dataset verification runs.*")
    
    with open(args.output_report, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(report_lines))
        
    print("Auto-tuning run completed and report generated!")

if __name__ == "__main__":
    main()
