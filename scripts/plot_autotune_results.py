import os
import matplotlib.pyplot as plt
import numpy as np

def generate_charts():
    # Define styles for premium visual aesthetic
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Data Definitions from Colab Run
    stages_data = [
        {
            "title": "Stage 1: Embedders",
            "options": ["bge-small", "bge-large", "minilm"],
            "scores": [0.980, 0.964, 0.992],
            "winner": "minilm"
        },
        {
            "title": "Stage 2: Retrievers",
            "options": ["dense", "sparse", "hybrid"],
            "scores": [0.992, 0.968, 0.984],
            "winner": "dense"
        },
        {
            "title": "Stage 3: Chunkers",
            "options": ["recursive", "semantic"],
            "scores": [0.992, 0.964],
            "winner": "recursive"
        },
        {
            "title": "Stage 4: Chunk Sizes",
            "options": ["256", "512", "1024"],
            "scores": [0.992, 0.952, 0.840],
            "winner": "256"
        },
        {
            "title": "Stage 5: Rerankers",
            "options": ["none", "cross-encoder"],
            "scores": [0.992, 0.976],
            "winner": "none"
        }
    ]

    # --- Chart 1: Option Sweeps Comparison (How winners were selected) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Colors for options (Teal for winner, Cool Gray for others)
    color_winner = '#0d9488'
    color_other = '#9ca3af'
    
    for i, stage in enumerate(stages_data):
        ax = axes[i]
        options = stage["options"]
        scores = [s * 100 for s in stage["scores"]]  # Convert to %
        winner = stage["winner"]
        
        # Determine colors for each bar
        colors = [color_winner if opt == winner else color_other for opt in options]
        
        bars = ax.bar(options, scores, color=colors, width=0.5, edgecolor='#4b5563', linewidth=1)
        ax.set_title(stage["title"], fontsize=12, fontweight='bold', pad=12, color='#1f2937')
        ax.set_ylabel("Adherence Score (%)", fontsize=10, color='#4b5563')
        ax.set_ylim(min(scores) - 5, 102)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
            
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    # Hide the 6th subplot space (since we only have 5 stages)
    axes[5].axis('off')
    
    plt.suptitle("RAG Auto-Tuning: Parameter Sweep Comparison (50-Record Slice)", fontsize=16, fontweight='bold', color='#111827', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Ensure directory exists and save
    os.makedirs("reports", exist_ok=True)
    comp_path = "reports/cuad_autotune_comparison.png"
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated comparison chart: {comp_path}")

    # --- Chart 2: Locked Full Dataset Progression Trend ---
    stages_prog = ["Stage 1\n(Embedder)", "Stage 2\n(Retriever)", "Stage 3\n(Chunker)", "Stage 4\n(Chunk Size)", "Stage 5\n(Reranker)"]
    full_dataset_scores = [97.53, 97.53, 97.53, 97.53, 97.53]
    slice_scores = [99.20, 99.20, 99.20, 99.20, 99.20]
    
    plt.figure(figsize=(10, 6))
    
    # Plot progression lines
    plt.plot(stages_prog, full_dataset_scores, marker='o', color='#0f766e', linewidth=2.5, markersize=8, label="Full Dataset (510 records) - Locked Winner")
    plt.plot(stages_prog, slice_scores, marker='s', linestyle='--', color='#0d9488', linewidth=1.5, markersize=6, label="Tuning Slice (50 records)")
    
    plt.title("RAG Pipeline Adherence Score Progression", fontsize=14, fontweight='bold', pad=15, color='#111827')
    plt.xlabel("Optimization Stage", fontsize=11, fontweight='semibold', labelpad=10, color='#374151')
    plt.ylabel("Target Metric: Adherence Score (%)", fontsize=11, fontweight='semibold', labelpad=10, color='#374151')
    plt.ylim(90, 102)
    
    # Annotate values
    for x, y in enumerate(full_dataset_scores):
        plt.text(x, y - 1.5, f"{y:.2f}%", ha='center', va='top', color='#0f766e', fontsize=10, fontweight='bold')
    for x, y in enumerate(slice_scores):
        plt.text(x, y + 1.0, f"{y:.2f}%", ha='center', va='bottom', color='#0d9488', fontsize=9, fontweight='semibold')
        
    plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='#e5e7eb')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    prog_path = "reports/cuad_autotune_progression.png"
    plt.savefig(prog_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated progression chart: {prog_path}")

if __name__ == '__main__':
    generate_charts()
