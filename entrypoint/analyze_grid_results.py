"""
Analyze Grid Search Results.

This script aggregates results from grid search experiments and generates:
- Comparison tables sorted by performance
- Plots showing hyperparameter effects
- Best configuration recommendations
"""
import sys
import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: pandas/matplotlib not available. Plotting disabled.")


def load_all_results(grid_dir: str) -> List[Dict]:
    """
    Load all experiment results from grid search directory.
    
    Args:
        grid_dir: Path to grid search output directory
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Check for aggregated results first
    csv_path = os.path.join(grid_dir, "grid_results.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row["lr"] = float(row["lr"])
                row["best_val_loss"] = float(row["best_val_loss"])
                row["final_top2_acc"] = float(row["final_top2_acc"])
                row["num_epochs_total"] = int(row["num_epochs_total"])
                results.append(row)
        return results
    
    # Otherwise, scan experiment directories
    for exp_dir in Path(grid_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_path = exp_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results.append(json.load(f))
    
    return results


def load_experiment_history(grid_dir: str, experiment_id: str) -> List[Dict]:
    """
    Load training history for a specific experiment.
    
    Args:
        grid_dir: Path to grid search output directory
        experiment_id: Experiment ID
        
    Returns:
        List of epoch history dictionaries
    """
    history_path = os.path.join(grid_dir, experiment_id, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            return json.load(f)
    return []


def print_results_table(results: List[Dict], top_n: int = 10):
    """
    Print formatted results table.
    
    Args:
        results: List of result dictionaries
        top_n: Number of top results to show
    """
    if not results:
        print("No results found.")
        return
    
    # Sort by best val loss
    sorted_results = sorted(results, key=lambda x: x["best_val_loss"])
    
    print("\n" + "=" * 100)
    print("GRID SEARCH RESULTS - Sorted by Validation Loss")
    print("=" * 100)
    
    # Header
    print(f"{'Rank':<5} {'Experiment ID':<45} {'Val Loss':<12} {'Top-2 Acc':<12} {'Epochs':<8}")
    print("-" * 100)
    
    for i, r in enumerate(sorted_results[:top_n]):
        print(f"{i+1:<5} {r['experiment_id']:<45} {r['best_val_loss']:<12.4f} {r['final_top2_acc']:<12.4f} {r['num_epochs_total']:<8}")
    
    if len(sorted_results) > top_n:
        print(f"... and {len(sorted_results) - top_n} more experiments")
    
    print("=" * 100)


def analyze_by_hyperparameter(results: List[Dict]):
    """
    Analyze results grouped by each hyperparameter.
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("ANALYSIS BY HYPERPARAMETER")
    print("=" * 80)
    
    # Group by each hyperparameter
    for param in ["lr", "freeze_strategy", "arch_variant", "reg_variant"]:
        print(f"\n--- {param.upper()} ---")
        
        groups = defaultdict(list)
        for r in results:
            key = r.get(param, "unknown")
            groups[key].append(r["best_val_loss"])
        
        # Compute stats for each group
        stats = []
        for key, losses in groups.items():
            stats.append({
                "value": key,
                "mean": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
                "count": len(losses)
            })
        
        # Sort by mean loss
        stats = sorted(stats, key=lambda x: x["mean"])
        
        print(f"{'Value':<20} {'Mean Loss':<12} {'Min':<12} {'Max':<12} {'Count':<8}")
        print("-" * 64)
        for s in stats:
            print(f"{str(s['value']):<20} {s['mean']:<12.4f} {s['min']:<12.4f} {s['max']:<12.4f} {s['count']:<8}")


def find_best_configuration(results: List[Dict]) -> Dict:
    """
    Find and display the best configuration.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Best result dictionary
    """
    if not results:
        return {}
    
    best = min(results, key=lambda x: x["best_val_loss"])
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Experiment ID: {best['experiment_id']}")
    print(f"Learning Rate: {best['lr']}")
    print(f"Freeze Strategy: {best['freeze_strategy']}")
    print(f"Architecture: {best['arch_variant']}")
    print(f"Regularization: {best['reg_variant']}")
    print("-" * 40)
    print(f"Best Validation Loss: {best['best_val_loss']:.4f}")
    print(f"Final Top-2 Accuracy: {best['final_top2_acc']:.4f}")
    print(f"Total Epochs: {best['num_epochs_total']}")
    print(f"Best Stage: {best.get('best_stage', 'N/A')}")
    print("=" * 80)
    
    return best


def plot_results(results: List[Dict], output_dir: str):
    """
    Generate plots for grid search analysis.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    if not HAS_PLOTTING:
        print("Plotting not available (missing pandas/matplotlib)")
        return
    
    if not results:
        print("No results to plot")
        return
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Grid Search Analysis', fontsize=14, fontweight='bold')
    
    # 1. Val Loss by Learning Rate
    ax1 = axes[0, 0]
    lr_groups = df.groupby('lr')['best_val_loss'].agg(['mean', 'std', 'min'])
    x = range(len(lr_groups))
    ax1.bar(x, lr_groups['mean'], yerr=lr_groups['std'], capsize=5, alpha=0.7)
    ax1.scatter(x, lr_groups['min'], color='red', marker='*', s=100, label='Best', zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{lr:.4f}" for lr in lr_groups.index])
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Val Loss by Learning Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Val Loss by Freeze Strategy
    ax2 = axes[0, 1]
    freeze_groups = df.groupby('freeze_strategy')['best_val_loss'].agg(['mean', 'std', 'min'])
    x = range(len(freeze_groups))
    ax2.bar(x, freeze_groups['mean'], yerr=freeze_groups['std'], capsize=5, alpha=0.7)
    ax2.scatter(x, freeze_groups['min'], color='red', marker='*', s=100, label='Best', zorder=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(freeze_groups.index, rotation=45, ha='right')
    ax2.set_xlabel('Freeze Strategy')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Val Loss by Freeze Strategy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Val Loss by Architecture
    ax3 = axes[1, 0]
    arch_groups = df.groupby('arch_variant')['best_val_loss'].agg(['mean', 'std', 'min'])
    x = range(len(arch_groups))
    ax3.bar(x, arch_groups['mean'], yerr=arch_groups['std'], capsize=5, alpha=0.7)
    ax3.scatter(x, arch_groups['min'], color='red', marker='*', s=100, label='Best', zorder=5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_groups.index)
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Val Loss by Architecture')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Top-2 Accuracy Distribution
    ax4 = axes[1, 1]
    ax4.hist(df['final_top2_acc'], bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(df['final_top2_acc'].max(), color='red', linestyle='--', 
                label=f"Best: {df['final_top2_acc'].max():.4f}")
    ax4.axvline(df['final_top2_acc'].mean(), color='green', linestyle='--',
                label=f"Mean: {df['final_top2_acc'].mean():.4f}")
    ax4.set_xlabel('Top-2 Accuracy')
    ax4.set_ylabel('Count')
    ax4.set_title('Top-2 Accuracy Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "grid_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def plot_learning_curves(results: List[Dict], grid_dir: str, output_dir: str, top_n: int = 5):
    """
    Plot learning curves for top experiments.
    
    Args:
        results: List of result dictionaries
        grid_dir: Grid search directory
        output_dir: Output directory for plots
        top_n: Number of top experiments to plot
    """
    if not HAS_PLOTTING:
        return
    
    if not results:
        return
    
    # Get top experiments
    sorted_results = sorted(results, key=lambda x: x["best_val_loss"])[:top_n]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Learning Curves - Top {top_n} Experiments', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for i, r in enumerate(sorted_results):
        history = load_experiment_history(grid_dir, r["experiment_id"])
        if not history:
            continue
        
        epochs = list(range(1, len(history) + 1))
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        
        label = f"{r['experiment_id'][:30]}... ({r['best_val_loss']:.3f})"
        
        # Train loss
        axes[0].plot(epochs, train_losses, color=colors[i], alpha=0.7, label=label)
        
        # Val loss
        axes[1].plot(epochs, val_losses, color=colors[i], alpha=0.7, label=label)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Learning curves saved to: {plot_path}")
    plt.close()


def generate_report(results: List[Dict], grid_dir: str, output_path: str):
    """
    Generate a markdown report of grid search results.
    
    Args:
        results: List of result dictionaries
        grid_dir: Grid search directory
        output_path: Path to save report
    """
    if not results:
        return
    
    sorted_results = sorted(results, key=lambda x: x["best_val_loss"])
    best = sorted_results[0]
    
    report = f"""# Grid Search Results Report

Generated: {Path(grid_dir).name}

## Summary

- **Total Experiments**: {len(results)}
- **Best Validation Loss**: {best['best_val_loss']:.4f}
- **Best Top-2 Accuracy**: {max(r['final_top2_acc'] for r in results):.4f}

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Experiment ID | {best['experiment_id']} |
| Learning Rate | {best['lr']} |
| Freeze Strategy | {best['freeze_strategy']} |
| Architecture | {best['arch_variant']} |
| Regularization | {best['reg_variant']} |
| Best Val Loss | {best['best_val_loss']:.4f} |
| Final Top-2 Acc | {best['final_top2_acc']:.4f} |

## Top 10 Experiments

| Rank | Experiment | Val Loss | Top-2 Acc |
|------|------------|----------|-----------|
"""
    
    for i, r in enumerate(sorted_results[:10]):
        report += f"| {i+1} | {r['experiment_id']} | {r['best_val_loss']:.4f} | {r['final_top2_acc']:.4f} |\n"
    
    report += """
## Hyperparameter Analysis

### By Learning Rate
"""
    
    lr_groups = defaultdict(list)
    for r in results:
        lr_groups[r['lr']].append(r['best_val_loss'])
    
    report += "| LR | Mean Loss | Best Loss | Count |\n|----|-----------|-----------|---------|\n"
    for lr in sorted(lr_groups.keys()):
        losses = lr_groups[lr]
        report += f"| {lr} | {sum(losses)/len(losses):.4f} | {min(losses):.4f} | {len(losses)} |\n"
    
    report += """
### By Freeze Strategy
"""
    
    freeze_groups = defaultdict(list)
    for r in results:
        freeze_groups[r['freeze_strategy']].append(r['best_val_loss'])
    
    report += "| Strategy | Mean Loss | Best Loss | Count |\n|----------|-----------|-----------|---------|\n"
    for strat in sorted(freeze_groups.keys()):
        losses = freeze_groups[strat]
        report += f"| {strat} | {sum(losses)/len(losses):.4f} | {min(losses):.4f} | {len(losses)} |\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Grid Search Results")
    parser.add_argument("--dir", type=str, default="models_grid",
                       help="Grid search results directory")
    parser.add_argument("--top", type=int, default=10,
                       help="Number of top results to show")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--report", action="store_true",
                       help="Generate markdown report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Grid Search Results Analysis")
    print("=" * 80)
    
    # Load results
    results = load_all_results(args.dir)
    
    if not results:
        print(f"No results found in {args.dir}")
        return
    
    print(f"Loaded {len(results)} experiment results from {args.dir}")
    
    # Print results table
    print_results_table(results, args.top)
    
    # Analyze by hyperparameter
    analyze_by_hyperparameter(results)
    
    # Find best configuration
    best = find_best_configuration(results)
    
    # Generate plots
    if args.plot:
        plot_results(results, args.dir)
        plot_learning_curves(results, args.dir, args.dir)
    
    # Generate report
    if args.report:
        report_path = os.path.join(args.dir, "grid_report.md")
        generate_report(results, args.dir, report_path)


if __name__ == "__main__":
    main()






