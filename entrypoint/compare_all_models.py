"""
Comprehensive comparison of all GNN approaches:
1. GCN (baseline)
2. GraphSAGE (from scratch)
3. GraphSAGE with Transfer Learning (staged training)

This script analyzes training histories and generates a comparison report.
"""
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_training_history(model_dir: str) -> Optional[Dict]:
    """
    Load training history from a model directory.
    
    Args:
        model_dir: Directory containing training_history.json
        
    Returns:
        Dictionary with training history or None if not found
    """
    history_path = os.path.join(model_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        return None
    
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"Error loading {history_path}: {e}")
        return None


def analyze_model_performance(history: List[Dict], model_name: str) -> Dict:
    """
    Analyze model performance from training history.
    
    Args:
        history: List of epoch results
        model_name: Name of the model
        
    Returns:
        Dictionary with performance metrics
    """
    if not history:
        return {}
    
    # Extract metrics
    train_losses = [h["train_loss"] for h in history if "train_loss" in h]
    val_losses = [h["val_loss"] for h in history if "val_loss" in h]
    epochs = len(history)
    
    best_val_loss = min(val_losses) if val_losses else float('inf')
    best_epoch = val_losses.index(best_val_loss) + 1 if val_losses else 0
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    
    # Calculate convergence (epochs to reach within 5% of best)
    threshold = best_val_loss * 1.05
    convergence_epoch = next((i+1 for i, loss in enumerate(val_losses) if loss <= threshold), epochs)
    
    # Check for overfitting (train loss much lower than val loss)
    if final_train_loss > 0:
        overfit_ratio = final_val_loss / final_train_loss
    else:
        overfit_ratio = 1.0
    
    # Extract metrics if available
    metrics = {}
    if history[-1].get("val_metrics"):
        metrics = history[-1]["val_metrics"]
    
    return {
        "model_name": model_name,
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "convergence_epoch": convergence_epoch,
        "overfit_ratio": overfit_ratio,
        "metrics": metrics,
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def compare_models(results: Dict[str, Dict]):
    """
    Compare multiple models and generate report.
    
    Args:
        results: Dictionary mapping model names to their analysis results
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    
    # Sort by best validation loss
    sorted_models = sorted(results.items(), key=lambda x: x[1]["best_val_loss"])
    
    print("\n📊 RANKING BY VALIDATION LOSS (Best to Worst):")
    print("-" * 80)
    
    for rank, (model_name, result) in enumerate(sorted_models, 1):
        symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"\n{symbol} Rank {rank}: {model_name}")
        print(f"   Best Val Loss: {result['best_val_loss']:.6f} (epoch {result['best_epoch']})")
        print(f"   Final Train Loss: {result['final_train_loss']:.6f}")
        print(f"   Final Val Loss: {result['final_val_loss']:.6f}")
        print(f"   Epochs Trained: {result['epochs_trained']}")
        print(f"   Convergence: Epoch {result['convergence_epoch']}")
        print(f"   Overfit Ratio: {result['overfit_ratio']:.3f} {'(⚠️ overfitting)' if result['overfit_ratio'] > 1.2 else '(✓ good fit)'}")
    
    # Detailed comparison table
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON TABLE")
    print("=" * 80)
    
    # Header
    print(f"\n{'Model':<30} {'Best Val Loss':<15} {'Epochs':<10} {'Convergence':<12} {'Overfit':<10}")
    print("-" * 80)
    
    for model_name, result in sorted_models:
        print(f"{model_name:<30} {result['best_val_loss']:<15.6f} {result['epochs_trained']:<10} "
              f"{result['convergence_epoch']:<12} {result['overfit_ratio']:<10.3f}")
    
    # Performance improvements
    if len(sorted_models) > 1:
        print("\n" + "=" * 80)
        print("PERFORMANCE IMPROVEMENTS")
        print("=" * 80)
        
        baseline_name, baseline_result = sorted_models[-1]  # Worst model
        best_name, best_result = sorted_models[0]  # Best model
        
        improvement = ((baseline_result['best_val_loss'] - best_result['best_val_loss']) 
                      / baseline_result['best_val_loss'] * 100)
        
        print(f"\n📈 {best_name} vs {baseline_name}:")
        print(f"   Validation loss improvement: {improvement:.2f}%")
        print(f"   Absolute difference: {baseline_result['best_val_loss'] - best_result['best_val_loss']:.6f}")
        
        # Compare convergence speed
        conv_improvement = ((baseline_result['convergence_epoch'] - best_result['convergence_epoch'])
                           / baseline_result['convergence_epoch'] * 100)
        print(f"   Convergence speed improvement: {conv_improvement:.1f}% faster")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    best_name, best_result = sorted_models[0]
    
    print(f"\n🎯 Best Model: {best_name}")
    print(f"   Achieves lowest validation loss of {best_result['best_val_loss']:.6f}")
    
    if best_result['overfit_ratio'] > 1.2:
        print("\n⚠️  Warning: Model shows signs of overfitting")
        print("   Consider:")
        print("   - Increasing dropout rates")
        print("   - Adding more regularization")
        print("   - Reducing model capacity")
        print("   - Collecting more training data")
    else:
        print("\n✓ Model shows good generalization (low overfitting)")
    
    if best_result['convergence_epoch'] < best_result['epochs_trained'] * 0.5:
        print(f"\n💡 Model converged quickly (epoch {best_result['convergence_epoch']})")
        print("   Consider reducing total epochs for faster training")
    
    # Architecture recommendations
    print("\n📐 Architecture Insights:")
    
    if any("GraphSAGE" in name for name, _ in sorted_models):
        sage_models = [(n, r) for n, r in sorted_models if "GraphSAGE" in n]
        best_sage = min(sage_models, key=lambda x: x[1]['best_val_loss'])
        
        if best_sage[1]['best_val_loss'] < best_result['best_val_loss'] * 1.01:
            print("   ✓ GraphSAGE performs competitively with other architectures")
            print("   ✓ Consider using GraphSAGE for its scalability and expressiveness")
    
    if any("Transfer" in name for name, _ in sorted_models):
        transfer_models = [(n, r) for n, r in sorted_models if "Transfer" in n]
        best_transfer = min(transfer_models, key=lambda x: x[1]['best_val_loss'])
        baseline_models = [(n, r) for n, r in sorted_models if "Transfer" not in n]
        
        if baseline_models:
            best_baseline = min(baseline_models, key=lambda x: x[1]['best_val_loss'])
            
            if best_transfer[1]['best_val_loss'] < best_baseline[1]['best_val_loss']:
                improvement = ((best_baseline[1]['best_val_loss'] - best_transfer[1]['best_val_loss'])
                             / best_baseline[1]['best_val_loss'] * 100)
                print(f"   ✓ Transfer learning improves performance by {improvement:.2f}%")
                print("   ✓ Staged training strategy is effective")
            else:
                print("   ⚠️  Transfer learning did not outperform from-scratch training")
                print("   → Domain-specific training may be more effective for Clash Royale")
                print("   → Consider using from-scratch GraphSAGE for better results")


def save_comparison_report(results: Dict[str, Dict], output_path: str):
    """
    Save comparison results to JSON file.
    
    Args:
        results: Dictionary of model results
        output_path: Path to save report
    """
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": results,
        "ranking": sorted(results.keys(), key=lambda k: results[k]["best_val_loss"]),
        "best_model": min(results.items(), key=lambda x: x[1]["best_val_loss"])[0]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Comparison report saved to: {output_path}")


def main():
    """Main comparison function."""
    print("=" * 80)
    print("Comprehensive GNN Model Comparison")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define model directories to compare
    model_dirs = {
        "GCN (Baseline)": "models",
        "GraphSAGE": "models_graphsage",
        "GraphSAGE Transfer Learning": "models_transfer"
    }
    
    # Load training histories
    results = {}
    
    for model_name, model_dir in model_dirs.items():
        print(f"Loading {model_name} from {model_dir}...")
        
        if not os.path.exists(model_dir):
            print(f"  ⚠️  Directory not found: {model_dir}")
            print(f"  Skipping {model_name}")
            continue
        
        history = load_training_history(model_dir)
        
        if history is None:
            print(f"  ⚠️  No training history found")
            continue
        
        print(f"  ✓ Loaded {len(history)} epochs")
        
        # Analyze performance
        analysis = analyze_model_performance(history, model_name)
        results[model_name] = analysis
    
    if not results:
        print("\n❌ No model results found to compare.")
        print("\nPlease train models first:")
        print("  1. GCN: python entrypoint/train.py (with gnn_type: 'GCN')")
        print("  2. GraphSAGE: python entrypoint/train.py (with gnn_type: 'GraphSAGE')")
        print("  3. Transfer: python entrypoint/train_transfer_learning.py")
        return
    
    # Compare models
    compare_models(results)
    
    # Save report
    report_path = os.path.join("models", "comprehensive_comparison.json")
    save_comparison_report(results, report_path)
    
    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()




