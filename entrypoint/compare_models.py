"""
Script to compare GCN vs GraphSAGE models.

This script trains both models with identical hyperparameters and compares:
- Validation loss
- Training time
- Model performance metrics
"""
import sys
import os
import json
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipelines.training_pipeline import train_model
from src.utils import load_config, setup_training_logger


def train_with_config(gnn_type: str, config: dict, output_suffix: str = ""):
    """
    Train a model with specified GNN type.
    
    Args:
        gnn_type: "GCN" or "GraphSAGE"
        config: Configuration dictionary
        output_suffix: Suffix for output files (e.g., "_gcn", "_sage")
        
    Returns:
        Dictionary with training results
    """
    # Update config for this run
    config["model"]["gnn_type"] = gnn_type
    
    # Create separate output directory
    original_save_dir = config["training"]["model_save_dir"]
    run_save_dir = f"{original_save_dir}_{gnn_type.lower()}{output_suffix}"
    config["training"]["model_save_dir"] = run_save_dir
    os.makedirs(run_save_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_training_logger(log_dir=run_save_dir, log_file="training_errors.log")
    
    print("=" * 80)
    print(f"Training {gnn_type} Model")
    print("=" * 80)
    print(f"Output directory: {run_save_dir}")
    print(f"Hidden dims: {config['model']['hidden_dims']}")
    print(f"Dropout rates: {config['model']['dropout_rates']}")
    print(f"Learning rate: {config['training']['lr']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    
    if gnn_type == "GraphSAGE":
        print(f"Aggregation: {config['model'].get('sage_aggr', 'mean')}")
    
    print()
    
    logger.info(f"Starting {gnn_type} training comparison")
    logger.info(f"Output directory: {run_save_dir}")
    
    # Train model
    start_time = time.time()
    try:
        model = train_model(config=config, logger=logger)
        training_time = time.time() - start_time
        
        # Load training history
        history_path = os.path.join(run_save_dir, "training_history.json")
        with open(history_path, "r") as f:
            history = json.load(f)
        
        # Extract metrics
        best_val_loss = min([h["val_loss"] for h in history if "val_loss" in h])
        final_train_loss = history[-1]["train_loss"]
        final_val_loss = history[-1]["val_loss"]
        epochs_trained = len(history)
        
        results = {
            "gnn_type": gnn_type,
            "output_dir": run_save_dir,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "epochs_trained": epochs_trained,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "history": history
        }
        
        logger.info(f"{gnn_type} training completed successfully")
        logger.info(f"Training time: {training_time/60:.2f} minutes")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training {gnn_type}: {str(e)}")
        raise


def compare_results(gcn_results: dict, sage_results: dict):
    """
    Compare training results between GCN and GraphSAGE.
    
    Args:
        gcn_results: Results from GCN training
        sage_results: Results from GraphSAGE training
    """
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print("\n📊 Training Time:")
    print(f"  GCN:       {gcn_results['training_time_minutes']:.2f} minutes")
    print(f"  GraphSAGE: {sage_results['training_time_minutes']:.2f} minutes")
    time_diff = sage_results['training_time_minutes'] - gcn_results['training_time_minutes']
    time_pct = (time_diff / gcn_results['training_time_minutes']) * 100
    print(f"  Difference: {time_diff:+.2f} minutes ({time_pct:+.1f}%)")
    
    print("\n📈 Epochs Trained:")
    print(f"  GCN:       {gcn_results['epochs_trained']}")
    print(f"  GraphSAGE: {sage_results['epochs_trained']}")
    
    print("\n🎯 Best Validation Loss:")
    print(f"  GCN:       {gcn_results['best_val_loss']:.6f}")
    print(f"  GraphSAGE: {sage_results['best_val_loss']:.6f}")
    loss_diff = sage_results['best_val_loss'] - gcn_results['best_val_loss']
    loss_pct = (loss_diff / gcn_results['best_val_loss']) * 100
    print(f"  Difference: {loss_diff:+.6f} ({loss_pct:+.2f}%)")
    
    if loss_diff < 0:
        print(f"  ✓ GraphSAGE is BETTER by {abs(loss_pct):.2f}%")
    elif loss_diff > 0:
        print(f"  ✗ GCN is BETTER by {abs(loss_pct):.2f}%")
    else:
        print(f"  = Performance is EQUAL")
    
    print("\n📉 Final Training Loss:")
    print(f"  GCN:       {gcn_results['final_train_loss']:.6f}")
    print(f"  GraphSAGE: {sage_results['final_train_loss']:.6f}")
    
    print("\n📉 Final Validation Loss:")
    print(f"  GCN:       {gcn_results['final_val_loss']:.6f}")
    print(f"  GraphSAGE: {sage_results['final_val_loss']:.6f}")
    
    print("\n💾 Model Outputs:")
    print(f"  GCN:       {gcn_results['output_dir']}")
    print(f"  GraphSAGE: {sage_results['output_dir']}")
    
    print("\n" + "=" * 80)
    
    # Determine winner
    if loss_diff < -0.001:  # GraphSAGE is significantly better
        print("🏆 WINNER: GraphSAGE")
        print(f"   GraphSAGE achieved {abs(loss_pct):.2f}% lower validation loss")
    elif loss_diff > 0.001:  # GCN is significantly better
        print("🏆 WINNER: GCN")
        print(f"   GCN achieved {abs(loss_pct):.2f}% lower validation loss")
    else:
        print("🤝 RESULT: Tie (performance within 0.1%)")
    
    print("=" * 80)


def save_comparison_results(gcn_results: dict, sage_results: dict, output_path: str):
    """
    Save comparison results to JSON file.
    
    Args:
        gcn_results: Results from GCN training
        sage_results: Results from GraphSAGE training
        output_path: Path to save comparison JSON
    """
    comparison = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gcn": gcn_results,
        "graphsage": sage_results,
        "comparison": {
            "val_loss_difference": sage_results['best_val_loss'] - gcn_results['best_val_loss'],
            "val_loss_improvement_pct": ((sage_results['best_val_loss'] - gcn_results['best_val_loss']) / gcn_results['best_val_loss']) * 100,
            "time_difference_minutes": sage_results['training_time_minutes'] - gcn_results['training_time_minutes'],
            "time_difference_pct": ((sage_results['training_time_minutes'] - gcn_results['training_time_minutes']) / gcn_results['training_time_minutes']) * 100,
            "winner": "GraphSAGE" if sage_results['best_val_loss'] < gcn_results['best_val_loss'] - 0.001 else ("GCN" if gcn_results['best_val_loss'] < sage_results['best_val_loss'] - 0.001 else "Tie")
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison results saved to: {output_path}")


def main():
    """Main comparison function."""
    print("=" * 80)
    print("GCN vs GraphSAGE Model Comparison")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load base config
    config = load_config()
    
    # Train GCN
    gcn_results = train_with_config("GCN", config.copy(), output_suffix="_comparison")
    
    print("\n" + "=" * 80)
    print("GCN training completed, starting GraphSAGE...")
    print("=" * 80 + "\n")
    
    # Train GraphSAGE
    sage_results = train_with_config("GraphSAGE", config.copy(), output_suffix="_comparison")
    
    # Compare results
    compare_results(gcn_results, sage_results)
    
    # Save comparison
    comparison_path = os.path.join(config["training"]["model_save_dir"], "model_comparison.json")
    save_comparison_results(gcn_results, sage_results, comparison_path)
    
    print(f"\n✓ Comparison completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()




