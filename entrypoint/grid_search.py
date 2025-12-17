"""
Grid Search for Hyperparameter Optimization.

This script systematically tests combinations of:
- Learning rates: [0.001, 0.003, 0.005, 0.01]
- Freezing strategies: freeze_all, freeze_partial, skip_stage3, adapter_only
- Architecture variants: baseline, no_finetune, deeper
- Regularization: standard, heavy

Results are saved to models_grid/ with complete checkpoints for resuming.
"""
import sys
import os
import json
import csv
import copy
import torch
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from itertools import product

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.pretrained_sage import CardRecommendationSAGEWithTransfer
from src.pipelines.training_pipeline import (
    load_training_data,
    CardDataset,
    create_dataloader,
    train_epoch,
    validate,
    save_training_history
)
from src.utils import load_config, get_device, ensure_dir, setup_training_logger


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

LEARNING_RATES = [0.01, 0.005]

FREEZE_STRATEGIES = {
    "skip_stage3": {
        "description": "Skip full fine-tuning stage (prevents overfitting)",
        "stages": ["adapter", "partial"],
        "stage_epochs": {"adapter": 3, "partial": 10}
    },
    "full_stages": {
        "description": "All 3 stages",
        "stages": ["adapter", "partial", "full"],
        "stage_epochs": {"adapter": 3, "partial": 7, "full": 5}
    }
}

ARCHITECTURE_VARIANTS = {
    "small": {
        "description": "Small model [256, 128] → [64] (164K params) - BEST",
        "finetune_dims": [64],
        "hidden_dims": [256, 128]
    },
    "medium": {
        "description": "Medium model [384, 192] → [96]",
        "finetune_dims": [96],
        "hidden_dims": [384, 192]
    },
    "large": {
        "description": "Large model [512, 256, 128] → [128, 64] (400K params)",
        "finetune_dims": [128, 64],
        "hidden_dims": [512, 256, 128]
    }
}

REGULARIZATION_VARIANTS = {
    "standard": {
        "description": "Standard dropout",
        "dropout_rates": [0.4, 0.3, 0.2, 0.1],
        "weight_decay": 0.01
    },
    "heavy": {
        "description": "Heavy regularization",
        "dropout_rates": [0.5, 0.4, 0.3, 0.2],
        "weight_decay": 0.02
    }
}


def create_experiment_id(lr: float, freeze: str, arch: str, reg: str) -> str:
    """Create unique experiment ID."""
    lr_str = f"lr{lr}".replace(".", "p")
    return f"{lr_str}_{freeze}_{arch}_{reg}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    stage: str,
    best_val_loss: float,
    config: Dict,
    history: List[Dict],
    checkpoint_path: str
):
    """
    Save complete checkpoint for resuming training.
    
    Saves:
    - model_state_dict: Full model weights
    - optimizer_state_dict: Optimizer state
    - scheduler_state_dict: LR scheduler state
    - epoch: Current epoch
    - stage: Current training stage
    - best_val_loss: Best validation loss
    - config: Full experiment config
    - training_history: All metrics
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "stage": stage,
        "best_val_loss": best_val_loss,
        "config": config,
        "training_history": history,
        "timestamp": datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device) -> Tuple:
    """
    Load checkpoint for resuming training.
    
    Returns:
        Tuple of (model, optimizer_state, scheduler_state, epoch, stage, best_val_loss, config, history)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return (
        model,
        checkpoint["optimizer_state_dict"],
        checkpoint.get("scheduler_state_dict"),
        checkpoint["epoch"],
        checkpoint["stage"],
        checkpoint["best_val_loss"],
        checkpoint["config"],
        checkpoint["training_history"]
    )


def run_experiment(
    experiment_id: str,
    lr: float,
    freeze_strategy: str,
    arch_variant: str,
    reg_variant: str,
    base_config: Dict,
    train_loader,
    val_loader,
    node_features: torch.Tensor,
    num_nodes: int,
    id_to_index: Dict,
    index_to_id: Dict,
    device: torch.device,
    output_dir: str,
    logger,
    resume_path: Optional[str] = None
) -> Dict:
    """
    Run a single experiment with given configuration.
    
    Args:
        experiment_id: Unique experiment identifier
        lr: Base learning rate
        freeze_strategy: One of FREEZE_STRATEGIES keys
        arch_variant: One of ARCHITECTURE_VARIANTS keys
        reg_variant: One of REGULARIZATION_VARIANTS keys
        base_config: Base configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        node_features: Node feature tensor
        num_nodes: Number of nodes
        id_to_index: Card ID to index mapping
        index_to_id: Index to card ID mapping
        device: Device to train on
        output_dir: Directory to save results
        logger: Logger instance
        resume_path: Path to checkpoint for resuming (optional)
        
    Returns:
        Dictionary with experiment results
    """
    exp_dir = os.path.join(output_dir, experiment_id)
    ensure_dir(exp_dir)
    
    # Get configurations
    freeze_config = FREEZE_STRATEGIES[freeze_strategy]
    arch_config = ARCHITECTURE_VARIANTS[arch_variant]
    reg_config = REGULARIZATION_VARIANTS[reg_variant]
    
    # Build experiment config
    exp_config = copy.deepcopy(base_config)
    exp_config["experiment"] = {
        "id": experiment_id,
        "lr": lr,
        "freeze_strategy": freeze_strategy,
        "arch_variant": arch_variant,
        "reg_variant": reg_variant
    }
    
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_id}")
    print(f"  LR: {lr}, Freeze: {freeze_strategy}, Arch: {arch_variant}, Reg: {reg_variant}")
    print(f"{'='*80}")
    
    logger.info(f"Starting experiment: {experiment_id}")
    
    # Initialize model with architecture variant
    model = CardRecommendationSAGEWithTransfer(
        num_nodes=num_nodes,
        node_feature_dim=node_features.size(1),
        pretrained_dim=128,
        hidden_dims=arch_config["hidden_dims"],
        finetune_dims=arch_config["finetune_dims"],
        dropout_rates=reg_config["dropout_rates"],
        num_cards=base_config["model"]["num_cards"],
        aggr=base_config["model"].get("sage_aggr", "mean"),
        use_pretrained=False
    )
    model = model.to(device)
    
    # Training history
    training_history = []
    overall_best_val_loss = float('inf')
    best_stage = None
    start_epoch = 0
    current_stage_idx = 0
    
    # Resume from checkpoint if provided
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        logger.info(f"Resuming from checkpoint: {resume_path}")
        (model, _, _, start_epoch, resume_stage, 
         overall_best_val_loss, _, training_history) = load_checkpoint(resume_path, model, device)
        # Find stage index
        stages = freeze_config["stages"]
        for i, s in enumerate(stages):
            if s == resume_stage:
                current_stage_idx = i
                break
    
    # Stage configurations
    stages = freeze_config["stages"]
    stage_epochs = freeze_config["stage_epochs"]
    
    # LR factors for stages
    lr_factors = {
        "adapter": 1.0,
        "partial": 0.5,
        "full": 0.1
    }
    
    for stage_idx, stage_key in enumerate(stages):
        if stage_idx < current_stage_idx:
            continue  # Skip already completed stages when resuming
            
        stage_name = f"Stage {stage_idx + 1}: {stage_key.title()}"
        num_epochs = stage_epochs.get(stage_key, 2)
        stage_lr = lr * lr_factors.get(stage_key, 1.0)
        
        # Set training stage
        print(f"\n{'='*60}")
        model.set_training_stage(stage_key)
        print(f"{'='*60}")
        
        # Create optimizer with regularization weight decay
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage_lr,
            weight_decay=reg_config["weight_decay"]
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7
        )
        
        stage_best_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 2  # Shorter patience for grid search
        
        total_train_time = 0.0
        total_val_time = 0.0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_metrics, train_time = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                id_to_index=id_to_index,
                index_to_id=index_to_id,
                num_cards=base_config["model"]["num_cards"],
                loss_aggregation=base_config["model"].get("loss_aggregation", "mean"),
                gradient_clip_norm=base_config["training"].get("gradient_clip_norm", 1.0),
                compute_metrics_flag=True,
                total_train_time=total_train_time,
                total_val_time=total_val_time,
                use_mixed_precision=base_config["training"].get("use_mixed_precision", False),
                logger=logger
            )
            total_train_time += train_time
            
            # Validation
            val_loss, val_metrics, val_time = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                id_to_index=id_to_index,
                index_to_id=index_to_id,
                num_cards=base_config["model"]["num_cards"],
                loss_aggregation=base_config["model"].get("loss_aggregation", "mean"),
                compute_metrics_flag=True,
                total_train_time=total_train_time,
                total_val_time=total_val_time,
                logger=logger
            )
            total_val_time += val_time
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log results
            epoch_result = {
                "experiment_id": experiment_id,
                "stage": stage_name,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics
            }
            training_history.append(epoch_result)
            
            # Print progress
            top2_acc = val_metrics.get("top_2_acc", 0.0) if val_metrics else 0.0
            print(f"\n{stage_name} - Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Val Top-2 Acc: {top2_acc:.4f}, LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < stage_best_loss:
                stage_best_loss = val_loss
                patience_counter = 0
                print(f"  ✓ New best: {stage_best_loss:.4f}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(exp_dir, "best_checkpoint.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, stage_key,
                    stage_best_loss, exp_config, training_history, checkpoint_path
                )
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
                
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping in {stage_name}")
                break
        
        # Track overall best
        if stage_best_loss < overall_best_val_loss:
            overall_best_val_loss = stage_best_loss
            best_stage = stage_name
    
    # Save final results
    results = {
        "experiment_id": experiment_id,
        "lr": lr,
        "freeze_strategy": freeze_strategy,
        "arch_variant": arch_variant,
        "reg_variant": reg_variant,
        "best_val_loss": overall_best_val_loss,
        "best_stage": best_stage,
        "final_top2_acc": training_history[-1]["val_metrics"].get("top_2_acc", 0.0) if training_history else 0.0,
        "num_epochs_total": len(training_history),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save history
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Save results summary
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Experiment {experiment_id} complete")
    print(f"  Best Val Loss: {overall_best_val_loss:.4f}")
    print(f"  Best Stage: {best_stage}")
    
    logger.info(f"Experiment {experiment_id} completed: best_val_loss={overall_best_val_loss:.4f}")
    
    return results


def main():
    """Run grid search over hyperparameters."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search")
    parser.add_argument("--lr", type=float, nargs="+", default=None,
                       help="Learning rates to test (default: all)")
    parser.add_argument("--freeze", type=str, nargs="+", default=None,
                       help="Freeze strategies to test (default: all)")
    parser.add_argument("--arch", type=str, nargs="+", default=None,
                       help="Architecture variants to test (default: all)")
    parser.add_argument("--reg", type=str, nargs="+", default=None,
                       help="Regularization variants to test (default: all)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume specific experiment from checkpoint")
    parser.add_argument("--output", type=str, default="models_grid",
                       help="Output directory (default: models_grid)")
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 80)
    print("Hyperparameter Grid Search")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config
    config = load_config()
    
    # Output directory
    output_dir = args.output
    ensure_dir(output_dir)
    
    # Setup logger
    logger = setup_training_logger(log_dir=output_dir, log_file="grid_search.log")
    logger.info("Starting grid search")
    
    # Get device
    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")
    
    # Load data once
    print("\nLoading training data...")
    (train_examples, val_examples, node_features, edge_index, edge_attr,
     id_to_index, index_to_id, num_nodes) = load_training_data(config)
    
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    # Create data loaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    num_cards = config["model"]["num_cards"]
    
    train_dataset = CardDataset(train_examples, node_features, edge_index, edge_attr,
                                id_to_index, index_to_id, num_cards)
    val_dataset = CardDataset(val_examples, node_features, edge_index, edge_attr,
                              id_to_index, index_to_id, num_cards)
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    
    # Determine experiments to run
    lrs = args.lr if args.lr else LEARNING_RATES
    freezes = args.freeze if args.freeze else list(FREEZE_STRATEGIES.keys())
    archs = args.arch if args.arch else list(ARCHITECTURE_VARIANTS.keys())
    regs = args.reg if args.reg else list(REGULARIZATION_VARIANTS.keys())
    
    # Generate experiment combinations
    experiments = list(product(lrs, freezes, archs, regs))
    
    print(f"\nGrid Search Configuration:")
    print(f"  Learning rates: {lrs}")
    print(f"  Freeze strategies: {freezes}")
    print(f"  Architectures: {archs}")
    print(f"  Regularization: {regs}")
    print(f"  Total experiments: {len(experiments)}")
    
    # Results aggregation
    all_results = []
    results_csv_path = os.path.join(output_dir, "grid_results.csv")
    
    # Run experiments
    for i, (lr, freeze, arch, reg) in enumerate(experiments):
        exp_id = create_experiment_id(lr, freeze, arch, reg)
        
        print(f"\n[{i+1}/{len(experiments)}] Running experiment: {exp_id}")
        logger.info(f"[{i+1}/{len(experiments)}] Starting experiment: {exp_id}")
        
        # Check if already completed
        results_path = os.path.join(output_dir, exp_id, "results.json")
        if os.path.exists(results_path) and not args.resume:
            print(f"  Skipping (already completed)")
            with open(results_path) as f:
                results = json.load(f)
            all_results.append(results)
            continue
        
        # Resume path
        resume_path = None
        if args.resume == exp_id:
            resume_path = os.path.join(output_dir, exp_id, "best_checkpoint.pt")
        
        try:
            results = run_experiment(
                experiment_id=exp_id,
                lr=lr,
                freeze_strategy=freeze,
                arch_variant=arch,
                reg_variant=reg,
                base_config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                node_features=node_features,
                num_nodes=num_nodes,
                id_to_index=id_to_index,
                index_to_id=index_to_id,
                device=device,
                output_dir=output_dir,
                logger=logger,
                resume_path=resume_path
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {str(e)}")
            print(f"  ERROR: {str(e)}")
            continue
        
        # Save intermediate results
        with open(results_csv_path, "w", newline="") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("Grid Search Complete")
    print(f"{'='*80}")
    
    if all_results:
        # Sort by best val loss
        sorted_results = sorted(all_results, key=lambda x: x["best_val_loss"])
        
        print("\nTop 5 Experiments:")
        print("-" * 80)
        for i, r in enumerate(sorted_results[:5]):
            print(f"{i+1}. {r['experiment_id']}")
            print(f"   Val Loss: {r['best_val_loss']:.4f}, Top-2 Acc: {r['final_top2_acc']:.4f}")
        
        # Save final summary
        summary_path = os.path.join(output_dir, "grid_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "total_experiments": len(all_results),
                "best_experiment": sorted_results[0] if sorted_results else None,
                "all_results": sorted_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - grid_results.csv")
        print(f"  - grid_summary.json")
    
    logger.info("Grid search completed")


if __name__ == "__main__":
    main()

