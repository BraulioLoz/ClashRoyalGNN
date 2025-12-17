"""
Training script for transfer learning with pretrained GraphSAGE.

This script implements staged training:
1. Stage 1: Train adapter + output head (pretrained encoder frozen)
2. Stage 2: Partial fine-tuning (unfreeze last 2 layers)
3. Stage 3: Full fine-tuning (all layers trainable)
"""
import sys
import os
import json
import torch
import torch.optim as optim
from datetime import datetime
from typing import Dict, Optional

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
from src.utils import load_config, get_device, save_model, setup_training_logger


def train_stage(
    model: CardRecommendationSAGEWithTransfer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device: torch.device,
    config: Dict,
    id_to_index: Dict,
    index_to_id: Dict,
    logger,
    stage_name: str,
    num_epochs: int,
    stage_history: list
) -> tuple:
    """
    Train a single stage of transfer learning.
    
    Args:
        model: Transfer learning model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Configuration dictionary
        id_to_index: Card ID to index mapping
        index_to_id: Index to card ID mapping
        logger: Logger instance
        stage_name: Name of current stage
        num_epochs: Number of epochs for this stage
        stage_history: History list to append to
        
    Returns:
        Tuple of (best_val_loss, model, history)
    """
    print(f"\n{'='*80}")
    print(f"Training {stage_name}")
    print(f"{'='*80}")
    
    logger.info(f"Starting {stage_name}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config["training"]["early_stopping_patience"]
    gradient_clip_norm = config["training"].get("gradient_clip_norm", None)
    loss_aggregation = config["model"].get("loss_aggregation", "mean")
    compute_metrics_flag = config["training"].get("compute_metrics", True)
    use_mixed_precision = config["training"].get("use_mixed_precision", False) and torch.cuda.is_available()
    num_cards = config["model"]["num_cards"]
    
    total_train_time = 0.0
    total_val_time = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_metrics, train_time = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            id_to_index=id_to_index,
            index_to_id=index_to_id,
            num_cards=num_cards,
            loss_aggregation=loss_aggregation,
            gradient_clip_norm=gradient_clip_norm,
            compute_metrics_flag=compute_metrics_flag,
            total_train_time=total_train_time,
            total_val_time=total_val_time,
            use_mixed_precision=use_mixed_precision,
            logger=logger
        )
        total_train_time += train_time
        
        # Validation
        model.eval()
        val_loss, val_metrics, val_time = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            id_to_index=id_to_index,
            index_to_id=index_to_id,
            num_cards=num_cards,
            loss_aggregation=loss_aggregation,
            compute_metrics_flag=compute_metrics_flag,
            total_train_time=total_train_time,
            total_val_time=total_val_time,
            logger=logger
        )
        total_val_time += val_time
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        epoch_result = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr
        }
        
        # Add metrics if available
        if train_metrics:
            epoch_result["train_metrics"] = train_metrics
        
        if val_metrics:
            epoch_result["val_metrics"] = val_metrics
            # Display Top-2 accuracy if available
            if "top_2_acc" in val_metrics:
                print(f"\n{stage_name} - Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
                print(f"  Val Top-2 Acc: {val_metrics['top_2_acc']:.4f}")
            else:
                print(f"\n{stage_name} - Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"\n{stage_name} - Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        stage_history.append(epoch_result)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
        
        if patience_counter >= early_stopping_patience:
            print(f"\n  Early stopping triggered in {stage_name}")
            logger.info(f"Early stopping triggered in {stage_name} at epoch {epoch+1}")
            break
    
    return best_val_loss, model, stage_history


def main():
    """Main transfer learning training function."""
    print("=" * 80)
    print("Transfer Learning with Pretrained GraphSAGE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config
    config = load_config()
    
    # Setup output directory
    model_save_dir = config["training"]["model_save_dir"] + "_transfer"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_training_logger(log_dir=model_save_dir, log_file="transfer_training.log")
    
    logger.info("Starting transfer learning training")
    logger.info(f"Output directory: {model_save_dir}")
    
    # Get device
    device = get_device(config["training"]["device"])
    
    # Load data
    print("Loading training data...")
    logger.info("Loading training data...")
    
    (train_examples, val_examples, node_features, edge_index, edge_attr,
     id_to_index, index_to_id, num_nodes) = load_training_data(config)
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Node features: {node_features.shape}")
    
    # Create datasets and loaders
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
    
    # Initialize transfer learning model
    print("\nInitializing transfer learning model...")
    logger.info("Initializing transfer learning model...")
    
    model = CardRecommendationSAGEWithTransfer(
        num_nodes=num_nodes,
        node_feature_dim=node_features.size(1),
        pretrained_dim=128,
        hidden_dims=[256, 128],
        finetune_dims=[64],
        dropout_rates=config["model"]["dropout_rates"],
        num_cards=config["model"]["num_cards"],
        aggr=config["model"].get("sage_aggr", "mean"),
        use_pretrained=True  # Start with random weights (no OGB pretrained available for this domain)
    )
    
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training history
    training_history = []
    
    # Load transfer learning configuration
    transfer_config = config["training"].get("transfer_learning", {})
    base_lr = config["training"]["lr"]
    
    # Stage-specific configurations (configurable from config.yaml)
    stages = [
        {
            "name": "Stage 1: Adapter Training",
            "stage_key": "adapter",
            "epochs": transfer_config.get("stage1_epochs", 10),
            "lr": base_lr
        },
        {
            "name": "Stage 2: Partial Fine-tuning",
            "stage_key": "partial",
            "epochs": transfer_config.get("stage2_epochs", 20),
            "lr": base_lr * transfer_config.get("stage2_lr_factor", 0.5)
        },
        {
            "name": "Stage 3: Full Fine-tuning",
            "stage_key": "full",
            "epochs": transfer_config.get("stage3_epochs", 20),
            "lr": base_lr * transfer_config.get("stage3_lr_factor", 0.1)
        }
    ]
    
    # Log configuration
    logger.info("Transfer learning configuration:")
    for stage in stages:
        logger.info(f"  {stage['name']}: {stage['epochs']} epochs, LR={stage['lr']:.6f}")
    
    print("\nTransfer Learning Configuration:")
    for stage in stages:
        print(f"  {stage['name']}: {stage['epochs']} epochs, LR={stage['lr']:.6f}")
    
    overall_best_val_loss = float('inf')
    best_stage = None
    
    for stage_config in stages:
        stage_name = stage_config["name"]
        stage_key = stage_config["stage_key"]
        stage_epochs = stage_config["epochs"]
        stage_lr = stage_config["lr"]
        
        # Set training stage (freezes/unfreezes layers)
        print(f"\n{'='*80}")
        model.set_training_stage(stage_key)
        print(f"{'='*80}")
        
        # Create optimizer for this stage
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage_lr,
            weight_decay=0.01
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train this stage
        stage_best_loss, model, training_history = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            id_to_index=id_to_index,
            index_to_id=index_to_id,
            logger=logger,
            stage_name=stage_name,
            num_epochs=stage_epochs,
            stage_history=training_history
        )
        
        # Track overall best
        if stage_best_loss < overall_best_val_loss:
            overall_best_val_loss = stage_best_loss
            best_stage = stage_name
            
            # Save best model
            model_path = os.path.join(model_save_dir, "best_model.pt")
            save_model(model, model_path, metadata={
                "stage": stage_name,
                "val_loss": stage_best_loss,
                "config": config
            })
            print(f"\n  ✓ Saved best model from {stage_name}")
            logger.info(f"Saved best model from {stage_name} with val_loss={stage_best_loss:.4f}")
    
    # Save training history
    history_json_path = os.path.join(model_save_dir, "training_history.json")
    history_csv_path = os.path.join(model_save_dir, "training_history.csv")
    save_training_history(
        training_history,
        history_json_path,
        history_csv_path,
        format_type="both"
    )
    
    print(f"\n{'='*80}")
    print("Transfer Learning Training Complete")
    print(f"{'='*80}")
    print(f"Best validation loss: {overall_best_val_loss:.4f}")
    print(f"Best stage: {best_stage}")
    print(f"Model saved to: {model_save_dir}/best_model.pt")
    print(f"History saved to: {history_json_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("Transfer learning training completed successfully")
    logger.info(f"Best validation loss: {overall_best_val_loss:.4f}")
    logger.info(f"Best stage: {best_stage}")


if __name__ == "__main__":
    main()




