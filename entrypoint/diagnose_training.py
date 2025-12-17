"""
Diagnostic script for training issues.
Loads a small subset of data and runs one forward/backward pass to diagnose issues.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import json
import numpy as np
from src.utils import load_config, get_device
from src.models.gnn_model import CardRecommendationGNN
from src.pipelines.training_pipeline import (
    CardDataset, collate_fn, compute_loss, check_gradients,
    compute_metrics, inspect_logits_and_probs
)
from torch.utils.data import DataLoader


def main():
    """Run diagnostic checks on training setup."""
    print("=" * 60)
    print("Clash Royale GNN - Training Diagnostics")
    print("=" * 60)
    
    # Load config
    config = load_config()
    device = get_device(config["training"]["device"])
    
    # Load graph data
    features_dir = config["data"]["features_dir"]
    graph_data_path = os.path.join(features_dir, "graph_data.json")
    examples_path = os.path.join(features_dir, "training_examples.json")
    
    if not os.path.exists(graph_data_path):
        print(f"Error: Graph data not found at {graph_data_path}")
        print("Please run process_features.py first")
        return
    
    if not os.path.exists(examples_path):
        print(f"Error: Training examples not found at {examples_path}")
        print("Please run process_features.py first")
        return
    
    print("\n1. Loading data...")
    with open(graph_data_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    
    with open(examples_path, "r", encoding="utf-8") as f:
        training_examples = json.load(f)
    
    # Use small subset for diagnostics
    sample_size = min(100, len(training_examples))
    sample_examples = training_examples[:sample_size]
    print(f"Using {sample_size} examples for diagnostics")
    
    # Convert to tensors
    node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_data["edge_attr"], dtype=torch.float32)
    id_to_index = {int(k): v for k, v in graph_data["id_to_index"].items()}
    index_to_id = {int(k): v for k, v in graph_data["index_to_id"].items()}
    num_nodes = graph_data["num_nodes"]
    
    if isinstance(edge_index, torch.Tensor):
        num_edges = edge_index.size(1) if edge_index.dim() > 1 else 0
    else:
        num_edges = len(edge_index[0]) if edge_index and len(edge_index) > 0 else 0
    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"Node features shape: {node_features.shape}")
    
    # Check feature statistics
    print("\n2. Feature Statistics:")
    for i, feat_name in enumerate(config["graph"]["node_features"]):
        if i < node_features.size(1):
            feat_col = node_features[:, i]
            print(f"  {feat_name}: mean={feat_col.mean().item():.4f}, "
                  f"std={feat_col.std().item():.4f}, "
                  f"min={feat_col.min().item():.4f}, max={feat_col.max().item():.4f}")
    
    # Create dataset and dataloader
    dataset = CardDataset(
        sample_examples, node_features, edge_index, edge_attr,
        id_to_index, index_to_id, config["model"]["num_cards"]
    )
    
    dataloader = DataLoader(
        dataset, batch_size=min(8, sample_size), shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Initialize model
    print("\n3. Initializing model...")
    model = CardRecommendationGNN(
        num_nodes=num_nodes,
        node_feature_dim=node_features.size(1),
        hidden_dims=config["model"]["hidden_dims"],
        dropout_rates=config["model"]["dropout_rates"],
        gnn_type=config["model"]["gnn_type"],
        num_cards=config["model"]["num_cards"],
        weight_init=config["model"].get("weight_init", "xavier")
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    lr = float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Learning rate: {lr}")
    
    # Run one forward/backward pass
    print("\n4. Running forward/backward pass...")
    model.train()
    
    batch = next(iter(dataloader))
    batched_data, target_list = batch
    batched_data = batched_data.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    logits = model(batched_data)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats - mean: {logits.mean().item():.4f}, "
          f"std: {logits.std().item():.4f}, "
          f"min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
    
    # Compute loss for each example in batch
    batch_loss = 0.0
    loss_aggregation = config["model"].get("loss_aggregation", "mean")
    
    if hasattr(batched_data, 'batch') and batched_data.batch is not None:
        batch_size = batched_data.batch.max().item() + 1
        
        for i in range(min(3, batch_size)):  # Check first 3 examples
            node_mask = batched_data.batch == i
            graph_logits = logits[node_mask]
            target_indices = target_list[i] if i < len(target_list) else []
            
            # Extract input card IDs
            input_card_ids = []
            if hasattr(batched_data, 'input_cards'):
                input_cards = batched_data.input_cards[node_mask]
                input_indices = torch.where(input_cards == 1)[0].cpu().numpy()
                for local_idx in input_indices:
                    local_idx_int = int(local_idx)
                    if local_idx_int in index_to_id:
                        input_card_ids.append(index_to_id[local_idx_int])
            
            loss = compute_loss(
                graph_logits, target_indices, input_card_ids,
                id_to_index, config["model"]["num_cards"], device, loss_aggregation
            )
            batch_loss += loss
            
            print(f"\n  Example {i+1}:")
            print(f"    Input cards: {input_card_ids}")
            print(f"    Target cards: {target_indices}")
            print(f"    Loss: {loss.item():.4f}")
            
            # Inspect logits and probs
            if len(target_indices) > 0:
                inspection = inspect_logits_and_probs(
                    graph_logits, target_indices, input_card_ids,
                    id_to_index, index_to_id, config["model"]["num_cards"], device, sample_size=3
                )
                print(f"    Top prediction: {inspection['top_predictions'][0] if inspection['top_predictions'] else 'N/A'}")
                if inspection['target_probs']:
                    print(f"    Target prob: {inspection['target_probs'][0]['prob']:.6f}")
                
                # Compute metrics
                metrics = compute_metrics(
                    graph_logits, target_indices, input_card_ids,
                    id_to_index, index_to_id, config["model"]["num_cards"], device
                )
                print(f"    Top-2 accuracy: {metrics.get('top_2_acc', 0):.4f}")
                print(f"    Mean target prob: {metrics.get('mean_target_prob', 0):.6f}")
    
    batch_loss = batch_loss / min(3, batch_size)
    
    # Backward pass
    batch_loss.backward()
    
    # Check gradients
    print("\n5. Gradient Statistics:")
    grad_stats = check_gradients(model)
    
    grad_norms = []
    for name, stats in grad_stats.items():
        if "no_grad" not in stats and stats["norm"] > 0:
            grad_norms.append(stats["norm"])
            if len(grad_norms) <= 10:  # Print first 10
                print(f"  {name}: norm={stats['norm']:.6f}, mean={stats['mean']:.6f}")
    
    if grad_norms:
        print(f"\n  Overall: mean={np.mean(grad_norms):.6f}, "
              f"min={np.min(grad_norms):.6f}, max={np.max(grad_norms):.6f}")
        
        if np.mean(grad_norms) < 1e-6:
            print("  WARNING: Very small gradients detected! Possible vanishing gradient problem.")
        elif np.mean(grad_norms) > 100:
            print("  WARNING: Very large gradients detected! Consider gradient clipping.")
    else:
        print("  WARNING: No gradients detected!")
    
    # Gradient clipping test
    gradient_clip_norm = config["training"].get("gradient_clip_norm", None)
    if gradient_clip_norm:
        print(f"\n6. Testing gradient clipping (max_norm={gradient_clip_norm})...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        grad_stats_after = check_gradients(model)
        grad_norms_after = [stats["norm"] for stats in grad_stats_after.values() 
                           if "no_grad" not in stats and stats["norm"] > 0]
        if grad_norms_after:
            print(f"  After clipping: mean={np.mean(grad_norms_after):.6f}, "
                  f"max={np.max(grad_norms_after):.6f}")
    
    print("\n" + "=" * 60)
    print("Diagnostics complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

