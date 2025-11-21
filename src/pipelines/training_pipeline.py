import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..models.gnn_model import CardRecommendationGNN
from ..utils import load_config, get_device, save_model, ensure_dir
from .feature_eng_pipeline import create_pyg_data


class CardDataset(Dataset):
    """Dataset for card recommendation training examples."""
    
    def __init__(
        self,
        examples: List[Dict],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        id_to_index: Dict[int, int],
        index_to_id: Dict[int, int],
        num_cards: int
    ):
        self.examples = examples
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.id_to_index = id_to_index
        self.index_to_id = index_to_id
        self.num_cards = num_cards
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_cards = example["input_cards"]
        target_cards = example["target_cards"]
        
        # Create PyG Data object
        data = create_pyg_data(
            self.node_features,
            self.edge_index,
            self.edge_attr,
            input_cards,
            self.id_to_index,
            self.index_to_id
        )
        
        # Create target tensor (one-hot or indices)
        target_indices = []
        for card_id in target_cards:
            if card_id in self.id_to_index:
                idx_val = self.id_to_index[card_id]
                if idx_val < self.num_cards:
                    target_indices.append(idx_val)
        
        # Return data and target
        return data, target_indices


def collate_fn(batch):
    """Custom collate function for batching PyG Data objects."""
    data_list, target_list = zip(*batch)
    
    # Remove dict attributes that can't be batched by PyG
    # These are the same for all graphs, so we can access them from the first graph
    cleaned_data_list = []
    dict_attrs = {}  # Store dict attributes from first graph
    
    for i, data in enumerate(data_list):
        # Create a copy without dict attributes
        cleaned_data = Data()
        
        # Copy all tensor and basic attributes, excluding dicts
        for key in data.keys():
            value = getattr(data, key)
            # Skip dict attributes that cause batching issues
            if key in ['card_id_to_index', 'index_to_card_id']:
                # Store from first graph only
                if i == 0:
                    dict_attrs[key] = value
                continue
            # Copy tensor and other batcheable attributes
            if torch.is_tensor(value) or isinstance(value, (int, float, str, bool, type(None))):
                setattr(cleaned_data, key, value)
        
        cleaned_data_list.append(cleaned_data)
    
    # Use Batch.from_data_list to create batched graphs
    # This allows processing multiple examples in parallel on GPU
    batched_data = Batch.from_data_list(cleaned_data_list)
    
    # Add back the dict mappings from the first graph (all graphs share same structure)
    for key, value in dict_attrs.items():
        setattr(batched_data, key, value)
    
    return batched_data, list(target_list)


def compute_loss(
    logits: torch.Tensor, 
    target_indices: List[int], 
    input_card_ids: List[int], 
    id_to_index: Dict, 
    num_cards: int, 
    device: torch.device,
    loss_aggregation: str = "mean"
) -> torch.Tensor:
    """
    Compute loss for card recommendation.
    Only compute loss on target cards, excluding input cards.
    
    Args:
        logits: Model output [num_nodes, num_cards]
        target_indices: List of target card indices
        input_card_ids: List of input card IDs
        id_to_index: Mapping from card ID to index
        num_cards: Total number of cards
        device: Device to compute on
        loss_aggregation: How to aggregate node logits ("mean", "max", "input_nodes")
        
    Returns:
        Loss value
    """
    # Get node-level predictions based on aggregation method
    if loss_aggregation == "mean":
        node_logits = logits.mean(dim=0)  # [num_cards]
    elif loss_aggregation == "max":
        node_logits = logits.max(dim=0)[0]  # [num_cards]
    elif loss_aggregation == "input_nodes":
        # Use only input node logits
        input_node_indices = []
        for card_id in input_card_ids:
            if card_id in id_to_index:
                idx = id_to_index[card_id]
                if idx < num_cards:
                    input_node_indices.append(idx)
        if input_node_indices:
            # Find which nodes correspond to input cards
            # This is approximate - we'd need node-to-card mapping
            node_logits = logits.mean(dim=0)  # Fallback to mean
        else:
            node_logits = logits.mean(dim=0)
    else:
        node_logits = logits.mean(dim=0)
    
    # Create mask to exclude input cards
    mask = torch.ones(num_cards, device=device, dtype=torch.bool)
    for card_id in input_card_ids:
        if card_id in id_to_index:
            idx = id_to_index[card_id]
            if idx < num_cards:
                mask[idx] = False
    
    # CRITICAL FIX: Instead of multiplying by 0, set to large negative value
    # This ensures softmax properly excludes input cards without numerical issues
    masked_logits = node_logits.clone()
    masked_logits[~mask] = float('-inf')
    
    # Create target tensor (multi-label: both target cards should be predicted)
    target = torch.zeros(num_cards, device=device)
    for idx in target_indices:
        if idx < num_cards:
            target[idx] = 1.0
    
    # Apply mask to target
    masked_target = target * mask.float()
    
    # Normalize target
    if masked_target.sum() > 0:
        masked_target = masked_target / masked_target.sum()
    else:
        # No valid targets, return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute cross-entropy loss
    # Convert to probabilities
    probs = torch.softmax(masked_logits, dim=0)
    
    # Compute negative log likelihood
    loss = -torch.sum(masked_target * torch.log(probs + 1e-8))
    
    return loss


def check_gradients(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Check gradient flow in the model by computing gradient norms per layer.
    
    Args:
        model: The model to check
        
    Returns:
        Dictionary with gradient statistics per parameter group
    """
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item()
            grad_min = param.grad.data.min().item()
            grad_max = param.grad.data.max().item()
            
            grad_stats[name] = {
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "min": grad_min,
                "max": grad_max
            }
        else:
            grad_stats[name] = {
                "norm": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "no_grad": True
            }
    
    return grad_stats


def compute_metrics(
    logits: torch.Tensor,
    target_indices: List[int],
    input_card_ids: List[int],
    id_to_index: Dict,
    index_to_id: Dict,
    num_cards: int,
    device: torch.device,
    top_k_values: List[int] = [1, 2, 5]
) -> Dict[str, float]:
    """
    Compute accuracy metrics for card recommendation.
    
    Args:
        logits: Model output [num_nodes, num_cards]
        target_indices: List of target card indices
        input_card_ids: List of input card IDs
        id_to_index: Mapping from card ID to index
        index_to_id: Mapping from index to card ID
        num_cards: Total number of cards
        device: Device to compute on
        top_k_values: List of k values for top-k accuracy
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Get node-level predictions
    node_logits = logits.mean(dim=0)  # [num_cards]
    
    # Create mask to exclude input cards
    mask = torch.ones(num_cards, device=device, dtype=torch.bool)
    for card_id in input_card_ids:
        if card_id in id_to_index:
            idx = id_to_index[card_id]
            if idx < num_cards:
                mask[idx] = False
    
    # Set input card logits to -inf
    masked_logits = node_logits.clone()
    masked_logits[~mask] = float('-inf')
    
    # Get probabilities
    probs = torch.softmax(masked_logits, dim=0)
    
    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probs, k=min(max(top_k_values), num_cards))
    
    metrics = {}
    
    # Compute top-k accuracy
    target_set = set(target_indices)
    for k in top_k_values:
        if k <= len(top_k_indices):
            top_k_pred = top_k_indices[:k].cpu().tolist()
            # Check if any target is in top-k predictions
            hit = len(target_set.intersection(set(top_k_pred))) > 0
            metrics[f"top_{k}_acc"] = 1.0 if hit else 0.0
    
    # Compute probability of target cards
    target_probs = []
    for idx in target_indices:
        if idx < num_cards:
            target_probs.append(probs[idx].item())
    
    if target_probs:
        metrics["mean_target_prob"] = np.mean(target_probs)
        metrics["min_target_prob"] = np.min(target_probs)
        metrics["max_target_prob"] = np.max(target_probs)
    else:
        metrics["mean_target_prob"] = 0.0
        metrics["min_target_prob"] = 0.0
        metrics["max_target_prob"] = 0.0
    
    # Logits statistics
    metrics["logits_mean"] = node_logits.mean().item()
    metrics["logits_std"] = node_logits.std().item()
    metrics["logits_min"] = node_logits.min().item()
    metrics["logits_max"] = node_logits.max().item()
    
    return metrics


def inspect_logits_and_probs(
    logits: torch.Tensor,
    target_indices: List[int],
    input_card_ids: List[int],
    id_to_index: Dict,
    index_to_id: Dict,
    num_cards: int,
    device: torch.device,
    sample_size: int = 5
) -> Dict:
    """
    Inspect logits and probabilities for debugging.
    
    Args:
        logits: Model output [num_nodes, num_cards]
        target_indices: List of target card indices
        input_card_ids: List of input card IDs
        id_to_index: Mapping from card ID to index
        index_to_id: Mapping from index to card ID
        num_cards: Total number of cards
        device: Device to compute on
        sample_size: Number of top predictions to return
        
    Returns:
        Dictionary with inspection results
    """
    node_logits = logits.mean(dim=0)  # [num_cards]
    
    # Create mask
    mask = torch.ones(num_cards, device=device, dtype=torch.bool)
    for card_id in input_card_ids:
        if card_id in id_to_index:
            idx = id_to_index[card_id]
            if idx < num_cards:
                mask[idx] = False
    
    masked_logits = node_logits.clone()
    masked_logits[~mask] = float('-inf')
    probs = torch.softmax(masked_logits, dim=0)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probs, k=min(sample_size, num_cards))
    
    inspection = {
        "logits_stats": {
            "mean": node_logits.mean().item(),
            "std": node_logits.std().item(),
            "min": node_logits.min().item(),
            "max": node_logits.max().item()
        },
        "probs_stats": {
            "mean": probs[mask].mean().item() if mask.sum() > 0 else 0.0,
            "std": probs[mask].std().item() if mask.sum() > 0 else 0.0,
            "min": probs[mask].min().item() if mask.sum() > 0 else 0.0,
            "max": probs[mask].max().item() if mask.sum() > 0 else 0.0
        },
        "top_predictions": [
            {
                "card_id": index_to_id.get(int(idx.item()), int(idx.item())),
                "index": int(idx.item()),
                "prob": float(prob.item())
            }
            for prob, idx in zip(top_probs, top_indices)
        ],
        "target_probs": [
            {
                "card_id": index_to_id.get(idx, idx),
                "index": idx,
                "prob": float(probs[idx].item()) if idx < num_cards else 0.0
            }
            for idx in target_indices
        ],
        "input_card_probs": [
            {
                "card_id": card_id,
                "index": id_to_index.get(card_id, -1),
                "prob": float(probs[id_to_index[card_id]].item()) if card_id in id_to_index and id_to_index[card_id] < num_cards else 0.0
            }
            for card_id in input_card_ids[:sample_size]
        ]
    }
    
    return inspection


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    id_to_index: Dict,
    index_to_id: Dict,
    num_cards: int,
    loss_aggregation: str = "mean",
    gradient_clip_norm: Optional[float] = None,
    compute_metrics_flag: bool = False
) -> Tuple[float, Dict]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Aggregate metrics across batches
    all_metrics = defaultdict(list)
    
    for batched_data, target_list in tqdm(dataloader, desc="Training"):
        # Move data to device
        batched_data = batched_data.to(device)
        
        # Forward pass on batched graph
        optimizer.zero_grad()
        logits = model(batched_data)  # [total_nodes_in_batch, num_cards]
        
        # Process each example in the batch
        batch_loss = 0.0
        batch_metrics = []
        
        # Check if this is a batched graph or single graph
        if hasattr(batched_data, 'batch') and batched_data.batch is not None:
            # Batched graph: process each graph separately
            batch_size = batched_data.batch.max().item() + 1
            
            for i in range(batch_size):
                # Get nodes belonging to graph i
                node_mask = batched_data.batch == i
                graph_logits = logits[node_mask]  # [num_nodes_i, num_cards]
                
                # Get target and input cards for this example
                target_indices = target_list[i] if i < len(target_list) else []
                
                # Extract input card IDs from batched data
                input_card_ids = []
                if hasattr(batched_data, 'input_cards'):
                    input_cards = batched_data.input_cards[node_mask]
                    # Get local node indices where input_cards == 1
                    input_indices = torch.where(input_cards == 1)[0].cpu().numpy()
                    
                    # Use index_to_id mapping (all graphs share same structure)
                    # Local indices are 0 to num_nodes-1 for each graph
                    for local_idx in input_indices:
                        local_idx_int = int(local_idx)
                        if local_idx_int in index_to_id:
                            input_card_ids.append(index_to_id[local_idx_int])
                
                # Compute loss for this example
                loss = compute_loss(
                    graph_logits, target_indices, input_card_ids, 
                    id_to_index, num_cards, device, loss_aggregation
                )
                batch_loss += loss
                
                # Compute metrics if requested
                if compute_metrics_flag and len(target_indices) > 0:
                    metrics = compute_metrics(
                        graph_logits, target_indices, input_card_ids,
                        id_to_index, index_to_id, num_cards, device
                    )
                    batch_metrics.append(metrics)
            
            # Average loss across batch
            batch_loss = batch_loss / batch_size
        else:
            # Single graph (fallback for batch_size=1)
            target_indices = target_list[0] if target_list else []
            
            # Get input card IDs
            input_card_ids = []
            if hasattr(batched_data, 'input_cards'):
                input_cards_tensor = batched_data.input_cards
                for idx, val in enumerate(input_cards_tensor):
                    if val == 1 and idx in index_to_id:
                        input_card_ids.append(index_to_id[idx])
            
            batch_loss = compute_loss(
                logits, target_indices, input_card_ids, 
                id_to_index, num_cards, device, loss_aggregation
            )
            
            # Compute metrics if requested
            if compute_metrics_flag and len(target_indices) > 0:
                metrics = compute_metrics(
                    logits, target_indices, input_card_ids,
                    id_to_index, index_to_id, num_cards, device
                )
                batch_metrics.append(metrics)
        
        batch_loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
        # Aggregate metrics
        for metrics in batch_metrics:
            for key, value in metrics.items():
                all_metrics[key].append(value)
    
    # Compute average metrics
    avg_metrics = {
        key: np.mean(values) if values else 0.0
        for key, values in all_metrics.items()
    }
    
    return total_loss / max(num_batches, 1), avg_metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id_to_index: Dict,
    index_to_id: Dict,
    num_cards: int,
    loss_aggregation: str = "mean",
    compute_metrics_flag: bool = True
) -> Tuple[float, Dict]:
    """
    Validate model.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Aggregate metrics across batches
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batched_data, target_list in tqdm(dataloader, desc="Validating"):
            batched_data = batched_data.to(device)
            
            # Forward pass
            logits = model(batched_data)
            
            # Process each example in the batch
            batch_loss = 0.0
            batch_metrics = []
            
            # Check if this is a batched graph or single graph
            if hasattr(batched_data, 'batch') and batched_data.batch is not None:
                # Batched graph: process each graph separately
                batch_size = batched_data.batch.max().item() + 1
                
                for i in range(batch_size):
                    # Get nodes belonging to graph i
                    node_mask = batched_data.batch == i
                    graph_logits = logits[node_mask]  # [num_nodes_i, num_cards]
                    
                    # Get target and input cards for this example
                    target_indices = target_list[i] if i < len(target_list) else []
                    
                    # Extract input card IDs from batched data
                    input_card_ids = []
                    if hasattr(batched_data, 'input_cards'):
                        input_cards = batched_data.input_cards[node_mask]
                        # Get local node indices where input_cards == 1
                        input_indices = torch.where(input_cards == 1)[0].cpu().numpy()
                        
                        # Use index_to_id mapping (all graphs share same structure)
                        for local_idx in input_indices:
                            local_idx_int = int(local_idx)
                            if local_idx_int in index_to_id:
                                input_card_ids.append(index_to_id[local_idx_int])
                    
                    # Compute loss for this example
                    loss = compute_loss(
                        graph_logits, target_indices, input_card_ids, 
                        id_to_index, num_cards, device, loss_aggregation
                    )
                    batch_loss += loss
                    
                    # Compute metrics if requested
                    if compute_metrics_flag and len(target_indices) > 0:
                        metrics = compute_metrics(
                            graph_logits, target_indices, input_card_ids,
                            id_to_index, index_to_id, num_cards, device
                        )
                        batch_metrics.append(metrics)
                
                # Average loss across batch
                batch_loss = batch_loss / batch_size
            else:
                # Single graph (fallback for batch_size=1)
                target_indices = target_list[0] if target_list else []
                
                # Get input card IDs
                input_card_ids = []
                if hasattr(batched_data, 'input_cards'):
                    input_cards_tensor = batched_data.input_cards
                    for idx, val in enumerate(input_cards_tensor):
                        if val == 1 and idx in index_to_id:
                            input_card_ids.append(index_to_id[idx])
                
                batch_loss = compute_loss(
                    logits, target_indices, input_card_ids, 
                    id_to_index, num_cards, device, loss_aggregation
                )
                
                # Compute metrics if requested
                if compute_metrics_flag and len(target_indices) > 0:
                    metrics = compute_metrics(
                        logits, target_indices, input_card_ids,
                        id_to_index, index_to_id, num_cards, device
                    )
                    batch_metrics.append(metrics)
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Aggregate metrics
            for metrics in batch_metrics:
                for key, value in metrics.items():
                    all_metrics[key].append(value)
    
    # Compute average metrics
    avg_metrics = {
        key: np.mean(values) if values else 0.0
        for key, values in all_metrics.items()
    }
    
    return total_loss / max(num_batches, 1), avg_metrics


def train_model(
    config: Dict = None,
    graph_data_path: str = None,
    examples_path: str = None,
    model_save_path: str = None
) -> CardRecommendationGNN:
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        graph_data_path: Path to graph_data.json
        examples_path: Path to training_examples.json
        model_save_path: Path to save trained model
        
    Returns:
        Trained model
    """
    if config is None:
        config = load_config()
    
    # Load paths from config
    if graph_data_path is None:
        features_dir = config["data"]["features_dir"]
        graph_data_path = os.path.join(features_dir, "graph_data.json")
    
    if examples_path is None:
        features_dir = config["data"]["features_dir"]
        examples_path = os.path.join(features_dir, "training_examples.json")
    
    if model_save_path is None:
        model_save_dir = config["training"]["model_save_dir"]
        ensure_dir(model_save_dir)
        model_save_path = os.path.join(model_save_dir, "best_model.pt")
    
    # Load graph data
    print("Loading graph data...")
    with open(graph_data_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    
    node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_data["edge_attr"], dtype=torch.float32)
    id_to_index = {int(k): v for k, v in graph_data["id_to_index"].items()}
    index_to_id = {int(k): v for k, v in graph_data["index_to_id"].items()}
    num_nodes = graph_data["num_nodes"]
    
    # Load training examples
    print("Loading training examples...")
    with open(examples_path, "r", encoding="utf-8") as f:
        training_examples = json.load(f)
    
    print(f"Loaded {len(training_examples)} training examples")
    
    # Split train/val
    val_split = config["training"]["val_split"]
    split_idx = int(len(training_examples) * (1 - val_split))
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]
    
    print(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")
    
    # Create datasets
    train_dataset = CardDataset(
        train_examples, node_features, edge_index, edge_attr,
        id_to_index, index_to_id, config["model"]["num_cards"]
    )
    val_dataset = CardDataset(
        val_examples, node_features, edge_index, edge_attr,
        id_to_index, index_to_id, config["model"]["num_cards"]
    )
    
    # Create dataloaders
    batch_size = config["training"]["batch_size"]
    # Use num_workers for parallel data loading, pin_memory for faster GPU transfer
    num_workers = config["training"].get("num_workers", min(4, os.cpu_count() or 1))
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    
    # On Windows, num_workers > 0 requires if __name__ == "__main__" guard (which we have)
    # If issues occur, set num_workers to 0 in config
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    # Get device
    device_pref = config["training"]["device"]
    device = get_device(device_pref)
    
    # Initialize model
    print("Initializing model...")
    weight_init = config["model"].get("weight_init", "xavier")
    model = CardRecommendationGNN(
        num_nodes=num_nodes,
        node_feature_dim=node_features.size(1),
        hidden_dims=config["model"]["hidden_dims"],
        dropout_rates=config["model"]["dropout_rates"],
        gnn_type=config["model"]["gnn_type"],
        num_cards=config["model"]["num_cards"],
        weight_init=weight_init
    )
    print(f"Model initialized with {weight_init} weight initialization")
    
    model = model.to(device)
    
    # Optimizer
    lr = config["training"]["lr"]
    # Ensure lr is a float (YAML might parse as string)
    lr = float(lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = None
    lr_scheduler_config = config["training"].get("lr_scheduler", {})
    scheduler_type = lr_scheduler_config.get("type", "reduce_on_plateau")
    
    if scheduler_type == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_scheduler_config.get("factor", 0.5),
            patience=lr_scheduler_config.get("patience", 5),
            min_lr=lr_scheduler_config.get("min_lr", 1e-6),
            # verbose=True
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=lr_scheduler_config.get("min_lr", 1e-6)
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_config.get("step_size", 10),
            gamma=lr_scheduler_config.get("factor", 0.5)
        )
    
    # Training parameters
    epochs = config["training"]["epochs"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    save_best = config["training"]["save_best_model"]
    gradient_clip_norm = config["training"].get("gradient_clip_norm", None)
    loss_aggregation = config["model"].get("loss_aggregation", "mean")
    log_gradients_every_n_epochs = config["training"].get("log_gradients_every_n_epochs", 5)
    compute_metrics_flag = config["training"].get("compute_metrics", True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Learning rate: {lr}, Loss aggregation: {loss_aggregation}")
    if gradient_clip_norm:
        print(f"Gradient clipping: {gradient_clip_norm}")
    if scheduler:
        print(f"LR scheduler: {scheduler_type}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, id_to_index, index_to_id, 
            config["model"]["num_cards"],
            loss_aggregation=loss_aggregation,
            gradient_clip_norm=gradient_clip_norm,
            compute_metrics_flag=compute_metrics_flag
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, device, id_to_index, index_to_id, 
            config["model"]["num_cards"],
            loss_aggregation=loss_aggregation,
            compute_metrics_flag=compute_metrics_flag
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Learning rate changed: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Log metrics
        log_str = f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        if val_metrics and "top_2_acc" in val_metrics:
            log_str += f", Val Top-2 Acc: {val_metrics['top_2_acc']:.4f}"
        print(log_str)
        
        # Gradient monitoring
        if (epoch + 1) % log_gradients_every_n_epochs == 0 or epoch == 0:
            grad_stats = check_gradients(model)
            # Log summary of gradient norms
            grad_norms = [stats["norm"] for stats in grad_stats.values() if "no_grad" not in stats]
            if grad_norms:
                print(f"Gradient norms - Mean: {np.mean(grad_norms):.6f}, "
                      f"Min: {np.min(grad_norms):.6f}, Max: {np.max(grad_norms):.6f}")
        
        # Early epoch diagnostics
        if epoch == 0:
            print("\n=== Early Epoch Diagnostics ===")
            # Sample a batch for inspection
            sample_batch = next(iter(val_loader))
            batched_data, target_list = sample_batch
            batched_data = batched_data.to(device)
            with torch.no_grad():
                sample_logits = model(batched_data)
            
            if hasattr(batched_data, 'batch') and batched_data.batch is not None:
                batch_size = min(1, batched_data.batch.max().item() + 1)
                for i in range(batch_size):
                    node_mask = batched_data.batch == i
                    graph_logits = sample_logits[node_mask]
                    target_indices = target_list[i] if i < len(target_list) else []
                    
                    input_card_ids = []
                    if hasattr(batched_data, 'input_cards'):
                        input_cards = batched_data.input_cards[node_mask]
                        input_indices = torch.where(input_cards == 1)[0].cpu().numpy()
                        for local_idx in input_indices:
                            local_idx_int = int(local_idx)
                            if local_idx_int in index_to_id:
                                input_card_ids.append(index_to_id[local_idx_int])
                    
                    if len(target_indices) > 0:
                        inspection = inspect_logits_and_probs(
                            graph_logits, target_indices, input_card_ids,
                            id_to_index, index_to_id, config["model"]["num_cards"], device
                        )
                        print(f"Sample logits stats: {inspection['logits_stats']}")
                        print(f"Sample probs stats: {inspection['probs_stats']}")
                        if inspection['top_predictions']:
                            print(f"Top prediction: {inspection['top_predictions'][0]}")
                        break
            print("=" * 30)
        
        # Save training history
        epoch_history = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        training_history.append(epoch_history)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if save_best:
                save_model(model, model_save_path, {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics
                })
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save training history
    history_path = os.path.join(config["training"]["model_save_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    
    # Load best model if saved
    if save_best and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best model weights")
    
    return model

