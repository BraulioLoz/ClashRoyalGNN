import os
import json
import csv
import time
import logging
import traceback
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
from ..models.graphsage_model import CardRecommendationSAGE
from ..utils import load_config, get_device, save_model, ensure_dir
from .feature_eng_pipeline import create_pyg_data


# ============================================================================
# OPTIMIZATION: Helper functions to reduce code duplication and improve performance
# ============================================================================

def extract_input_card_ids(
    batched_data: Batch,
    node_mask: Optional[torch.Tensor],
    index_to_id: Dict[int, int],
    device: torch.device
) -> List[int]:
    """
    Extract input card IDs from batched data without CPU conversion.
    
    OPTIMIZATION: Avoids .cpu().numpy() conversion in training loop.
    
    Args:
        batched_data: Batched graph data
        node_mask: Mask for nodes in current graph (None for single graph)
        index_to_id: Mapping from index to card ID
        device: Device to compute on
        
    Returns:
        List of input card IDs
    """
    input_card_ids = []
    
    if not hasattr(batched_data, 'input_cards'):
        return input_card_ids
    
    if node_mask is not None:
        # Batched graph: get local input cards
        input_cards = batched_data.input_cards[node_mask]
        # OPTIMIZATION: Stay on GPU, use torch.where instead of .cpu().numpy()
        input_indices = torch.where(input_cards == 1)[0]
        
        # Convert to list without CPU conversion
        for idx_tensor in input_indices:
            local_idx = idx_tensor.item()
            if local_idx in index_to_id:
                input_card_ids.append(index_to_id[local_idx])
    else:
        # Single graph
        input_cards_tensor = batched_data.input_cards
        # OPTIMIZATION: Vectorized check
        input_indices = torch.where(input_cards_tensor == 1)[0]
        for idx_tensor in input_indices:
            idx = idx_tensor.item()
            if idx in index_to_id:
                input_card_ids.append(index_to_id[idx])
    
    return input_card_ids


def create_input_mask(
    input_card_ids: List[int],
    id_to_index: Dict[int, int],
    num_cards: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create mask to exclude input cards (vectorized).
    
    OPTIMIZATION: Vectorized creation instead of loop.
    
    Args:
        input_card_ids: List of input card IDs
        id_to_index: Mapping from card ID to index
        num_cards: Total number of cards
        device: Device to create tensor on
        
    Returns:
        Boolean mask tensor [num_cards], True for valid cards
    """
    # OPTIMIZATION: Vectorized creation instead of loop
    mask = torch.ones(num_cards, device=device, dtype=torch.bool)
    
    if input_card_ids:
        # Get indices in one go
        indices = [
            id_to_index[card_id] 
            for card_id in input_card_ids 
            if card_id in id_to_index and id_to_index[card_id] < num_cards
        ]
        
        if indices:
            # Vectorized assignment
            mask[torch.tensor(indices, device=device, dtype=torch.long)] = False
    
    return mask


def format_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.
    
    Args:
        seconds: Time in seconds (can be float)
        
    Returns:
        Formatted time string as HH:MM:SS
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_batch_examples(
    batched_data: Batch,
    logits: torch.Tensor,
    target_list: List[List[int]],
    id_to_index: Dict[int, int],
    index_to_id: Dict[int, int],
    num_cards: int,
    device: torch.device,
    loss_aggregation: str,
    compute_metrics_flag: bool = False,
    weights: Optional[torch.Tensor] = None
) -> Tuple[float, List[Dict]]:
    """
    Process batched examples to compute weighted loss and metrics.
    Shared logic between train_epoch and validate.
    
    OPTIMIZATION: Eliminates ~150 lines of duplicated code.
    
    Args:
        batched_data: Batched graph data
        logits: Model output logits
        target_list: List of target indices for each example
        id_to_index: Mapping from card ID to index
        index_to_id: Mapping from index to card ID
        num_cards: Total number of cards
        device: Device to compute on
        loss_aggregation: How to aggregate node logits
        compute_metrics_flag: Whether to compute metrics
        weights: Optional tensor of sample weights for win/loss weighting
        
    Returns:
        Tuple of (batch_loss, batch_metrics)
    """
    batch_loss = 0.0
    batch_metrics = []
    total_weight = 0.0
    
    # Check if this is a batched graph or single graph
    if hasattr(batched_data, 'batch') and batched_data.batch is not None:
        # Batched graph: process each graph separately
        batch_size = batched_data.batch.max().item() + 1
        
        for i in range(batch_size):
            # Get nodes belonging to graph i
            node_mask = batched_data.batch == i
            graph_logits = logits[node_mask]
            
            # Get target and input cards
            target_indices = target_list[i] if i < len(target_list) else []
            input_card_ids = extract_input_card_ids(
                batched_data, node_mask, index_to_id, device
            )
            
            # Get sample weight
            sample_weight = weights[i].item() if weights is not None and i < len(weights) else 1.0
            
            # Compute weighted loss
            loss = compute_loss(
                graph_logits, target_indices, input_card_ids,
                id_to_index, num_cards, device, loss_aggregation,
                sample_weight=sample_weight
            )
            batch_loss += loss
            total_weight += sample_weight
            
            # Compute metrics if requested
            if compute_metrics_flag and len(target_indices) > 0:
                metrics = compute_metrics(
                    graph_logits, target_indices, input_card_ids,
                    id_to_index, index_to_id, num_cards, device
                )
                batch_metrics.append(metrics)
        
        # Normalize loss by total weight instead of batch size for proper weighting
        if total_weight > 0:
            batch_loss = batch_loss / total_weight
        else:
            batch_loss = batch_loss / batch_size
    else:
        # Single graph (fallback for batch_size=1)
        target_indices = target_list[0] if target_list else []
        input_card_ids = extract_input_card_ids(
            batched_data, None, index_to_id, device
        )
        
        # Get sample weight
        sample_weight = weights[0].item() if weights is not None and len(weights) > 0 else 1.0
        
        batch_loss = compute_loss(
            logits, target_indices, input_card_ids,
            id_to_index, num_cards, device, loss_aggregation,
            sample_weight=sample_weight
        )
        
        # Compute metrics if requested
        if compute_metrics_flag and len(target_indices) > 0:
            metrics = compute_metrics(
                logits, target_indices, input_card_ids,
                id_to_index, index_to_id, num_cards, device
            )
            batch_metrics.append(metrics)
    
    return batch_loss, batch_metrics

def save_training_history(
    history: List[Dict], 
    json_path: str, 
    csv_path: Optional[str] = None, 
    format_type: str = "json"
):
    """
    Save training history incrementally in JSON and/or CSV format.
    
    Args:
        history: List of training history dictionaries
        json_path: Path to save JSON file
        csv_path: Path to save CSV file (optional)
        format_type: "json", "csv", or "both"
    """
    if not history:
        return
    
    # Guardar JSON
    if format_type in ["json", "both"]:
        json_dir = os.path.dirname(json_path)
        if json_dir:  # Only create dir if path has a directory component
            os.makedirs(json_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    
    # Guardar CSV para fácil visualización
    if format_type in ["csv", "both"] and csv_path:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:  # Only create dir if path has a directory component
            os.makedirs(csv_dir, exist_ok=True)
        # Extraer todas las métricas posibles
        fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
        
        # Agregar métricas dinámicas
        for entry in history:
            if "train_metrics" in entry and entry["train_metrics"]:
                for key in entry["train_metrics"].keys():
                    field_name = f"train_{key}"
                    if field_name not in fieldnames:
                        fieldnames.append(field_name)
            if "val_metrics" in entry and entry["val_metrics"]:
                for key in entry["val_metrics"].keys():
                    field_name = f"val_{key}"
                    if field_name not in fieldnames:
                        fieldnames.append(field_name)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in history:
                row = {
                    "epoch": entry.get("epoch"),
                    "train_loss": entry.get("train_loss"),
                    "val_loss": entry.get("val_loss"),
                    "lr": entry.get("lr")
                }
                
                # Agregar métricas de train
                if "train_metrics" in entry and entry["train_metrics"]:
                    for key, value in entry["train_metrics"].items():
                        row[f"train_{key}"] = value
                
                # Agregar métricas de val
                if "val_metrics" in entry and entry["val_metrics"]:
                    for key, value in entry["val_metrics"].items():
                        row[f"val_{key}"] = value
                
                writer.writerow(row)


class CardDataset(Dataset):
    """Dataset for card recommendation training examples with win/loss weighting."""
    
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
        weight = example.get("weight", 1.0)  # Default weight of 1.0 for legacy data
        
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
        
        # Return data, target, and weight
        return data, target_indices, weight


def collate_fn(batch):
    """Custom collate function for batching PyG Data objects with weights."""
    data_list, target_list, weight_list = zip(*batch)
    
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
    
    # Convert weights to tensor
    weights = torch.tensor(weight_list, dtype=torch.float32)
    
    return batched_data, list(target_list), weights


def compute_loss(
    logits: torch.Tensor, 
    target_indices: List[int], 
    input_card_ids: List[int], 
    id_to_index: Dict, 
    num_cards: int, 
    device: torch.device,
    loss_aggregation: str = "mean",
    sample_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute weighted loss for card recommendation.
    Only compute loss on target cards, excluding input cards.
    
    Args:
        logits: Model output [num_nodes, num_cards]
        target_indices: List of target card indices
        input_card_ids: List of input card IDs
        id_to_index: Mapping from card ID to index
        num_cards: Total number of cards
        device: Device to compute on
        loss_aggregation: How to aggregate node logits ("mean", "max", "input_nodes")
        sample_weight: Weight for this sample (for win/loss weighting)
        
    Returns:
        Weighted loss value
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
    
    # OPTIMIZATION: Use helper function for vectorized mask creation
    mask = create_input_mask(input_card_ids, id_to_index, num_cards, device)
    
    # OPTIMIZATION: Avoid .clone(), use torch.where for efficient masking
    # CRITICAL FIX: Instead of multiplying by 0, set to large negative value
    # This ensures softmax properly excludes input cards without numerical issues
    masked_logits = torch.where(
        mask,
        node_logits,
        torch.tensor(float('-inf'), device=device, dtype=node_logits.dtype)
    )
    
    # OPTIMIZATION: Vectorized target creation
    if not target_indices:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    valid_indices = [idx for idx in target_indices if idx < num_cards]
    if not valid_indices:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Create target tensor vectorized
    target = torch.zeros(num_cards, device=device)
    target[torch.tensor(valid_indices, device=device, dtype=torch.long)] = 1.0
    
    # Apply mask to target and normalize
    masked_target = target * mask.float()
    if masked_target.sum() > 0:
        masked_target = masked_target / masked_target.sum()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute cross-entropy loss
    # Convert to probabilities
    probs = torch.softmax(masked_logits, dim=0)
    
    # Compute negative log likelihood and apply sample weight
    loss = -torch.sum(masked_target * torch.log(probs + 1e-8))
    
    # Apply sample weight (for win/loss weighting)
    weighted_loss = loss * sample_weight
    
    return weighted_loss


def check_gradients(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Check gradient flow in the model by computing gradient norms per layer.
    
    OPTIMIZATION: Reduced .item() calls by computing stats in one pass.
    
    Args:
        model: The model to check
        
    Returns:
        Dictionary with gradient statistics per parameter group
    """
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            # OPTIMIZATION: Compute all stats in one pass, reduce .item() calls
            grad_stats[name] = {
                "norm": grad.norm(2).item(),
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item()
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
    
    # OPTIMIZATION: Use helper function for vectorized mask creation
    mask = create_input_mask(input_card_ids, id_to_index, num_cards, device)
    
    # OPTIMIZATION: Avoid .clone(), use torch.where
    masked_logits = torch.where(
        mask,
        node_logits,
        torch.tensor(float('-inf'), device=device, dtype=node_logits.dtype)
    )
    
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
    
    # OPTIMIZATION: Vectorized target probs extraction
    if target_indices:
        valid_indices = [idx for idx in target_indices if idx < num_cards]
        if valid_indices:
            target_probs = probs[torch.tensor(valid_indices, device=device, dtype=torch.long)]
            metrics["mean_target_prob"] = target_probs.mean().item()
            metrics["min_target_prob"] = target_probs.min().item()
            metrics["max_target_prob"] = target_probs.max().item()
        else:
            metrics["mean_target_prob"] = 0.0
            metrics["min_target_prob"] = 0.0
            metrics["max_target_prob"] = 0.0
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
    
    # OPTIMIZATION: Use helper function for vectorized mask creation
    mask = create_input_mask(input_card_ids, id_to_index, num_cards, device)
    
    # OPTIMIZATION: Avoid .clone(), use torch.where
    masked_logits = torch.where(
        mask,
        node_logits,
        torch.tensor(float('-inf'), device=device, dtype=node_logits.dtype)
    )
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


def load_training_data(config: Dict = None) -> Tuple:
    """
    Load training data, graph, and mappings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_examples, val_examples, node_features, edge_index, edge_attr,
                  id_to_index, index_to_id, num_nodes)
    """
    if config is None:
        config = load_config()
    
    # Load paths from config
    features_dir = config["data"]["features_dir"]
    graph_data_path = os.path.join(features_dir, "graph_data.json")
    examples_path = os.path.join(features_dir, "training_examples.json")
    
    # Load graph data
    with open(graph_data_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    
    node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_data["edge_attr"], dtype=torch.float32)
    id_to_index = {int(k): v for k, v in graph_data["id_to_index"].items()}
    index_to_id = {int(k): v for k, v in graph_data["index_to_id"].items()}
    num_nodes = graph_data["num_nodes"]
    
    # Load training examples
    with open(examples_path, "r", encoding="utf-8") as f:
        training_examples = json.load(f)
    
    # Split train/val
    val_split = config["training"]["val_split"]
    split_idx = int(len(training_examples) * (1 - val_split))
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]
    
    return (train_examples, val_examples, node_features, edge_index, edge_attr,
            id_to_index, index_to_id, num_nodes)


def create_dataloader(
    dataset: CardDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for the CardDataset.
    
    Args:
        dataset: CardDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance
    """
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


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
    compute_metrics_flag: bool = False,
    total_train_time: float = 0.0,
    total_val_time: float = 0.0,
    use_mixed_precision: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, Dict, float]:
    """
    Train for one epoch with optional mixed precision (FP16) training.
    
    OPTIMIZATION: Uses shared process_batch_examples to eliminate code duplication.
    Uses mixed precision and non-blocking transfers for better GPU utilization.
    
    Args:
        total_train_time: Accumulated training time from previous epochs
        total_val_time: Accumulated validation time from previous epochs
        use_mixed_precision: Whether to use FP16 mixed precision training
    
    Returns:
        Tuple of (average_loss, metrics_dict, epoch_time)
    """
    start_time = time.time()
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Mixed precision scaler (only if CUDA available and mixed precision enabled)
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    
    # Aggregate metrics across batches
    all_metrics = defaultdict(list)
    
    # Create tqdm with postfix showing accumulated times
    total_time = total_train_time + total_val_time
    pbar = tqdm(dataloader, desc="Training")
    pbar.set_postfix({
        "Total": format_time(total_time),
        "Train": format_time(total_train_time),
        "Val": format_time(total_val_time)
    })
    
    for batched_data, target_list, weights in pbar:
        # Move data to device with non-blocking transfer (overlaps with GPU computation)
        batched_data = batched_data.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        
        # Forward pass on batched graph
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                logits = model(batched_data)  # [total_nodes_in_batch, num_cards]
                
                # OPTIMIZATION: Use shared processing function to eliminate duplication
                batch_loss, batch_metrics = process_batch_examples(
                    batched_data, logits, target_list,
                    id_to_index, index_to_id, num_cards, device,
                    loss_aggregation, compute_metrics_flag,
                    weights=weights
                )
            
            # Mixed precision backward pass
            scaler.scale(batch_loss).backward()
            
            # Gradient clipping with scaler
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision forward pass
            logits = model(batched_data)  # [total_nodes_in_batch, num_cards]
            
            # OPTIMIZATION: Use shared processing function to eliminate duplication
            batch_loss, batch_metrics = process_batch_examples(
                batched_data, logits, target_list,
                id_to_index, index_to_id, num_cards, device,
                loss_aggregation, compute_metrics_flag,
                weights=weights
            )
            
            batch_loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            
            optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
        # Update progress bar with current loss information
        current_loss = batch_loss.item()
        running_avg_loss = total_loss / num_batches
        pbar.set_postfix({
            "Loss": f"{current_loss:.4f}",
            "Avg": f"{running_avg_loss:.4f}",
            "Total": format_time(total_time),
            "Train": format_time(total_train_time),
            "Val": format_time(total_val_time)
        })
        
        # Aggregate metrics
        for metrics in batch_metrics:
            for key, value in metrics.items():
                all_metrics[key].append(value)
    
    # Compute average metrics
    avg_metrics = {
        key: np.mean(values) if values else 0.0
        for key, values in all_metrics.items()
    }
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    
    # Log warnings if loss is suspicious
    if logger:
        if avg_loss > 10.0:
            logger.warning(f"Very high training loss detected: {avg_loss:.4f}")
        if any(np.isnan(v) for v in avg_metrics.values() if isinstance(v, (int, float))):
            logger.warning("NaN values detected in training metrics")
    
    return avg_loss, avg_metrics, epoch_time


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id_to_index: Dict,
    index_to_id: Dict,
    num_cards: int,
    loss_aggregation: str = "mean",
    compute_metrics_flag: bool = True,
    total_train_time: float = 0.0,
    total_val_time: float = 0.0,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, Dict, float]:
    """
    Validate model.
    
    OPTIMIZATION: Uses shared process_batch_examples to eliminate code duplication.
    
    Args:
        total_train_time: Accumulated training time from previous epochs
        total_val_time: Accumulated validation time from previous epochs
    
    Returns:
        Tuple of (average_loss, metrics_dict, epoch_time)
    """
    start_time = time.time()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Aggregate metrics across batches
    all_metrics = defaultdict(list)
    
    # Create tqdm with postfix showing accumulated times
    total_time = total_train_time + total_val_time
    pbar = tqdm(dataloader, desc="Validating")
    pbar.set_postfix({
        "Total": format_time(total_time),
        "Train": format_time(total_train_time),
        "Val": format_time(total_val_time)
    })
    
    with torch.no_grad():
        for batched_data, target_list, weights in pbar:
            # Move data to device with non-blocking transfer (overlaps with GPU computation)
            batched_data = batched_data.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(batched_data)
            
            # OPTIMIZATION: Use shared processing function to eliminate duplication
            batch_loss, batch_metrics = process_batch_examples(
                batched_data, logits, target_list,
                id_to_index, index_to_id, num_cards, device,
                loss_aggregation, compute_metrics_flag,
                weights=weights
            )
            
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
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    
    # Log warnings if metrics are suspicious
    if logger:
        if avg_loss > 10.0:
            logger.warning(f"Very high validation loss detected: {avg_loss:.4f}")
        if any(np.isnan(v) for v in avg_metrics.values() if isinstance(v, (int, float))):
            logger.warning("NaN values detected in validation metrics")
        if compute_metrics_flag and "top_2_acc" in avg_metrics:
            if avg_metrics["top_2_acc"] < 0.01:
                logger.warning(f"Very low top-2 accuracy: {avg_metrics['top_2_acc']:.4f}")
    
    return avg_loss, avg_metrics, epoch_time


def train_model(
    config: Dict = None,
    graph_data_path: str = None,
    examples_path: str = None,
    model_save_path: str = None,
    logger: Optional[logging.Logger] = None
) -> CardRecommendationGNN:
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        graph_data_path: Path to graph_data.json
        examples_path: Path to training_examples.json
        model_save_path: Path to save trained model
        logger: Optional logger instance for error logging
        
    Returns:
        Trained model
    """
    if config is None:
        config = load_config()
    
    # Setup logger if not provided (backward compatibility)
    if logger is None:
        logger = logging.getLogger('training')
        if not logger.handlers:
            # Create a basic logger if none exists
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
    
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
    
    # OPTIMIZATION: Validate files before processing
    if not os.path.exists(graph_data_path):
        error_msg = f"Graph data not found: {graph_data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    if not os.path.exists(examples_path):
        error_msg = f"Training examples not found: {examples_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load graph data
    print("Loading graph data...")
    logger.info("Loading graph data...")
    try:
        with open(graph_data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        
        node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(graph_data["edge_attr"], dtype=torch.float32)
        id_to_index = {int(k): v for k, v in graph_data["id_to_index"].items()}
        index_to_id = {int(k): v for k, v in graph_data["index_to_id"].items()}
        num_nodes = graph_data["num_nodes"]
        logger.info(f"Loaded graph data: {num_nodes} nodes, {edge_index.size(1)} edges")
    except Exception as e:
        logger.error(f"Error loading graph data: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Load training examples
    print("Loading training examples...")
    logger.info("Loading training examples...")
    try:
        with open(examples_path, "r", encoding="utf-8") as f:
            training_examples = json.load(f)
        logger.info(f"Loaded {len(training_examples)} training examples")
        print(f"Loaded {len(training_examples)} training examples")
    except Exception as e:
        logger.error(f"Error loading training examples: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Split train/val
    val_split = config["training"]["val_split"]
    split_idx = int(len(training_examples) * (1 - val_split))
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]
    
    print(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")
    logger.info(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")
    
    # Create datasets
    try:
        train_dataset = CardDataset(
            train_examples, node_features, edge_index, edge_attr,
            id_to_index, index_to_id, config["model"]["num_cards"]
        )
        val_dataset = CardDataset(
            val_examples, node_features, edge_index, edge_attr,
            id_to_index, index_to_id, config["model"]["num_cards"]
        )
        logger.debug("Datasets created successfully")
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
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
    device_name = str(device)
    if torch.cuda.is_available() and device.type == 'cuda':
        device_name = f"{device} ({torch.cuda.get_device_name(0)})"
    logger.info(f"Using device: {device_name}")
    
    # Initialize model
    print("Initializing model...")
    logger.info("Initializing model...")
    try:
        weight_init = config["model"].get("weight_init", "xavier")
        gnn_type = config["model"]["gnn_type"]
        
        # Choose model based on gnn_type
        if gnn_type == "GraphSAGE":
            aggr = config["model"].get("sage_aggr", "mean")
            model = CardRecommendationSAGE(
                num_nodes=num_nodes,
                node_feature_dim=node_features.size(1),
                hidden_dims=config["model"]["hidden_dims"],
                dropout_rates=config["model"]["dropout_rates"],
                gnn_type=gnn_type,
                num_cards=config["model"]["num_cards"],
                weight_init=weight_init,
                aggr=aggr
            )
            logger.info(f"GraphSAGE model initialized with '{aggr}' aggregation")
            print(f"GraphSAGE model initialized with '{aggr}' aggregation")
        else:
            # Default to GCN
            model = CardRecommendationGNN(
                num_nodes=num_nodes,
                node_feature_dim=node_features.size(1),
                hidden_dims=config["model"]["hidden_dims"],
                dropout_rates=config["model"]["dropout_rates"],
                gnn_type=gnn_type,
                num_cards=config["model"]["num_cards"],
                weight_init=weight_init
            )
            logger.info(f"GCN model initialized")
            print(f"GCN model initialized")
        
        logger.info(f"Model initialized with {weight_init} weight initialization")
        print(f"Model initialized with {weight_init} weight initialization")
        
        model = model.to(device)
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Optimizer
    lr = config["training"]["lr"]
    # Ensure lr is a float (YAML might parse as string)
    lr = float(lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = None
    lr_scheduler_config = config["training"].get("lr_scheduler", {})
    
    factor = float(lr_scheduler_config.get("factor", 0.5))
    patience = int(lr_scheduler_config.get("patience", 5))
    min_lr = float(lr_scheduler_config.get("min_lr", 0.000001)) # 1e-6 en numero sería 0.000001
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )
    # Training parameters
    epochs = config["training"]["epochs"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    save_best = config["training"]["save_best_model"]
    gradient_clip_norm = config["training"].get("gradient_clip_norm", None)
    loss_aggregation = config["model"].get("loss_aggregation", "mean")
    log_gradients_every_n_epochs = config["training"].get("log_gradients_every_n_epochs", 5)
    compute_metrics_flag = config["training"].get("compute_metrics", True)
    
    save_history_every_n_epochs = config["training"].get("save_history_every_n_epochs", 5)
    save_history_format = config["training"].get("save_history_format", "json")
    
    history_path = os.path.join(config["training"]["model_save_dir"], "training_history.json")
    history_csv_path = os.path.join(config["training"]["model_save_dir"], "training_history.csv")
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Learning rate: {lr}, Loss aggregation: {loss_aggregation}")
    if gradient_clip_norm:
        print(f"Gradient clipping: {gradient_clip_norm}")
    if scheduler:
        print(f"LR scheduler: ReduceLROnPlateau")
    if save_history_every_n_epochs > 0:
        print(f"Training history will be saved every {save_history_every_n_epochs} epochs ({save_history_format} format)")
        print(f"  - JSON: {history_path}")
        if save_history_format in ["csv", "both"]:
            print(f"  - CSV: {history_csv_path}")
    
    # Initialize time accumulators
    total_train_time = 0.0
    total_val_time = 0.0
    
    # Mixed precision training
    use_mixed_precision = config["training"].get("use_mixed_precision", False)
    if use_mixed_precision and torch.cuda.is_available():
        print("Mixed precision (FP16) training enabled")
        logger.info("Mixed precision (FP16) training enabled")
    elif use_mixed_precision and not torch.cuda.is_available():
        warning_msg = "Mixed precision requested but CUDA not available, using FP32"
        print(f"Warning: {warning_msg}")
        logger.warning(warning_msg)
        use_mixed_precision = False
    
    # Log training configuration
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Learning rate: {lr}, Batch size: {batch_size}, Loss aggregation: {loss_aggregation}")
    logger.info(f"Device: {device_name}, Mixed precision: {use_mixed_precision}")
    if gradient_clip_norm:
        logger.info(f"Gradient clipping: {gradient_clip_norm}")
    if scheduler:
        logger.info(f"LR scheduler: ReduceLROnPlateau (patience={patience}, factor={factor}, min_lr={min_lr})")
    
    try:
        for epoch in range(epochs):
            try:
                logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                print(f"\nEpoch {epoch + 1}/{epochs}")
                
                # Train
                try:
                    train_loss, train_metrics, train_time = train_epoch(
                        model, train_loader, optimizer, device, id_to_index, index_to_id, 
                        config["model"]["num_cards"],
                        loss_aggregation=loss_aggregation,
                        gradient_clip_norm=gradient_clip_norm,
                        compute_metrics_flag=compute_metrics_flag,
                        total_train_time=total_train_time,
                        total_val_time=total_val_time,
                        use_mixed_precision=use_mixed_precision,
                        logger=logger
                    )
                    total_train_time += train_time
                    logger.debug(f"Epoch {epoch + 1} training completed in {train_time:.2f}s")
                except Exception as e:
                    logger.error(f"ERROR during training epoch {epoch + 1}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                
                # Validate
                try:
                    val_loss, val_metrics, val_time = validate(
                        model, val_loader, device, id_to_index, index_to_id, 
                        config["model"]["num_cards"],
                        loss_aggregation=loss_aggregation,
                        compute_metrics_flag=compute_metrics_flag,
                        total_train_time=total_train_time,
                        total_val_time=total_val_time,
                        logger=logger
                    )
                    total_val_time += val_time
                    logger.debug(f"Epoch {epoch + 1} validation completed in {val_time:.2f}s")
                except Exception as e:
                    logger.error(f"ERROR during validation epoch {epoch + 1}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            except KeyboardInterrupt:
                logger.warning(f"Training interrupted at epoch {epoch + 1}/{epochs}")
                logger.warning(f"Last completed epoch: {epoch}")
                if epoch >= 0:
                    logger.info(f"Last epoch metrics - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
                    logger.info(f"Total training time: {format_time(total_train_time)}")
                    logger.info(f"Total validation time: {format_time(total_val_time)}")
                    logger.info(f"Total time: {format_time(total_train_time + total_val_time)}")
                    # Try to save history before re-raising
                    try:
                        if training_history:
                            save_training_history(
                                training_history, 
                                history_path, 
                                history_csv_path if save_history_format in ["csv", "both"] else None,
                                save_history_format
                            )
                            logger.info("Training history saved before interruption")
                    except Exception as save_error:
                        logger.error(f"Error saving history before interruption: {str(save_error)}")
                raise
            except Exception as e:
                logger.error(f"FATAL ERROR at epoch {epoch + 1}/{epochs}")
                logger.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())
                if epoch >= 0:
                    logger.error(f"Last completed epoch: {epoch}")
                    logger.error(f"Best validation loss so far: {best_val_loss:.4f}")
                    logger.error(f"Total training time: {format_time(total_train_time)}")
                    logger.error(f"Total validation time: {format_time(total_val_time)}")
                raise
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    lr_change_msg = f"Learning rate changed: {current_lr:.6f} -> {new_lr:.6f}"
                    print(lr_change_msg)
                    logger.info(lr_change_msg)
            
            # Log metrics
            log_str = f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
            if val_metrics and "top_2_acc" in val_metrics:
                log_str += f", Val Top-2 Acc: {val_metrics['top_2_acc']:.4f}"
            print(log_str)
            logger.info(f"Epoch {epoch + 1}: {log_str}")
            
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
                        
                        # OPTIMIZATION: Use helper function to avoid .cpu().numpy()
                        input_card_ids = extract_input_card_ids(
                            batched_data, node_mask, index_to_id, device
                        )
                        
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
            
            if save_history_every_n_epochs > 0 and ((epoch + 1) % save_history_every_n_epochs == 0 or epoch == 0):
                save_training_history(
                    training_history, 
                    history_path, 
                    history_csv_path if save_history_format in ["csv", "both"] else None,
                    save_history_format
                )
                saved_formats = []
                if save_history_format in ["json", "both"]:
                    saved_formats.append("JSON")
                if save_history_format in ["csv", "both"]:
                    saved_formats.append("CSV")
                history_save_msg = f"Training history saved (epoch {epoch + 1}/{epochs}) - {', '.join(saved_formats)}"
                print(history_save_msg)
                logger.info(history_save_msg)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_best:
                    try:
                        save_model(model, model_save_path, {
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "train_metrics": train_metrics,
                            "val_metrics": val_metrics
                        })
                        checkpoint_msg = f"Saved best model (val_loss: {val_loss:.4f})"
                        print(checkpoint_msg)
                        logger.info(checkpoint_msg)
                    except Exception as e:
                        logger.error(f"Error saving best model: {str(e)}")
                        logger.error(traceback.format_exc())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    early_stop_msg = f"Early stopping triggered after {epoch + 1} epochs (patience: {early_stopping_patience})"
                    print(early_stop_msg)
                    logger.warning(early_stop_msg)
                    logger.info(f"Best validation loss: {best_val_loss:.4f}")
                    break
        
        # Save training history (final save)
        save_training_history(
            training_history, 
            history_path, 
            history_csv_path if save_history_format in ["csv", "both"] else None,
            save_history_format
        )
        
        # Print summary of saved files
        saved_files = [history_path]
        if save_history_format in ["csv", "both"]:
            saved_files.append(history_csv_path)
        
        print(f"\nTraining history saved to:")
        for file_path in saved_files:
            if os.path.exists(file_path):
                print(f"  - {file_path}")
        
        completion_msg = f"Training complete. Best validation loss: {best_val_loss:.4f}"
        print(f"\n{completion_msg}")
        logger.info(completion_msg)
        logger.info(f"Total training time: {format_time(total_train_time)}")
        logger.info(f"Total validation time: {format_time(total_val_time)}")
        logger.info(f"Total time: {format_time(total_train_time + total_val_time)}")
        print(f"Total training time: {format_time(total_train_time)}")
        print(f"Total validation time: {format_time(total_val_time)}")
        print(f"Total time: {format_time(total_train_time + total_val_time)}")
        
        # Load best model if saved
        if save_best and os.path.exists(model_save_path):
            try:
                checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded best model weights")
                logger.info("Loaded best model weights")
            except Exception as e:
                logger.error(f"Error loading best model: {str(e)}")
                logger.error(traceback.format_exc())
    
    except KeyboardInterrupt:
        # This should be caught in the inner try-except, but just in case
        logger.warning("Training interrupted by user (outer catch)")
        raise
    except Exception as e:
        logger.error(f"FATAL ERROR in train_model (outer catch): {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    return model

