import os
import yaml
import torch
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def load_config(config_path: str = None) -> Dict:
    """
    Load and parse config.yaml file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Default to config/config.yaml relative to project root
        # __file__ is now in src/utils/__init__.py, so go up 2 levels
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Auto-detect GPU availability and return appropriate device.
    
    Args:
        device_preference: "auto", "cuda", or "cpu"
        
    Returns:
        torch.device object
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA requested but not available")
    else:
        device = torch.device("cpu")
    
    return device


def get_card_id_mapping(cards_data: Dict) -> Dict[str, int]:
    """
    Map card names to IDs from cards API response.
    
    Args:
        cards_data: Dictionary from get_cards() API call
        
    Returns:
        Dictionary mapping card name to card ID
    """
    card_mapping = {}
    
    # Process items
    items = cards_data.get("items", [])
    for item in items:
        card_id = item.get("id")
        name_data = item.get("name", {})
        if isinstance(name_data, dict):
            # Get English name or first available name
            card_name = name_data.get("en") or name_data.get(list(name_data.keys())[0])
        else:
            card_name = name_data
        
        if card_id is not None and card_name:
            card_mapping[card_name] = card_id
    
    # Process supportItems
    support_items = cards_data.get("supportItems", [])
    for item in support_items:
        card_id = item.get("id")
        name_data = item.get("name", {})
        if isinstance(name_data, dict):
            card_name = name_data.get("en") or name_data.get(list(name_data.keys())[0])
        else:
            card_name = name_data
        
        if card_id is not None and card_name:
            card_mapping[card_name] = card_id
    
    return card_mapping


def get_card_id_to_index_mapping(cards_data: Dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create mappings between card IDs and indices for graph nodes.
    
    Args:
        cards_data: Dictionary from get_cards() API call
        
    Returns:
        Tuple of (id_to_index, index_to_id) dictionaries
    """
    all_card_ids = set()
    
    # Collect all card IDs
    items = cards_data.get("items", [])
    for item in items:
        card_id = item.get("id")
        if card_id is not None:
            all_card_ids.add(card_id)
    
    support_items = cards_data.get("supportItems", [])
    for item in support_items:
        card_id = item.get("id")
        if card_id is not None:
            all_card_ids.add(card_id)
    
    # Sort for consistent indexing
    sorted_ids = sorted(all_card_ids)
    
    id_to_index = {card_id: idx for idx, card_id in enumerate(sorted_ids)}
    index_to_id = {idx: card_id for card_id, idx in id_to_index.items()}
    
    return id_to_index, index_to_id


def save_model(model: torch.nn.Module, path: str, metadata: Dict = None):
    """
    Save PyTorch model with optional metadata.
    
    Args:
        model: PyTorch model to save
        path: Path to save model
        metadata: Optional dictionary with additional metadata
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__
    }
    
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(path: str, model_class: torch.nn.Module = None) -> Dict:
    """
    Load PyTorch model state dict.
    
    Args:
        path: Path to saved model
        model_class: Optional model class (for full model loading)
        
    Returns:
        Dictionary with model state dict and metadata
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False) 
    
    if model_class:
        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
        return {"model": model, "metadata": checkpoint.get("metadata", {})}
    else:
        return {
            "state_dict": checkpoint["model_state_dict"],
            "metadata": checkpoint.get("metadata", {})
        }


def ensure_dir(path: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def setup_training_logger(log_dir: str = "models", log_file: str = "training_errors.log") -> logging.Logger:
    """
    Setup logger for training that writes to both console and file.
    
    The logger is thread-safe and can be used with DataLoader multiprocessing.
    File handler captures all levels (DEBUG and above), console handler only shows
    WARNING and above to avoid saturating stdout.
    
    Args:
        log_dir: Directory to save log file
        log_file: Name of log file
        
    Returns:
        Logger instance configured for training
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger with unique name to avoid conflicts
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates when called multiple times
    logger.handlers = []
    
    # File handler (captures everything from DEBUG and above)
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (only warnings and above to avoid saturating stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

