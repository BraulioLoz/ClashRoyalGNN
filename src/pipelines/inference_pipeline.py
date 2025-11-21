import os
import json
import torch
from typing import List, Tuple, Dict

from ..models.gnn_model import CardRecommendationGNN
from ..utils import load_config, get_device, load_model
from .feature_eng_pipeline import create_pyg_data


def load_inference_model(
    model_path: str,
    config: Dict = None,
    device_preference: str = "auto"
) -> Tuple[CardRecommendationGNN, torch.device]:
    """
    Load trained model for inference.
    
    Args:
        model_path: Path to saved model
        config: Configuration dictionary
        device_preference: Device preference ("auto", "cuda", "cpu")
        
    Returns:
        Tuple of (model, device)
    """
    if config is None:
        config = load_config()
    
    # Get device
    device = get_device(device_preference)
    
    # Load graph data to get dimensions
    features_dir = config["data"]["features_dir"]
    graph_data_path = os.path.join(features_dir, "graph_data.json")
    
    with open(graph_data_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    
    num_nodes = graph_data["num_nodes"]
    node_feature_dim = len(graph_data["node_features"][0])
    
    # Initialize model
    model = CardRecommendationGNN(
        num_nodes=num_nodes,
        node_feature_dim=node_feature_dim,
        hidden_dims=config["model"]["hidden_dims"],
        dropout_rates=config["model"]["dropout_rates"],
        gnn_type=config["model"]["gnn_type"],
        num_cards=config["model"]["num_cards"]
    )
    
    # Load weights
    checkpoint = load_model(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Using device: {device}")
    
    return model, device


def load_graph_data(config: Dict = None) -> Dict:
    """
    Load graph data for inference.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with graph data
    """
    if config is None:
        config = load_config()
    
    features_dir = config["data"]["features_dir"]
    graph_data_path = os.path.join(features_dir, "graph_data.json")
    
    with open(graph_data_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    
    return graph_data


def predict_cards(
    model: CardRecommendationGNN,
    input_card_ids: List[int],
    graph_data: Dict,
    device: torch.device,
    top_k: int = 2,
    exclude_input: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Predict recommended cards given input cards.
    
    Args:
        model: Trained GNN model
        input_card_ids: List of 6 input card IDs
        graph_data: Graph data dictionary
        device: Device to run inference on
        top_k: Number of cards to recommend
        exclude_input: Whether to exclude input cards from predictions
        
    Returns:
        Tuple of (recommended_card_ids, probabilities)
    """
    # Convert graph data to tensors
    node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32).to(device)
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long).to(device)
    edge_attr = torch.tensor(graph_data["edge_attr"], dtype=torch.float32).to(device)
    
    id_to_index = {int(k): v for k, v in graph_data["id_to_index"].items()}
    index_to_id = {int(k): v for k, v in graph_data["index_to_id"].items()}
    
    # Create PyG Data object
    data = create_pyg_data(
        node_features,
        edge_index,
        edge_attr,
        input_card_ids,
        id_to_index,
        index_to_id
    )
    data = data.to(device)
    
    # Make prediction
    recommended_ids, probabilities = model.predict_cards(
        data,
        input_card_ids,
        top_k=top_k,
        exclude_input=exclude_input
    )
    
    return recommended_ids, probabilities


def run_inference(
    input_card_ids: List[int],
    model_path: str = None,
    config: Dict = None,
    device_preference: str = "auto"
) -> Dict:
    """
    Main inference function.
    
    Args:
        input_card_ids: List of 6 input card IDs
        model_path: Path to trained model
        config: Configuration dictionary
        device_preference: Device preference
        
    Returns:
        Dictionary with predictions
    """
    if config is None:
        config = load_config()
    
    if model_path is None:
        model_save_dir = config["training"]["model_save_dir"]
        model_path = os.path.join(model_save_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load model
    print("Loading model...")
    model, device = load_inference_model(model_path, config, device_preference)
    
    # Load graph data
    print("Loading graph data...")
    graph_data = load_graph_data(config)
    
    # Make prediction
    print(f"Predicting cards for input: {input_card_ids}")
    recommended_ids, probabilities = predict_cards(
        model,
        input_card_ids,
        graph_data,
        device,
        top_k=2,
        exclude_input=True
    )
    
    result = {
        "input_cards": input_card_ids,
        "recommended_cards": recommended_ids,
        "probabilities": probabilities
    }
    
    print(f"Recommended cards: {recommended_ids}")
    print(f"Probabilities: {probabilities}")
    
    return result

