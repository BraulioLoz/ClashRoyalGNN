import os
import json
import torch
from typing import List, Tuple, Dict

from ..models.gnn_model import CardRecommendationGNN
from ..models.graphsage_model import CardRecommendationSAGE
from ..models.pretrained_sage import CardRecommendationSAGEWithTransfer
from ..utils import load_config, get_device, load_model
from .feature_eng_pipeline import create_pyg_data


def load_inference_model(
    model_path: str,
    config: Dict = None,
    device_preference: str = "auto"
) -> Tuple[torch.nn.Module, torch.device]:
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
    
    # Load checkpoint to detect model type
    checkpoint = load_model(model_path)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", {}))
    
    # Detect if this is a transfer learning model
    is_transfer_model = any("feature_adapter" in key or "pretrained_encoder" in key for key in state_dict.keys())
    
    aggr = config["model"].get("sage_aggr", "mean")
    
    if is_transfer_model:
        # Transfer learning model - detect architecture from state_dict
        # Count encoder layers
        encoder_layers = len([k for k in state_dict.keys() if "pretrained_encoder.convs" in k and "lin_l.weight" in k])
        # Count finetune layers
        finetune_layers = len([k for k in state_dict.keys() if "finetune_layers" in k and "lin_l.weight" in k])
        
        # Infer hidden_dims from state_dict shapes
        hidden_dims = []
        for i in range(encoder_layers):
            key = f"pretrained_encoder.convs.{i}.lin_l.weight"
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])
        
        # Infer finetune_dims from state_dict shapes
        finetune_dims = []
        for i in range(finetune_layers):
            key = f"finetune_layers.{i}.lin_l.weight"
            if key in state_dict:
                finetune_dims.append(state_dict[key].shape[0])
        
        # Get pretrained_dim from feature adapter output
        pretrained_dim = state_dict.get("feature_adapter.proj2.weight", torch.zeros(128, 1)).shape[0]
        
        model = CardRecommendationSAGEWithTransfer(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            pretrained_dim=pretrained_dim,
            hidden_dims=hidden_dims if hidden_dims else [512, 256, 128],
            finetune_dims=finetune_dims if finetune_dims else [128, 64],
            dropout_rates=config["model"]["dropout_rates"],
            num_cards=config["model"]["num_cards"],
            aggr=aggr,
            use_pretrained=False
        )
        model.load_state_dict(state_dict)
        print("Loaded transfer learning model")
    elif config["model"]["gnn_type"] == "GraphSAGE":
        model = CardRecommendationSAGE(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            hidden_dims=config["model"]["hidden_dims"],
            dropout_rates=config["model"]["dropout_rates"],
            gnn_type=config["model"]["gnn_type"],
            num_cards=config["model"]["num_cards"],
            aggr=aggr
        )
        model.load_state_dict(state_dict)
    else:
        model = CardRecommendationGNN(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            hidden_dims=config["model"]["hidden_dims"],
            dropout_rates=config["model"]["dropout_rates"],
            gnn_type=config["model"]["gnn_type"],
            num_cards=config["model"]["num_cards"]
        )
        model.load_state_dict(state_dict)
    
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
    device_preference: str = "auto",
    card_mapper = None
) -> Dict:
    """
    Main inference function.
    
    Args:
        input_card_ids: List of 6 input card IDs
        model_path: Path to trained model
        config: Configuration dictionary
        device_preference: Device preference
        card_mapper: Optional CardMapper instance for name conversion
        
    Returns:
        Dictionary with predictions (includes card names if mapper provided)
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
    
    # Convert IDs to names if mapper is provided
    if card_mapper is not None:
        try:
            input_card_names = card_mapper.batch_id_to_name(input_card_ids)
            recommended_card_names = card_mapper.batch_id_to_name(recommended_ids)
            
            result["input_card_names"] = input_card_names
            result["recommended_card_names"] = recommended_card_names
            
            print(f"Recommended cards: {recommended_card_names}")
            print(f"Probabilities: {probabilities}")
        except ValueError as e:
            print(f"Warning: Could not convert IDs to names: {e}")
            print(f"Recommended cards: {recommended_ids}")
            print(f"Probabilities: {probabilities}")
    else:
        print(f"Recommended cards: {recommended_ids}")
        print(f"Probabilities: {probabilities}")
    
    return result

