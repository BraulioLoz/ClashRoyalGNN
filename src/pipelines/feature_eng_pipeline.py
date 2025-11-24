import os
import json
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional

from ..utils import get_card_id_to_index_mapping, load_config
from ..data_collection.data_fetcher import extract_decks_from_battle_logs


def encode_rarity(rarity: str) -> int:
    """Encode rarity as integer."""
    rarity_map = {
        "COMMON": 0,
        "RARE": 1,
        "EPIC": 2,
        "LEGENDARY": 3,
        "CHAMPION": 4
    }
    return rarity_map.get(rarity.upper(), 0)


def extract_node_features(
    cards_data: Dict, 
    node_features: List[str], 
    id_to_index: Dict[int, int],
    normalize: bool = True,
    normalization_method: str = "standard"
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Extract node features for all cards.
    
    Args:
        cards_data: Dictionary from get_cards() API call
        node_features: List of feature names to extract
        id_to_index: Mapping from card ID to node index
        normalize: Whether to normalize features
        normalization_method: "standard" (zero mean, unit variance) or "minmax" (0-1 range)
        
    Returns:
        Tuple of (feature_tensor, normalization_params_dict)
        normalization_params_dict contains mean/std or min/max for each feature
    """
    num_nodes = len(id_to_index)
    
    # OPTIMIZACIÓN 1: Combinar items y supportItems en un solo loop
    card_data_map = {}
    for item_list in [cards_data.get("items", []), cards_data.get("supportItems", [])]:
        for item in item_list:
            card_id = item.get("id")
            if card_id is not None:
                card_data_map[card_id] = item
    
    # OPTIMIZACIÓN 2: Pre-allocar tensor en lugar de lista de listas
    feature_tensor = torch.zeros((num_nodes, len(node_features)), dtype=torch.float32)
    
    # OPTIMIZACIÓN 3: Crear mapeo de feature name a función extractora
    feature_extractors = {
        "id": lambda card_id, card_data: float(card_id),
        "elixirCost": lambda card_id, card_data: float(card_data.get("elixirCost", 0)),
        "rarity": lambda card_id, card_data: float(encode_rarity(card_data.get("rarity", "COMMON"))),
        "maxLevel": lambda card_id, card_data: float(card_data.get("maxLevel", 1)),
        "maxEvolutionLevel": lambda card_id, card_data: float(card_data.get("maxEvolutionLevel", 0)),
    }
    
    # OPTIMIZACIÓN 4: Iterar por índice en lugar de ordenar
    # Crear reverse mapping: index -> card_id
    index_to_card_id = {idx: card_id for card_id, idx in id_to_index.items()}
    
    # Verificar que tenemos mapeo completo y consecutivo
    if len(index_to_card_id) != num_nodes:
        raise ValueError(f"Mismatch between num_nodes ({num_nodes}) and id_to_index mapping ({len(index_to_card_id)})")
    
    # Verificar que los índices son consecutivos desde 0
    expected_indices = set(range(num_nodes))
    actual_indices = set(index_to_card_id.keys())
    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        raise ValueError(f"Non-consecutive indices: missing {missing}, extra {extra}")
    
    for idx in range(num_nodes):
        card_id = index_to_card_id[idx]
        card_data = card_data_map.get(card_id, {})
        
        for feat_idx, feat_name in enumerate(node_features):
            extractor = feature_extractors.get(feat_name)
            if extractor:
                feature_tensor[idx, feat_idx] = extractor(card_id, card_data)
            else:
                feature_tensor[idx, feat_idx] = 0.0
    
    # OPTIMIZACIÓN 5: Normalización vectorizada
    normalization_params = None
    
    if normalize and feature_tensor.size(0) > 0:
        normalization_params = {}
        
        if normalization_method == "standard":
            # Vectorized: compute mean and std for all features at once
            mean = feature_tensor.mean(dim=0, keepdim=True)
            std = feature_tensor.std(dim=0, keepdim=True)
            
            # Avoid division by zero
            std = torch.clamp(std, min=1e-8)
            feature_tensor = (feature_tensor - mean) / std
            
            # Store params
            for feat_idx, feat_name in enumerate(node_features):
                normalization_params[feat_name] = {
                    "method": "standard",
                    "mean": mean[0, feat_idx].item(),
                    "std": std[0, feat_idx].item()
                }
                
        elif normalization_method == "minmax":
            # Vectorized: compute min and max for all features at once
            min_vals = feature_tensor.min(dim=0, keepdim=True)[0]
            max_vals = feature_tensor.max(dim=0, keepdim=True)[0]
            
            # Avoid division by zero
            ranges = max_vals - min_vals
            ranges = torch.clamp(ranges, min=1e-8)
            feature_tensor = (feature_tensor - min_vals) / ranges
            
            # Store params
            for feat_idx, feat_name in enumerate(node_features):
                normalization_params[feat_name] = {
                    "method": "minmax",
                    "min": min_vals[0, feat_idx].item(),
                    "max": max_vals[0, feat_idx].item()
                }
        
        # OPTIMIZACIÓN 6: Reducir prints en producción (opcional)
        if __debug__:  # Solo en modo debug
            print(f"Normalized features using {normalization_method} method")
            for feat_idx, feat_name in enumerate(node_features):
                feat_col = feature_tensor[:, feat_idx]
                print(f"  {feat_name}: mean={feat_col.mean().item():.4f}, "
                      f"std={feat_col.std().item():.4f}")
    
    return feature_tensor, normalization_params


def build_graph_from_co_occurrence(
    co_occurrence_data: Dict,
    id_to_index: Dict[int, int],
    edge_threshold: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph edge_index and edge_attr from co-occurrence matrix.
    
    Args:
        co_occurrence_data: Dictionary with edge_list from co-occurrence matrix
        id_to_index: Mapping from card ID to node index
        edge_threshold: Minimum weight for edges (applies additional filtering if needed)
        
    Returns:
        Tuple of (edge_index, edge_weights)
    """
    edge_list = co_occurrence_data.get("edge_list", [])
    
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    
    # OPTIMIZACIÓN: Pre-filtrar y usar list comprehensions más eficientes
    valid_edges = []
    valid_weights = []
    
    for edge in edge_list:
        source_id = edge.get("source")
        target_id = edge.get("target")
        weight = edge.get("weight", 1)
        
        # Apply edge_threshold filter (defense in depth)
        if weight < edge_threshold:
            continue
        
        if source_id in id_to_index and target_id in id_to_index:
            source_idx = id_to_index[source_id]
            target_idx = id_to_index[target_id]
            
            # Skip self-loops
            if source_idx == target_idx:
                continue
            
            # Add both directions for undirected graph
            valid_edges.extend([(source_idx, target_idx), (target_idx, source_idx)])
            valid_weights.extend([weight, weight])
    
    if not valid_edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    
    # OPTIMIZACIÓN: Crear tensor directamente desde lista de tuples
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(valid_weights, dtype=torch.float32)
    
    return edge_index, edge_attr


def create_training_examples(
    decks: List[Dict],
    id_to_index: Dict[int, int],
    index_to_id: Dict[int, int]
) -> List[Dict]:
    """
    Create training examples from decks.
    For each deck of 8 cards, create examples with 6 input cards and 2 target cards.
    
    Args:
        decks: List of deck dictionaries with card IDs
        id_to_index: Mapping from card ID to node index
        index_to_id: Mapping from node index to card ID
        
    Returns:
        List of training examples
    """
    examples = []
    
    for deck in decks:
        card_ids = deck.get("cards", [])
        
        # Filter valid card IDs
        valid_card_ids = [cid for cid in card_ids if cid in id_to_index]
        
        if len(valid_card_ids) != 8:
            continue
        
        # Create multiple examples by selecting different combinations of 6 cards
        # For simplicity, we'll create one example per deck
        # You could create more by selecting different 6-card combinations
        
        # Use first 6 as input, last 2 as target
        input_cards = valid_card_ids[:6]
        target_cards = valid_card_ids[6:8]
        
        examples.append({
            "input_cards": input_cards,
            "target_cards": target_cards,
            "deck": valid_card_ids
        })
        
        # Also create example with last 6 as input, first 2 as target
        input_cards_alt = valid_card_ids[2:8]
        target_cards_alt = valid_card_ids[0:2]
        
        examples.append({
            "input_cards": input_cards_alt,
            "target_cards": target_cards_alt,
            "deck": valid_card_ids
        })
    
    print(f"Created {len(examples)} training examples from {len(decks)} decks")
    return examples


def create_pyg_data(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    input_card_ids: List[int],
    id_to_index: Dict[int, int],
    index_to_id: Dict[int, int]
) -> Data:
    """
    Create PyTorch Geometric Data object.
    
    Args:
        node_features: Node feature tensor [num_nodes, feature_dim]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge weights [num_edges]
        input_card_ids: List of input card IDs
        id_to_index: Mapping from card ID to node index
        index_to_id: Mapping from node index to card ID
        
    Returns:
        PyTorch Geometric Data object
    """
    num_nodes = node_features.size(0)
    
    # Create binary indicator for input cards
    input_cards = torch.zeros(num_nodes, dtype=torch.long)
    
    # OPTIMIZACIÓN: Más explícito y seguro - evitar KeyError
    indices = []
    for card_id in input_card_ids:
        if card_id in id_to_index:
            indices.append(id_to_index[card_id])
    
    if indices:
        input_cards[indices] = 1
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        input_cards=input_cards,
        card_id_to_index=id_to_index,
        index_to_card_id=index_to_id
    )
    
    return data


def process_features(
    config: Dict,
    cards_data_path: str = None,
    co_occurrence_path: str = None,
    decks_path: str = None,
    output_dir: str = None
) -> Tuple[Data, List[Dict], Dict, Dict]:
    """
    Main function to process features and create graph structure.
    
    Args:
        config: Configuration dictionary
        cards_data_path: Path to cards.json file
        co_occurrence_path: Path to co_occurrence_matrix.json file
        decks_path: Path to decks data (or will extract from battle logs)
        output_dir: Directory to save processed features
        
    Returns:
        Tuple of (graph_data, training_examples, id_to_index, index_to_id)
    """
    # Load paths from config if not provided
    if cards_data_path is None:
        raw_dir = config["data"]["raw_dir"]
        cards_data_path = os.path.join(raw_dir, "cards.json")
    
    if co_occurrence_path is None:
        processed_dir = config["data"]["processed_dir"]
        co_occurrence_path = os.path.join(processed_dir, "co_occurrence_matrix.json")
    
    if output_dir is None:
        output_dir = config["data"]["features_dir"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # OPTIMIZACIÓN: Validar archivos antes de procesar
    if not os.path.exists(cards_data_path):
        raise FileNotFoundError(f"Cards data not found: {cards_data_path}")
    
    if not os.path.exists(co_occurrence_path):
        raise FileNotFoundError(f"Co-occurrence matrix not found: {co_occurrence_path}")
    
    # Load cards data
    print("Loading cards data...")
    with open(cards_data_path, "r", encoding="utf-8") as f:
        cards_data = json.load(f)
    
    # Create card ID to index mapping
    id_to_index, index_to_id = get_card_id_to_index_mapping(cards_data)
    num_cards = len(id_to_index)
    print(f"Found {num_cards} unique cards")
    
    # Extract node features
    print("Extracting node features...")
    node_features_list = config["graph"]["node_features"]
    normalize_features = config["graph"].get("normalize_features", True)
    normalization_method = config["graph"].get("normalization_method", "standard")
    
    node_features, normalization_params = extract_node_features(
        cards_data, node_features_list, id_to_index,
        normalize=normalize_features,
        normalization_method=normalization_method
    )
    print(f"Node features shape: {node_features.shape}")
    
    # Load co-occurrence matrix
    print("Loading co-occurrence matrix...")
    with open(co_occurrence_path, "r", encoding="utf-8") as f:
        co_occurrence_data = json.load(f)
    
    # Build graph structure
    print("Building graph structure...")
    edge_threshold = config["graph"]["edge_threshold"]
    edge_index, edge_attr = build_graph_from_co_occurrence(
        co_occurrence_data, id_to_index, edge_threshold
    )
    print(f"Graph has {edge_index.size(1)} edges")
    
    # Load or extract decks
    if decks_path and os.path.exists(decks_path):
        print("Loading decks from file...")
        with open(decks_path, "r", encoding="utf-8") as f:
            decks = json.load(f)
    else:
        # Extract decks from battle logs
        print("Extracting decks from battle logs...")
        
        battle_logs_path = os.path.join(config["data"]["raw_dir"], "battle_logs.json")
        if os.path.exists(battle_logs_path):
            with open(battle_logs_path, "r", encoding="utf-8") as f:
                battle_logs = json.load(f)
            decks = extract_decks_from_battle_logs(battle_logs)
        else:
            print("Warning: No battle logs found. Creating empty deck list.")
            decks = []
    
    # Create training examples
    print("Creating training examples...")
    training_examples = create_training_examples(decks, id_to_index, index_to_id)
    
    # Save processed data
    print("Saving processed features...")
    
    # Save graph structure
    # OPTIMIZACIÓN: Usar .cpu().tolist() en lugar de .numpy().tolist() para mejor compatibilidad
    graph_data = {
        "node_features": node_features.cpu().tolist(),
        "edge_index": edge_index.cpu().tolist(),
        "edge_attr": edge_attr.cpu().tolist(),
        "id_to_index": {str(k): v for k, v in id_to_index.items()},
        "index_to_id": {str(k): v for k, v in index_to_id.items()},
        "num_nodes": num_cards,
        "normalization_params": normalization_params if normalization_params else None
    }
    
    graph_path = os.path.join(output_dir, "graph_data.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    
    # Save training examples
    examples_path = os.path.join(output_dir, "training_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(training_examples, f, indent=2)
    
    # Create a sample PyG Data object for reference
    sample_data = create_pyg_data(
        node_features, edge_index, edge_attr, [], id_to_index, index_to_id
    )
    
    print(f"Feature engineering complete. Saved to {output_dir}")
    
    return sample_data, training_examples, id_to_index, index_to_id

