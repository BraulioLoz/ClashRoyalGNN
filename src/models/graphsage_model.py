import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class CardRecommendationSAGE(nn.Module):
    """
    Graph Neural Network using GraphSAGE for Clash Royale card recommendation.
    
    Takes a graph with all cards as nodes and recommends 2 cards given 6 input cards.
    Uses GraphSAGE layers with neighbor sampling and aggregation.
    
    GraphSAGE advantages over GCN:
    - Samples neighbors instead of using all (more scalable)
    - Learned aggregation function (more expressive)
    - Better at handling heterogeneous relationships
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        hidden_dims: list,
        dropout_rates: list,
        gnn_type: str = "GraphSAGE",
        num_cards: int = 110,
        weight_init: str = "xavier",
        aggr: str = "mean"  # GraphSAGE aggregation: 'mean', 'max', 'lstm'
    ):
        """
        Initialize the GraphSAGE model.
        
        Args:
            num_nodes: Number of nodes in the graph (total cards)
            node_feature_dim: Dimension of node features
            hidden_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates for each layer
            gnn_type: Type of GNN layer (default: "GraphSAGE")
            num_cards: Total number of cards (for output layer)
            weight_init: Weight initialization method (default: "xavier")
            aggr: Aggregation method for GraphSAGE ('mean', 'max', 'lstm')
        """
        super(CardRecommendationSAGE, self).__init__()
        
        self.num_cards = num_cards
        self.aggr = aggr
        
        # Add binary indicator for input cards (1 if card is in input deck, 0 otherwise)
        input_dim = node_feature_dim + 1
        
        # Build GraphSAGE layers
        self.gnn_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            # SAGEConv with specified aggregation
            self.gnn_layers.append(SAGEConv(in_dim, out_dim, aggr=aggr))
            
            if i < len(dropout_rates):
                self.dropouts.append(nn.Dropout(dropout_rates[i]))
        
        # Output layer: predict probability for each card
        self.output_layer = nn.Linear(hidden_dims[-1], num_cards)
        
        # Initialize weights with Xavier
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights with Xavier uniform initialization.
        GraphSAGE layers use smaller std for output layer stability.
        """
        # Initialize GraphSAGE layers with Xavier
        for gnn_layer in self.gnn_layers:
            if hasattr(gnn_layer, 'lin_l') and gnn_layer.lin_l is not None:
                nn.init.xavier_uniform_(gnn_layer.lin_l.weight)
                if gnn_layer.lin_l.bias is not None:
                    nn.init.constant_(gnn_layer.lin_l.bias, 0)
            if hasattr(gnn_layer, 'lin_r') and gnn_layer.lin_r is not None:
                nn.init.xavier_uniform_(gnn_layer.lin_r.weight)
                if gnn_layer.lin_r.bias is not None:
                    nn.init.constant_(gnn_layer.lin_r.bias, 0)
        
        # Initialize output layer with smaller weights for stability
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GraphSAGE network.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - input_cards: Binary indicator [num_nodes] (1 if card is in input deck)
                
        Returns:
            Logits for each card [num_nodes, num_cards]
        """
        x = data.x
        edge_index = data.edge_index
        
        # Add binary indicator for input cards
        input_indicator = getattr(data, 'input_cards', torch.zeros(x.size(0), device=x.device))
        input_indicator = input_indicator.unsqueeze(1).float()
        
        # Concatenate node features with input indicator
        x = torch.cat([x, input_indicator], dim=1)
        
        # Pass through GraphSAGE layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # Output layer: predict for each node
        logits = self.output_layer(x)
        
        return logits
    
    def predict_cards(
        self,
        data: Data,
        input_card_ids: list,
        top_k: int = 2,
        exclude_input: bool = True
    ) -> tuple:
        """
        Predict top K cards given input cards.
        
        Args:
            data: PyTorch Geometric Data object
            input_card_ids: List of card IDs in the input deck
            top_k: Number of cards to recommend
            exclude_input: Whether to exclude input cards from predictions
            
        Returns:
            Tuple of (recommended_card_ids, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)  # [num_nodes, num_cards]
            
            # Aggregate logits across nodes
            node_logits = logits.mean(dim=0)  # [num_cards]
            
            # Apply softmax to get probabilities
            probs = F.softmax(node_logits, dim=0)
            
            # Exclude input cards if requested
            if exclude_input:
                card_id_to_index = getattr(data, 'card_id_to_index', None)
                if card_id_to_index is not None:
                    # Create mask more efficiently
                    mask = torch.ones(self.num_cards, device=logits.device, dtype=torch.bool)
                    
                    # Set mask to False for input cards
                    for card_id in input_card_ids:
                        idx = card_id_to_index.get(card_id)
                        if idx is not None and idx < self.num_cards:
                            mask[idx] = False
                    
                    # Set excluded cards to -inf for proper softmax behavior
                    probs = probs.clone()
                    probs[~mask] = float('-inf')
                    probs = F.softmax(probs, dim=0)
            
            # Get top K
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_cards))
            
            # Convert indices back to card IDs
            index_to_card_id = getattr(data, 'index_to_card_id', None)
            if index_to_card_id is not None:
                recommended_ids = [index_to_card_id[idx.item()] for idx in top_indices]
            else:
                recommended_ids = top_indices.cpu().tolist()
            
            return recommended_ids, top_probs.cpu().tolist()




