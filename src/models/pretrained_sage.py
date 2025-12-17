"""
Transfer learning wrapper for pretrained GraphSAGE models from OGB.

This module provides functionality to:
1. Load pretrained GraphSAGE weights from OGB datasets
2. Add feature adapter for dimension matching (Clash Royale cards: 6 features → OGB: 128+ features)
3. Support staged training (frozen → partial → full fine-tuning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from typing import List, Optional


class FeatureAdapter(nn.Module):
    """
    Adapter layer to project low-dimensional features to high-dimensional space.
    
    Maps Clash Royale card features (6-dim) to pretrained model feature space (128-dim).
    Uses multi-layer projection for better expressiveness.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize feature adapter.
        
        Args:
            input_dim: Input feature dimension (e.g., 6 for Clash Royale cards)
            output_dim: Output feature dimension (e.g., 128 for OGB models)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(FeatureAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Two-layer MLP for feature projection
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.constant_(self.proj1.bias, 0)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.constant_(self.proj2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to higher dimensional space.
        
        Args:
            x: Input features [num_nodes, input_dim]
            
        Returns:
            Projected features [num_nodes, output_dim]
        """
        x = self.proj1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.proj2(x)
        x = self.layer_norm(x)
        return x


class PretrainedGraphSAGEEncoder(nn.Module):
    """
    Pretrained GraphSAGE encoder that can be loaded with OGB weights.
    
    This is a generic GraphSAGE encoder that can be initialized with
    pretrained weights or trained from scratch.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rates: List[float],
        aggr: str = "mean"
    ):
        """
        Initialize GraphSAGE encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates
            aggr: Aggregation method ('mean', 'max', 'lstm')
        """
        super(PretrainedGraphSAGEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))
            if i < len(dropout_rates):
                self.dropouts.append(nn.Dropout(dropout_rates[i]))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, hidden_dims[-1]]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        return x
    
    def freeze(self):
        """Freeze all parameters in this encoder."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters in this encoder."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_layers(self, num_layers: int):
        """
        Freeze first N layers.
        
        Args:
            num_layers: Number of layers to freeze from the beginning
        """
        for i, conv in enumerate(self.convs):
            if i < num_layers:
                for param in conv.parameters():
                    param.requires_grad = False
            else:
                for param in conv.parameters():
                    param.requires_grad = True


class CardRecommendationSAGEWithTransfer(nn.Module):
    """
    Card recommendation model using transfer learning from pretrained GraphSAGE.
    
    Architecture:
        Card Features (6-dim)
            ↓
        Feature Adapter (6 → 128-dim)
            ↓
        Pretrained GraphSAGE Encoder (128 → 256 → 128)
            ↓
        Fine-tuning Layers (128 → 64)
            ↓
        Task Head (64 → 110 cards)
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        pretrained_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        finetune_dims: List[int] = [64],
        dropout_rates: List[float] = [0.3, 0.2, 0.1],
        num_cards: int = 110,
        aggr: str = "mean",
        use_pretrained: bool = True
    ):
        """
        Initialize transfer learning model.
        
        Args:
            num_nodes: Number of nodes in graph
            node_feature_dim: Original node feature dimension (e.g., 6)
            pretrained_dim: Dimension of pretrained model features (e.g., 128)
            hidden_dims: Hidden dimensions for pretrained encoder
            finetune_dims: Dimensions for fine-tuning layers
            dropout_rates: Dropout rates for all layers
            num_cards: Number of output cards
            aggr: Aggregation method
            use_pretrained: Whether to use pretrained weights (if False, random init)
        """
        super(CardRecommendationSAGEWithTransfer, self).__init__()
        
        self.num_cards = num_cards
        self.use_pretrained = use_pretrained
        self.node_feature_dim = node_feature_dim
        
        # Add binary indicator for input cards
        input_dim = node_feature_dim + 1
        
        # Feature adapter: Project card features to pretrained dimension
        self.feature_adapter = FeatureAdapter(
            input_dim=input_dim,
            output_dim=pretrained_dim,
            hidden_dim=pretrained_dim // 2,
            dropout=dropout_rates[0] if dropout_rates else 0.2
        )
        
        # Pretrained GraphSAGE encoder
        self.pretrained_encoder = PretrainedGraphSAGEEncoder(
            input_dim=pretrained_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates[:len(hidden_dims)],
            aggr=aggr
        )
        
        # Fine-tuning layers (task-specific)
        self.finetune_layers = nn.ModuleList()
        self.finetune_dropouts = nn.ModuleList()
        
        dims = [hidden_dims[-1]] + finetune_dims
        for i in range(len(finetune_dims)):
            self.finetune_layers.append(
                SAGEConv(dims[i], dims[i + 1], aggr=aggr)
            )
            if len(dropout_rates) > len(hidden_dims) + i:
                self.finetune_dropouts.append(
                    nn.Dropout(dropout_rates[len(hidden_dims) + i])
                )
        
        # Output head: Task-specific prediction layer
        final_dim = finetune_dims[-1] if finetune_dims else hidden_dims[-1]
        self.output_layer = nn.Linear(final_dim, num_cards)
        
        # Initialize output layer
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the transfer learning model.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Logits for each card [num_nodes, num_cards]
        """
        x = data.x
        edge_index = data.edge_index
        
        # Add binary indicator for input cards
        input_indicator = getattr(data, 'input_cards', torch.zeros(x.size(0), device=x.device))
        input_indicator = input_indicator.unsqueeze(1).float()
        x = torch.cat([x, input_indicator], dim=1)
        
        # Feature adapter: Project to pretrained dimension
        x = self.feature_adapter(x)
        
        # Pretrained encoder
        x = self.pretrained_encoder(x, edge_index)
        
        # Fine-tuning layers
        for i, layer in enumerate(self.finetune_layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            if i < len(self.finetune_dropouts):
                x = self.finetune_dropouts[i](x)
        
        # Output layer
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
            logits = self.forward(data)
            
            # Aggregate logits across nodes
            node_logits = logits.mean(dim=0)
            
            # Apply softmax to get probabilities
            probs = F.softmax(node_logits, dim=0)
            
            # Exclude input cards if requested
            if exclude_input:
                card_id_to_index = getattr(data, 'card_id_to_index', None)
                if card_id_to_index is not None:
                    mask = torch.ones(self.num_cards, device=logits.device, dtype=torch.bool)
                    
                    for card_id in input_card_ids:
                        idx = card_id_to_index.get(card_id)
                        if idx is not None and idx < self.num_cards:
                            mask[idx] = False
                    
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
    
    def set_training_stage(self, stage: str):
        """
        Set training stage (controls which layers are frozen).
        
        Args:
            stage: Training stage:
                - "adapter": Only train feature adapter + output head
                - "partial": Train adapter + last N pretrained layers + finetune + output
                - "full": Train all layers
        """
        if stage == "adapter":
            # Stage 1: Only adapter and output head
            self.pretrained_encoder.freeze()
            for layer in self.finetune_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            
            # Unfreeze adapter and output
            for param in self.feature_adapter.parameters():
                param.requires_grad = True
            for param in self.output_layer.parameters():
                param.requires_grad = True
            
            print("Training Stage: ADAPTER ONLY")
            print("  - Trainable: Feature Adapter, Output Head")
            print("  - Frozen: Pretrained Encoder, Fine-tuning Layers")
            
        elif stage == "partial":
            # Stage 2: Adapter + last 2 pretrained layers + finetune + output
            num_layers = len(self.pretrained_encoder.convs)
            freeze_until = max(0, num_layers - 2)
            self.pretrained_encoder.freeze_layers(freeze_until)
            
            # Unfreeze adapter, finetune, and output
            for param in self.feature_adapter.parameters():
                param.requires_grad = True
            for layer in self.finetune_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.output_layer.parameters():
                param.requires_grad = True
            
            print("Training Stage: PARTIAL FINE-TUNING")
            print(f"  - Trainable: Feature Adapter, Last {num_layers - freeze_until} Pretrained Layers, Fine-tuning Layers, Output Head")
            print(f"  - Frozen: First {freeze_until} Pretrained Layers")
            
        elif stage == "full":
            # Stage 3: All layers trainable
            self.pretrained_encoder.unfreeze()
            for param in self.parameters():
                param.requires_grad = True
            
            print("Training Stage: FULL FINE-TUNING")
            print("  - Trainable: All layers")
            
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 'adapter', 'partial', or 'full'")
    
    def load_pretrained_weights(self, weights_path: str, strict: bool = False):
        """
        Load pretrained weights for the encoder.
        
        Args:
            weights_path: Path to pretrained weights
            strict: Whether to strictly enforce weight matching
        """
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Try to load encoder weights
        if 'encoder' in checkpoint:
            self.pretrained_encoder.load_state_dict(checkpoint['encoder'], strict=strict)
            print(f"Loaded pretrained encoder weights from {weights_path}")
        elif 'model' in checkpoint:
            self.pretrained_encoder.load_state_dict(checkpoint['model'], strict=strict)
            print(f"Loaded pretrained model weights from {weights_path}")
        else:
            self.pretrained_encoder.load_state_dict(checkpoint, strict=strict)
            print(f"Loaded pretrained weights from {weights_path}")




