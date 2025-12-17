"""
Script to explore OGB pretrained models and available datasets.

This script helps understand what pretrained GraphSAGE models are available
from the Open Graph Benchmark and their characteristics.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from ogb.nodeproppred import NodePropPredDataset
    from ogb.graphproppred import GraphPropPredDataset
    import torch
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    print(f"Error importing OGB: {e}")
    print("Please install OGB: pip install ogb")
    sys.exit(1)


def explore_node_datasets():
    """Explore available node property prediction datasets."""
    print("=" * 80)
    print("OGB Node Property Prediction Datasets")
    print("=" * 80)
    
    # Common datasets suitable for GraphSAGE
    datasets = [
        ('ogbn-products', 'Amazon product co-purchasing network'),
        ('ogbn-arxiv', 'Citation network of arXiv papers'),
        ('ogbn-papers100M', 'Citation network (100M+ nodes, very large)'),
        ('ogbn-proteins', 'Protein-protein association network'),
    ]
    
    for dataset_name, description in datasets:
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"   Description: {description}")
        
        try:
            # Try to load dataset metadata without downloading
            print(f"   Loading metadata...")
            dataset = NodePropPredDataset(name=dataset_name, root='/tmp/ogb')
            
            graph = dataset[0]  # Get first graph
            num_nodes = graph['num_nodes']
            num_edges = graph['edge_index'].shape[1]
            num_features = graph['node_feat'].shape[1] if 'node_feat' in graph else 0
            
            print(f"   ✓ Nodes: {num_nodes:,}")
            print(f"   ✓ Edges: {num_edges:,}")
            print(f"   ✓ Node features: {num_features}")
            
            split_idx = dataset.get_idx_split()
            print(f"   ✓ Train: {len(split_idx['train']):,} nodes")
            print(f"   ✓ Val: {len(split_idx['valid']):,} nodes")
            print(f"   ✓ Test: {len(split_idx['test']):,} nodes")
            
        except Exception as e:
            print(f"   ✗ Error loading: {str(e)[:100]}")
    
    print("\n" + "=" * 80)


def create_pretrained_graphsage_example(input_dim: int = 100, hidden_dim: int = 256, output_dim: int = 128):
    """
    Create an example GraphSAGE model that could be pretrained.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        
    Returns:
        GraphSAGE model
    """
    print("\n" + "=" * 80)
    print("Example GraphSAGE Architecture for Transfer Learning")
    print("=" * 80)
    
    class PretrainedGraphSAGE(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, output_dim)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = self.conv3(x, edge_index)
            return x
    
    model = PretrainedGraphSAGE(input_dim, hidden_dim, output_dim)
    
    print(f"\n📐 Model Architecture:")
    print(f"   Input: {input_dim} features")
    print(f"   Hidden: {hidden_dim} features")
    print(f"   Output: {output_dim} features (embedding)")
    print(f"\n   Layer 1: SAGEConv({input_dim} → {hidden_dim}) + ReLU")
    print(f"   Layer 2: SAGEConv({hidden_dim} → {hidden_dim}) + ReLU")
    print(f"   Layer 3: SAGEConv({hidden_dim} → {output_dim})")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Total parameters: {total_params:,}")
    
    return model


def explain_transfer_learning_approach():
    """Explain how to use OGB pretrained models for Clash Royale."""
    print("\n" + "=" * 80)
    print("Transfer Learning Strategy for Clash Royale Cards")
    print("=" * 80)
    
    print("""
🎯 Challenge:
   - OGB models trained on citation/product networks (100-128+ features)
   - Clash Royale cards have only 5-6 features
   - Need to bridge this feature dimension gap

💡 Solution: Feature Adapter + Pretrained Encoder
   
   Architecture:
   
   Card Features (6-dim)
       ↓
   Feature Adapter (6 → 128-dim) [Trainable]
       ↓
   Pretrained GraphSAGE Layers [Frozen initially]
       ↓
   Fine-tuning Layers [Trainable]
       ↓
   Task-Specific Head (→ 110 cards) [Trainable]

📋 Training Strategy:

   Stage 1 (5-10 epochs):
      ✓ Train: Feature Adapter + Task Head
      ✗ Freeze: Pretrained GraphSAGE layers
      Goal: Learn card feature representation

   Stage 2 (10-20 epochs):
      ✓ Train: Feature Adapter + Last 2 GraphSAGE layers + Task Head
      ✗ Freeze: First GraphSAGE layers
      Goal: Fine-tune high-level features

   Stage 3 (10-20 epochs):
      ✓ Train: All layers
      Goal: Full fine-tuning for Clash Royale domain

⚠️  Reality Check:
   
   OGB pretrained models may NOT transfer well because:
   - Different domain (citations/products vs. game cards)
   - Different graph structure (citation network vs. synergy graph)
   - Different task (node classification vs. card recommendation)
   
   However, they might provide:
   ✓ Better initialization than random weights
   ✓ Learned graph reasoning capabilities
   ✓ Faster convergence (potentially)

🔬 Recommendation:
   
   1. Start with from-scratch GraphSAGE (simpler, domain-specific)
   2. Compare with pretrained transfer learning
   3. Only use pretrained if it shows clear benefit
   
   For Clash Royale, from-scratch training is likely better because:
   - Small dataset (can train quickly)
   - Domain-specific patterns (card synergies)
   - Simple graph (110 nodes vs. millions in OGB)
""")


def main():
    """Main exploration function."""
    print("\n" + "=" * 80)
    print("OGB Pretrained Models Explorer")
    print("=" * 80)
    print("\nThis script explores pretrained GraphSAGE options from OGB.")
    
    # Explore datasets
    explore_node_datasets()
    
    # Show example architecture
    create_pretrained_graphsage_example(input_dim=100, hidden_dim=256, output_dim=128)
    
    # Explain transfer learning
    explain_transfer_learning_approach()
    
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("""
1. Implement feature adapter (6 → 128 dimensions)
2. Create transfer learning wrapper
3. Implement staged training strategy
4. Compare pretrained vs. from-scratch GraphSAGE
5. Evaluate which approach works best for Clash Royale

Note: The pretrained_sage.py module will implement the full transfer learning
      pipeline with OGB pretrained weights.
""")


if __name__ == "__main__":
    main()




