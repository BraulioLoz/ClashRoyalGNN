"""
Script to process features and create graph structure for training.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pipelines.feature_eng_pipeline import process_features
from src.utils import load_config

def main():
    """Main feature processing function."""
    print("=" * 60)
    print("Clash Royale GNN - Feature Engineering")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Path to decks collected by optimized pipeline
    decks_path = os.path.join(config["data"]["raw_dir"], "decks.json")
    
    # Process features and create graph structure
    print("\nStarting feature engineering...")
    graph_data, training_examples, id_to_index, index_to_id = process_features(
        config,
        decks_path=decks_path  # Pass the decks path
    )
    
    print("\n" + "=" * 60)
    print("Feature engineering completed successfully!")
    print("=" * 60)
    print(f"Graph created with {len(id_to_index)} nodes")
    print(f"Created {len(training_examples)} training examples")
    print(f"Features saved to: {config['data']['features_dir']}")

if __name__ == "__main__":
    main()