"""
Main entry point for training the GNN card recommendation model.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipelines.training_pipeline import train_model
from src.utils import load_config


def main():
    """Main training function."""
    print("=" * 60)
    print("Clash Royale GNN Card Recommendation - Training")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Train model
    model = train_model(config=config)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {config['training']['model_save_dir']}/best_model.pt")


if __name__ == "__main__":
    main()

