"""
Main entry point for inference with the trained GNN model.
"""
import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipelines.inference_pipeline import run_inference
from src.utils import load_config


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Clash Royale Card Recommendation Inference")
    parser.add_argument(
        "--cards",
        type=int,
        nargs=6,
        required=True,
        help="6 input card IDs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (default: models/best_model.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Clash Royale GNN Card Recommendation - Inference")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Run inference
    result = run_inference(
        input_card_ids=args.cards,
        model_path=args.model,
        config=config,
        device_preference=args.device
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Input Cards: {result['input_cards']}")
    print(f"Recommended Cards: {result['recommended_cards']}")
    print(f"Probabilities: {[f'{p:.4f}' for p in result['probabilities']]}")
    print("=" * 60)


if __name__ == "__main__":
    main()

