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
from src.utils.card_mapper import CardMapper


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Clash Royale Card Recommendation Inference")
    parser.add_argument(
        "--cards",
        type=str,
        required=True,
        help="6 input card names separated by commas (e.g., 'Hog Rider, mini PEKKA, Giant Snowball, Skeletons, Electro Spirit, Cannon')"
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
    
    # Parse comma-separated card names
    card_names = [name.strip() for name in args.cards.split(',')]
    
    # Validate exactly 6 cards
    if len(card_names) != 6:
        print(f"\nError: Expected 6 cards but got {len(card_names)}")
        print(f"Cards provided: {card_names}")
        sys.exit(1)
    
    # Load config
    config = load_config()
    
    # Initialize card mapper
    try:
        card_mapper = CardMapper()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Convert card names to IDs
    print(f"\nConverting card names to IDs...")
    try:
        input_card_ids = card_mapper.batch_name_to_id(card_names)
        print(f"Input cards: {', '.join(card_names)}")
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Run inference
    result = run_inference(
        input_card_ids=input_card_ids,
        model_path=args.model,
        config=config,
        device_preference=args.device,
        card_mapper=card_mapper
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    # Display card names instead of IDs
    if 'input_card_names' in result:
        print(f"Input Cards: {result['input_card_names']}")
    else:
        print(f"Input Cards: {result['input_cards']}")
    
    if 'recommended_card_names' in result:
        print(f"Recommended Cards: {result['recommended_card_names']}")
    else:
        print(f"Recommended Cards: {result['recommended_cards']}")
    
    print(f"Probabilities: {[f'{p*100:.2f}%' for p in result['probabilities']]}")
    print("=" * 60)


if __name__ == "__main__":
    main()

