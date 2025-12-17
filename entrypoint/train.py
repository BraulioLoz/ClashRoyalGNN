"""
Main entry point for training the GNN card recommendation model.
"""
import sys
import os
import traceback
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipelines.training_pipeline import train_model
from src.utils import load_config, setup_training_logger


def main():
    """Main training function."""
    config = load_config()
    model_save_dir = config['training']['model_save_dir']
    
    # Setup logger
    logger = setup_training_logger(log_dir=model_save_dir, log_file="training_errors.log")
    
    try:
        logger.info("=" * 60)
        logger.info("Clash Royale GNN Card Recommendation - Training")
        logger.info("=" * 60)
        logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Config: epochs={config['training']['epochs']}, lr={config['training']['lr']}, batch_size={config['training']['batch_size']}")
        
        print("=" * 60)
        print("Clash Royale GNN Card Recommendation - Training")
        print("=" * 60)
        print(f"Logging errors to: {os.path.join(model_save_dir, 'training_errors.log')}")
        
        # Train model
        model = train_model(config=config, logger=logger)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_save_dir}/best_model.pt")
        print("\nTraining completed successfully!")
        print(f"Model saved to: {model_save_dir}/best_model.pt")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        logger.warning(f"Interrupted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nTraining interrupted by user")
        print(f"Check {os.path.join(model_save_dir, 'training_errors.log')} for details")
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"FATAL ERROR during training: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        print(f"\nERROR: {error_msg}")
        print(f"Check {os.path.join(model_save_dir, 'training_errors.log')} for details")
        raise


if __name__ == "__main__":
    main()

