"""
Script to visualize training progress from saved history.
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


def load_training_history(history_path: str):
    """Load training history from JSON or CSV."""
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    if history_path.endswith('.csv'):
        df = pd.read_csv(history_path)
        # Add global epoch counter (continuous across stages)
        df['global_epoch'] = range(1, len(df) + 1)
        return df
    elif history_path.endswith('.json'):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        rows = []
        for entry in history:
            row = {
                'epoch': entry.get('epoch'),
                'train_loss': entry.get('train_loss'),
                'val_loss': entry.get('val_loss'),
                'lr': entry.get('lr')
            }
            # Flatten metrics
            if 'train_metrics' in entry and entry['train_metrics']:
                for k, v in entry['train_metrics'].items():
                    row[f'train_{k}'] = v
            if 'val_metrics' in entry and entry['val_metrics']:
                for k, v in entry['val_metrics'].items():
                    row[f'val_{k}'] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        
        # Add global epoch counter (continuous across stages)
        df['global_epoch'] = range(1, len(df) + 1)
        
        return df
    else:
        raise ValueError(f"Unsupported file format: {history_path}")


def plot_training_history(df: pd.DataFrame, output_path: str = None, show: bool = True):
    """
    Plot training history with multiple subplots.
    
    Args:
        df: DataFrame with training history
        output_path: Path to save the plot
        show: Whether to display the plot
    """
    if df.empty:
        print("No training history to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = df['global_epoch'].values if 'global_epoch' in df.columns else df['epoch'].values
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train_loss'], label='Train Loss', marker='o', markersize=3, linewidth=1.5)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', marker='s', markersize=3, linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning rate
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['lr'], label='Learning Rate', color='green', marker='o', markersize=3, linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top-k Accuracy (if available)
    ax3 = axes[1, 0]
    top_k_cols = [col for col in df.columns if 'top_' in col and 'acc' in col and 'val_' in col]
    if top_k_cols:
        for col in sorted(top_k_cols):
            label = col.replace('val_', '').replace('_', ' ').title()
            ax3.plot(epochs, df[col], label=label, marker='o', markersize=3, linewidth=1.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Top-K Validation Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.1])
    else:
        ax3.text(0.5, 0.5, 'No accuracy metrics available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Top-K Accuracy (No Data)')
    
    # 4. Target probabilities (if available)
    ax4 = axes[1, 1]
    prob_cols = [col for col in df.columns if 'target_prob' in col and 'val_' in col]
    if prob_cols:
        for col in sorted(prob_cols):
            label = col.replace('val_', '').replace('_', ' ').title()
            ax4.plot(epochs, df[col], label=label, marker='o', markersize=3, linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Probability')
        ax4.set_title('Target Card Probabilities')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No probability metrics available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Target Probabilities (No Data)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument('--history', type=str, default=None,
                       help='Path to training history file (JSON or CSV)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the plot')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display the plot')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode: continuously update plot')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds for watch mode')
    
    args = parser.parse_args()
    
    # Load config if history not specified
    if args.history is None:
        config = load_config()
        model_save_dir = config["training"]["model_save_dir"]
        # Try CSV first (easier to read), then JSON
        csv_path = os.path.join(model_save_dir, "training_history.csv")
        json_path = os.path.join(model_save_dir, "training_history.json")
        
        if os.path.exists(csv_path):
            args.history = csv_path
        elif os.path.exists(json_path):
            args.history = json_path
        else:
            print(f"Error: No training history found in {model_save_dir}")
            print("Please run training first or specify --history path")
            return
    
    if args.watch:
        # Watch mode: continuously update
        import time
        print(f"Watching {args.history} (updating every {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                if os.path.exists(args.history):
                    try:
                        df = load_training_history(args.history)
                        if not df.empty:
                            plot_training_history(df, args.output, show=not args.no_show)
                            print(f"Updated at {pd.Timestamp.now().strftime('%H:%M:%S')}")
                    except Exception as e:
                        print(f"Error loading history: {e}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped watching")
    else:
        # Single plot
        try:
            df = load_training_history(args.history)
            plot_training_history(df, args.output, show=not args.no_show)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

