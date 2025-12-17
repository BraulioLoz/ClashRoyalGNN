"""
Main entry point for data collection pipeline.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_collection.cr_api import CRApiClient
from src.data_collection.data_fetcher import (
    fetch_cards,
    fetch_clans,
    extract_player_tags_from_clans,
    fetch_player_battle_logs,
    fetch_and_extract_decks,
    extract_decks_from_battle_logs,
    build_co_occurrence_matrix,
    load_existing_decks,
    append_decks,
    get_processed_player_tags
)
from src.utils import load_config


def main():
    """Main data collection function."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale GNN Data Collection")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: skip already processed players and append to existing data"
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Use optimized collection: on-the-fly extraction, deduplication, checkpointing"
    )
    parser.add_argument(
        "--max-clans",
        type=int,
        default=60,
        help="Maximum number of clans to process (default: 60)"
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Maximum number of players to process (default: None = all available)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N players (default: 50, only with --optimized)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Clash Royale GNN - Data Collection Pipeline")
    if args.incremental:
        print("Mode: INCREMENTAL (will skip processed players and append data)")
    else:
        print("Mode: FULL (will overwrite existing data)")
    if args.optimized:
        print("Optimization: ENABLED (on-the-fly extraction, deduplication, checkpoints)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Initialize API client
    api_client = CRApiClient()
    
    decks_path = os.path.join(config["data"]["raw_dir"], "decks.json")
    
    # Check existing data if incremental mode
    if args.incremental and os.path.exists(decks_path):
        existing_decks = load_existing_decks(decks_path)
        processed_tags = get_processed_player_tags(decks_path)
        print(f"\nFound existing data: {len(existing_decks)} decks from {len(processed_tags)} players")
    
    # Step 1: Fetch cards (only if not exists or not incremental)
    if not args.incremental or not os.path.exists(os.path.join(config["data"]["raw_dir"], "cards.json")):
        print("\n[Step 1/5] Fetching cards...")
        cards_data = fetch_cards(api_client, config["data"]["raw_dir"])
    else:
        print("\n[Step 1/5] Skipping cards (already exists)")
        with open(os.path.join(config["data"]["raw_dir"], "cards.json"), "r", encoding="utf-8") as f:
            cards_data = json.load(f)
    
    # Step 2: Fetch clans (only if not exists or not incremental)
    if not args.incremental or not os.path.exists(os.path.join(config["data"]["raw_dir"], "clans.json")):
        print("\n[Step 2/5] Fetching clans...")
        min_score = config["data"]["min_clan_score"]
        clans = fetch_clans(api_client, min_score, config["data"]["raw_dir"])
    else:
        print("\n[Step 2/5] Loading existing clans...")
        with open(os.path.join(config["data"]["raw_dir"], "clans.json"), "r", encoding="utf-8") as f:
            clans_data = json.load(f)
            if isinstance(clans_data, dict) and "items" in clans_data:
                clans = clans_data["items"]
            elif isinstance(clans_data, list):
                clans = clans_data
            else:
                clans = []
    
    # Step 3: Extract player tags from clans
    print(f"\n[Step 3/5] Extracting player tags from {args.max_clans} clans...")
    player_tags = extract_player_tags_from_clans(clans, api_client, max_clans=args.max_clans)
    
    # Filter out already processed players in incremental mode (only for legacy mode)
    if args.incremental and not args.optimized:
        processed_tags = get_processed_player_tags(decks_path)
        original_count = len(player_tags)
        player_tags = [tag for tag in player_tags if tag not in processed_tags]
        skipped = original_count - len(player_tags)
        print(f"Filtered players: {skipped} already processed, {len(player_tags)} new players to process")
        
        if not player_tags:
            print("No new players to process. Exiting.")
            return
    
    # Step 4 & 5: Fetch and extract decks
    if args.optimized:
        # Use optimized pipeline: fetch and extract on-the-fly with deduplication
        print("\n[Step 4-5/5] Fetching and extracting decks (optimized)...")
        all_decks = fetch_and_extract_decks(
            api_client,
            player_tags,
            config["data"]["raw_dir"],
            max_players=args.max_players,
            skip_processed=args.incremental,
            decks_path=decks_path if args.incremental else None,
            checkpoint_interval=args.checkpoint_interval,
            resume_from_checkpoint=args.incremental
        )
        
        if not all_decks:
            print("No decks collected.")
            return
        
        # Save decks
        with open(decks_path, "w", encoding="utf-8") as f:
            json.dump(all_decks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_decks)} decks to {decks_path}")
        
        new_decks = all_decks  # For stats at the end
        
    else:
        # Legacy pipeline: fetch logs, then extract
        print("\n[Step 4/5] Fetching battle logs...")
        battle_logs = fetch_player_battle_logs(
            api_client,
            player_tags,
            config["data"]["raw_dir"],
            max_players=args.max_players,
            skip_processed=args.incremental,
            decks_path=decks_path
        )
        
        if not battle_logs:
            print("No new battle logs to process.")
            # Still rebuild co-occurrence matrix with existing data
            if args.incremental:
                existing_decks = load_existing_decks(decks_path)
                if existing_decks:
                    print("\nRebuilding co-occurrence matrix with existing decks...")
                    edge_threshold = config["graph"]["edge_threshold"]
                    co_occurrence = build_co_occurrence_matrix(
                        existing_decks,
                        config["data"]["processed_dir"],
                        min_co_occurrence=edge_threshold
                    )
                    print(f"Co-occurrence matrix rebuilt with {co_occurrence['total_edges']} edges")
            return
        
        # Step 5: Extract decks and build co-occurrence matrix
        print("\n[Step 5/5] Extracting decks and building co-occurrence matrix...")
        new_decks = extract_decks_from_battle_logs(battle_logs)
        
        # Append or save decks
        if args.incremental:
            all_decks = append_decks(new_decks, decks_path)
        else:
            all_decks = new_decks
            with open(decks_path, "w", encoding="utf-8") as f:
                json.dump(all_decks, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(all_decks)} decks to {decks_path}")
    
    # Build co-occurrence matrix with ALL decks
    edge_threshold = config["graph"]["edge_threshold"]
    co_occurrence = build_co_occurrence_matrix(
        all_decks,
        config["data"]["processed_dir"],
        min_co_occurrence=edge_threshold
    )
    
    # Print win/loss statistics
    wins = sum(1 for d in all_decks if d.get("is_winner", False))
    losses = sum(1 for d in all_decks if d.get("crown_margin", 0) < 0)
    draws = sum(1 for d in all_decks if d.get("crown_margin", 0) == 0)
    
    print("\n" + "=" * 60)
    print("Data collection completed successfully!")
    print("=" * 60)
    print(f"Cards: {len(cards_data.get('items', []))} items")
    print(f"Clans processed: {args.max_clans}")
    print(f"New players processed: {len(player_tags)}")
    print(f"New decks added: {len(new_decks)}")
    print(f"Total decks: {len(all_decks)}")
    print(f"  - Wins: {wins} ({100*wins/len(all_decks):.1f}%)")
    print(f"  - Losses: {losses} ({100*losses/len(all_decks):.1f}%)")
    print(f"  - Draws: {draws} ({100*draws/len(all_decks):.1f}%)")
    print(f"Co-occurrence edges: {co_occurrence['total_edges']} edges")


if __name__ == "__main__":
    main()
