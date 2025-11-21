import os
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

from .cr_api import CRApiClient


def get_processed_player_tags(decks_path: str) -> set:
    """
    Get set of player tags that have already been processed (exist in decks.json).
    
    Args:
        decks_path: Path to decks.json file
        
    Returns:
        Set of player tags that have been processed
    """
    if not os.path.exists(decks_path):
        return set()
    
    try:
        with open(decks_path, "r", encoding="utf-8") as f:
            decks = json.load(f)
        
        processed_tags = set()
        for deck in decks:
            player_tag = deck.get("player_tag")
            if player_tag:
                processed_tags.add(player_tag)
        
        return processed_tags
    except Exception as e:
        print(f"Warning: Could not read existing decks: {e}")
        return set()


def fetch_cards(api_client: CRApiClient, output_dir: str) -> Dict:
    """
    Fetch all cards from the /cards endpoint.
    
    Args:
        api_client: CRApiClient instance
        output_dir: Directory to save cards data
        
    Returns:
        Dictionary containing items and supportItems
    """
    print("Fetching cards from API...")
    cards_data = api_client.get_cards()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cards.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cards data saved to {output_path}")
    print(f"Found {len(cards_data.get('items', []))} items and {len(cards_data.get('supportItems', []))} support items")
    
    return cards_data


def fetch_clans(api_client: CRApiClient, min_score: int, output_dir: str) -> List[Dict]:
    """
    Fetch clans with minimum score filter.
    
    Args:
        api_client: CRApiClient instance
        min_score: Minimum clan score for filtering
        output_dir: Directory to save clans data
        
    Returns:
        List of clan dictionaries
    """
    print(f"Fetching clans with min_score >= {min_score}...")
    clans_data = api_client.get_clans(min_score=min_score)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clans.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clans_data, f, indent=2, ensure_ascii=False)
    
    # Extract clan list (API might return items list or direct list)
    if isinstance(clans_data, dict) and "items" in clans_data:
        clans_list = clans_data["items"]
    elif isinstance(clans_data, list):
        clans_list = clans_data
    else:
        clans_list = []
    
    print(f"Clans data saved to {output_path}")
    print(f"Found {len(clans_list)} clans")
    
    return clans_list


def extract_player_tags_from_clans(clans: List[Dict], api_client: CRApiClient, max_clans: int = None) -> List[str]:
    """
    Extract all player tags from clan member lists by calling /clans/{clanTag}/members endpoint.
    
    Args:
        clans: List of clan dictionaries (from /clans endpoint)
        api_client: CRApiClient instance to fetch clan members
        max_clans: Maximum number of clans to process (None for all)
        
    Returns:
        List of unique player tags
    """
    player_tags = set()
    
    if max_clans:
        clans = clans[:max_clans]
    
    print(f"Extracting player tags from {len(clans)} clans...")
    print("This will call /clans/{clanTag}/members for each clan")
    
    failed_clans = []
    
    for i, clan in enumerate(tqdm(clans, desc="Fetching clan members")):
        clan_tag = clan.get("tag")
        if not clan_tag:
            continue
        
        try:
            # Call /clans/{clanTag}/members endpoint
            members_data = api_client.get_clan_members(clan_tag)
            
            # Handle different response formats
            if isinstance(members_data, dict):
                # Response might have "items" key
                member_list = members_data.get("items", [])
            elif isinstance(members_data, list):
                # Response is directly a list
                member_list = members_data
            else:
                member_list = []
            
            # Extract player tags from members
            for member in member_list:
                if isinstance(member, dict):
                    player_tag = member.get("tag")
                    if player_tag:
                        player_tags.add(player_tag)
            
            # Progress update every 10 clans
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(clans)} clans, found {len(player_tags)} unique players so far...")
                
        except Exception as e:
            failed_clans.append(clan_tag)
            if (i + 1) % 10 == 0:
                print(f"Warning: Could not fetch members for clan {clan_tag}: {e}")
            continue
    
    player_tags_list = list(player_tags)
    print(f"\nExtracted {len(player_tags_list)} unique player tags from {len(clans)} clans")
    if failed_clans:
        print(f"Failed to fetch members from {len(failed_clans)} clans")
    
    return player_tags_list


def fetch_player_battle_logs(
    api_client: CRApiClient, 
    player_tags: List[str], 
    output_dir: str, 
    max_players: int = None,
    skip_processed: bool = False,
    decks_path: str = None
) -> List[Dict]:
    """
    Fetch battle logs for multiple players.
    
    Args:
        api_client: CRApiClient instance
        player_tags: List of player tags
        output_dir: Directory to save battle logs
        max_players: Maximum number of players to process (None for all)
        skip_processed: If True, skip players that are already in decks.json
        decks_path: Path to decks.json to check for processed players
        
    Returns:
        List of battle log entries
    """
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out already processed players if skip_processed is True
    if skip_processed and decks_path:
        processed_tags = get_processed_player_tags(decks_path)
        original_count = len(player_tags)
        player_tags = [tag for tag in player_tags if tag not in processed_tags]
        skipped_count = original_count - len(player_tags)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already processed players. {len(player_tags)} new players to process.")
    
    if max_players:
        player_tags = player_tags[:max_players]
    
    if not player_tags:
        print("No new players to process.")
        return []
    
    all_battle_logs = []
    failed_tags = []
    
    print(f"Fetching battle logs for {len(player_tags)} players...")
    print(f"Rate limit: {api_client.requests_per_second} requests/second")
    
    for i, player_tag in enumerate(tqdm(player_tags, desc="Fetching battle logs")):
        try:
            battle_log = api_client.get_battle_log(player_tag)
            
            # Add player tag to each battle entry for tracking
            if isinstance(battle_log, list):
                for battle in battle_log:
                    battle["player_tag"] = player_tag
                all_battle_logs.extend(battle_log)
            else:
                battle_log["player_tag"] = player_tag
                all_battle_logs.append(battle_log)
            
            # Log progreso cada 10 requests
            if (i + 1) % 10 == 0:
                print(f"Procesados {i + 1}/{len(player_tags)} jugadores...")
                
        except Exception as e:
            print(f"Error fetching battle log for {player_tag}: {e}")
            failed_tags.append(player_tag)
            # Esperar un poco más si hay error
            time.sleep(api_client.min_delay * 2)
            continue
    
    # Append to existing battle logs if file exists
    output_path = os.path.join(output_dir, "battle_logs.json")
    existing_logs = []
    if os.path.exists(output_path) and skip_processed:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing battle logs: {e}")
    
    # Combine existing and new logs
    all_battle_logs = existing_logs + all_battle_logs
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_battle_logs, f, indent=2, ensure_ascii=False)
    
    print(f"Battle logs saved to {output_path}")
    print(f"Successfully fetched {len(all_battle_logs) - len(existing_logs)} new battle entries")
    print(f"Total battle entries: {len(all_battle_logs)}")
    if failed_tags:
        print(f"Failed to fetch logs for {len(failed_tags)} players")
    
    return all_battle_logs


def extract_decks_from_battle_logs(battle_logs: List[Dict]) -> List[Dict]:
    """
    Extract deck compositions from battle logs.
    
    Args:
        battle_logs: List of battle log entries
        
    Returns:
        List of deck dictionaries with 8 cards each
    """
    decks = []
    
    for battle in battle_logs:
        # Try to extract deck from different battle formats
        team = battle.get("team", [])
        opponent = battle.get("opponent", [])
        battle_type = battle.get("type", "")
        
        # Process team deck
        if team:
            team_cards = []
            for player in team:
                cards = player.get("cards", [])
                if cards:
                    card_ids = [card.get("id") for card in cards if card.get("id")]
                    if len(card_ids) == 8:
                        team_cards.append(card_ids)
            
            for card_ids in team_cards:
                decks.append({
                    "cards": card_ids,
                    "battle_type": battle_type,
                    "player_tag": battle.get("player_tag")
                })
        
        # Process opponent deck if available
        if opponent:
            opponent_cards = []
            for player in opponent:
                cards = player.get("cards", [])
                if cards:
                    card_ids = [card.get("id") for card in cards if card.get("id")]
                    if len(card_ids) == 8:
                        opponent_cards.append(card_ids)
            
            for card_ids in opponent_cards:
                decks.append({
                    "cards": card_ids,
                    "battle_type": battle_type,
                    "player_tag": battle.get("player_tag")
                })
    
    print(f"Extracted {len(decks)} decks from battle logs")
    return decks


def load_existing_decks(decks_path: str) -> List[Dict]:
    """
    Load existing decks from file.
    
    Args:
        decks_path: Path to decks.json
        
    Returns:
        List of existing decks
    """
    if not os.path.exists(decks_path):
        return []
    
    try:
        with open(decks_path, "r", encoding="utf-8") as f:
            decks = json.load(f)
        return decks if isinstance(decks, list) else []
    except Exception as e:
        print(f"Warning: Could not load existing decks: {e}")
        return []


def append_decks(new_decks: List[Dict], decks_path: str) -> List[Dict]:
    """
    Append new decks to existing decks file.
    
    Args:
        new_decks: New decks to add
        decks_path: Path to decks.json
        
    Returns:
        Combined list of all decks
    """
    existing_decks = load_existing_decks(decks_path)
    
    # Combine existing and new decks
    all_decks = existing_decks + new_decks
    
    # Save combined decks
    os.makedirs(os.path.dirname(decks_path), exist_ok=True)
    with open(decks_path, "w", encoding="utf-8") as f:
        json.dump(all_decks, f, indent=2, ensure_ascii=False)
    
    print(f"Added {len(new_decks)} new decks. Total decks: {len(all_decks)}")
    return all_decks


def build_co_occurrence_matrix(decks: List[Dict], output_dir: str, min_co_occurrence: int = 1) -> Dict:
    """
    Build co-occurrence matrix from deck data.
    Counts how often card pairs appear together in decks.
    
    Args:
        decks: List of deck dictionaries with card IDs
        output_dir: Directory to save co-occurrence matrix
        min_co_occurrence: Minimum co-occurrence count to include
        
    Returns:
        Dictionary with co-occurrence data
    """
    print("Building co-occurrence matrix...")
    
    co_occurrence = defaultdict(int)
    card_counts = defaultdict(int)
    
    # Count co-occurrences
    for deck in decks:
        card_ids = deck.get("cards", [])
        if len(card_ids) != 8:
            continue
        
        # Count individual cards
        for card_id in card_ids:
            card_counts[card_id] += 1
        
        # Count pairs
        for i in range(len(card_ids)):
            for j in range(i + 1, len(card_ids)):
                card1, card2 = card_ids[i], card_ids[j]
                # Use sorted tuple to ensure consistent ordering
                pair = tuple(sorted([card1, card2]))
                co_occurrence[pair] += 1
    
    # Filter by minimum co-occurrence
    filtered_co_occurrence = {
        f"{pair[0]}_{pair[1]}": count
        for pair, count in co_occurrence.items()
        if count >= min_co_occurrence
    }
    
    # Build edge list for graph
    edge_list = []
    for pair_key, count in filtered_co_occurrence.items():
        card1, card2 = pair_key.split("_")
        edge_list.append({
            "source": int(card1),
            "target": int(card2),
            "weight": count
        })
    
    result = {
        "co_occurrence_matrix": dict(filtered_co_occurrence),
        "edge_list": edge_list,
        "card_counts": dict(card_counts),
        "total_decks": len(decks),
        "total_edges": len(edge_list)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "co_occurrence_matrix.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Co-occurrence matrix saved to {output_path}")
    print(f"Total unique card pairs: {len(filtered_co_occurrence)}")
    print(f"Total edges: {len(edge_list)}")
    
    return result

