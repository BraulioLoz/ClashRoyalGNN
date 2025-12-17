"""
Card name to ID mapping utility for inference.
"""
import json
import os
from typing import List, Dict, Optional
from difflib import SequenceMatcher


class CardMapper:
    """Maps card names to IDs and vice versa with fuzzy matching support."""
    
    def __init__(self, cards_json_path: str = "data/01-raw/cards.json"):
        """
        Initialize CardMapper with cards data.
        
        Args:
            cards_json_path: Path to cards.json file
        """
        if not os.path.exists(cards_json_path):
            raise FileNotFoundError(f"Cards file not found at {cards_json_path}")
        
        with open(cards_json_path, "r", encoding="utf-8") as f:
            cards_data = json.load(f)
        
        self.cards = cards_data["items"]
        
        # Create mappings
        self.id_to_name_dict: Dict[int, str] = {}
        self.name_to_id_map: Dict[str, int] = {}
        self.normalized_to_original: Dict[str, str] = {}
        
        for card in self.cards:
            card_id = card["id"]
            card_name = card["name"]
            
            self.id_to_name_dict[card_id] = card_name
            
            # Store normalized version
            normalized = self.normalize_name(card_name)
            self.name_to_id_map[normalized] = card_id
            self.normalized_to_original[normalized] = card_name
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize card name for matching.
        
        Removes dots, hyphens, extra spaces, and converts to lowercase.
        Examples:
            "P.E.K.K.A." -> "pekka"
            "Mini P.E.K.K.A" -> "mini pekka"
            "Hog Rider" -> "hog rider"
            "X-Bow" -> "xbow"
        
        Args:
            name: Original card name
            
        Returns:
            Normalized name
        """
        # Remove dots and hyphens
        normalized = name.replace(".", "").replace("-", "")
        # Convert to lowercase
        normalized = normalized.lower()
        # Remove extra spaces
        normalized = " ".join(normalized.split())
        return normalized
    
    def name_to_id(self, name: str) -> int:
        """
        Convert card name to ID.
        
        Args:
            name: Card name (case-insensitive, flexible with dots/hyphens)
            
        Returns:
            Card ID
            
        Raises:
            ValueError: If card name not found
        """
        normalized = self.normalize_name(name)
        
        if normalized in self.name_to_id_map:
            return self.name_to_id_map[normalized]
        
        # Try to find suggestions
        suggestions = self.find_similar_cards(name, limit=5)
        suggestion_text = "\n  - ".join(suggestions)
        
        raise ValueError(
            f"Card '{name}' not found.\n"
            f"Did you mean one of these?\n  - {suggestion_text}\n"
            f"Use get_all_card_names() to see all available cards."
        )
    
    def id_to_name(self, card_id: int) -> str:
        """
        Convert card ID to name.
        
        Args:
            card_id: Card ID
            
        Returns:
            Card name
            
        Raises:
            ValueError: If card ID not found
        """
        if card_id in self.id_to_name_dict:
            return self.id_to_name_dict[card_id]
        
        raise ValueError(f"Card ID {card_id} not found in cards database.")
    
    def find_similar_cards(self, name: str, limit: int = 5) -> List[str]:
        """
        Find similar card names using sequence matching.
        
        Args:
            name: Input card name
            limit: Maximum number of suggestions
            
        Returns:
            List of similar card names
        """
        normalized_input = self.normalize_name(name)
        
        # Calculate similarity scores
        similarities = []
        for normalized, original in self.normalized_to_original.items():
            ratio = SequenceMatcher(None, normalized_input, normalized).ratio()
            similarities.append((ratio, original))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top N card names
        return [card_name for _, card_name in similarities[:limit]]
    
    def get_all_card_names(self) -> List[str]:
        """
        Get all available card names.
        
        Returns:
            List of all card names (sorted alphabetically)
        """
        return sorted(self.id_to_name_dict.values())
    
    def batch_name_to_id(self, names: List[str]) -> List[int]:
        """
        Convert multiple card names to IDs.
        
        Args:
            names: List of card names
            
        Returns:
            List of card IDs
            
        Raises:
            ValueError: If any card name not found
        """
        return [self.name_to_id(name) for name in names]
    
    def batch_id_to_name(self, card_ids: List[int]) -> List[str]:
        """
        Convert multiple card IDs to names.
        
        Args:
            card_ids: List of card IDs
            
        Returns:
            List of card names
            
        Raises:
            ValueError: If any card ID not found
        """
        return [self.id_to_name(card_id) for card_id in card_ids]

