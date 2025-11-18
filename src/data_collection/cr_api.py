import os
import requests
import yaml

class CRApiClient:
    def __init__(self, token: str = None, base_url: str = None):
        if token:
            self.token = token
        else:
            # try config.yaml then environment
            cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.token = cfg.get("api", {}).get("token") or os.getenv("CR_API_TOKEN")
                self.base_url = cfg.get("api", {}).get("base_url", "https://api.clashroyale.com/v1")
            except FileNotFoundError:
                self.token = os.getenv("CR_API_TOKEN")
                self.base_url = base_url or "https://api.clashroyale.com/v1"
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get(self, path: str, params: dict = None):
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, headers=self.headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # Example helpers
    def get_player(self, player_tag: str):
        # player_tag should include %23 if needed or '#' replaced
        tag = player_tag.replace("#", "%23")
        return self._get(f"players/{tag}")

    def get_battle_log(self, player_tag: str):
        tag = player_tag.replace("#", "%23")
        return self._get(f"players/{tag}/battlelog")

    def get_cards(self):
        """Get all cards metadata including stats, types, rarity"""
        return self._get("cards")

    def get_top_players(self, limit: int = 200):
        """Get top players from global rankings"""
        try:
            response = self._get("locations/global/rankings/players", {"limit": limit})
            print(f"DEBUG - Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
            print(f"DEBUG - Response type: {type(response)}")
            if isinstance(response, dict):
                print(f"DEBUG - Items count: {len(response.get('items', []))}")
            return response
        except Exception as e:
            print(f"ERROR fetching top players: {e}")
            raise

    def get_player_by_tag(self, player_tag: str):
        """Get player info by tag"""
        tag = player_tag.replace("#", "%23")
        return self._get(f"players/{tag}")

    def get_top_clans(self, limit: int = 200):
        """Get top clans from global rankings"""
        return self._get("locations/global/rankings/clans", {"limit": limit})

    def get_clan_members(self, clan_tag: str):
        """Get members of a specific clan"""
        tag = clan_tag.replace("#", "%23")
        return self._get(f"clans/{tag}/members")

    def get_clan_info(self, clan_tag: str):
        """Get clan information"""
        tag = clan_tag.replace("#", "%23")
        return self._get(f"clans/{tag}")