import os
import requests
import yaml
import time
from time import sleep
from typing import Optional

class CRApiClient:
    def __init__(self, token: str = None, base_url: str = None, config: dict = None):
        if token:
            self.token = token
        else:
            # try config.yaml then environment
            # Try multiple paths to find config.yaml
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
                os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml"),
                "config/config.yaml",
                os.path.join(os.getcwd(), "config", "config.yaml"),
            ]
            
            cfg = {}
            cfg_path = None
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    cfg_path = abs_path
                    try:
                        with open(abs_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)
                        break
                    except Exception as e:
                        print(f"Warning: Could not load config from {abs_path}: {e}")
                        continue
            
            if cfg_path:
                self.token = cfg.get("api", {}).get("token") or os.getenv("CR_API_TOKEN")
                self.base_url = cfg.get("api", {}).get("base_url", "https://api.clashroyale.com/v1")
                self.config = cfg
            else:
                # Fallback to environment variable
                self.token = os.getenv("CR_API_TOKEN")
                self.base_url = base_url or "https://api.clashroyale.com/v1"
                self.config = {}
                if not self.token:
                    print("Warning: Could not find config.yaml and CR_API_TOKEN environment variable is not set")
        
        # Strip token of any whitespace
        if self.token:
            self.token = self.token.strip()
        
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        
        # Rate limiting configuration
        rate_limit_config = self.config.get("api", {}).get("rate_limit", {})
        self.requests_per_second = rate_limit_config.get("requests_per_second", 0.5)
        self.min_delay = 1.0 / self.requests_per_second  # Tiempo mínimo entre requests
        self.max_retries = rate_limit_config.get("max_retries", 3)
        self.retry_delay = rate_limit_config.get("retry_delay", 60)
        
        # Track last request time
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """Espera el tiempo necesario para respetar el rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_delay:
            sleep_time = self.min_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _get(self, path: str, params: dict = None, retry_count: int = 0):
        """
        Realiza una petición GET con rate limiting y manejo de errores 429.
        
        Args:
            path: Ruta del endpoint
            params: Parámetros de la petición
            retry_count: Contador de reintentos
            
        Returns:
            Respuesta JSON de la API
        """
        # Esperar para respetar rate limit
        self._wait_for_rate_limit()
        
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            # Manejar error 403 (Forbidden) - Token inválido o expirado
            if resp.status_code == 403:
                print("\n" + "="*60)
                print("ERROR 403: Token de API inválido o expirado")
                print("="*60)
                print(f"URL: {url}")
                print(f"Token presente: {bool(self.token)}")
                if self.token:
                    print(f"Longitud del token: {len(self.token)}")
                    print(f"Token comienza con: {self.token[:30]}...")
                    print(f"Header Authorization: {self.headers.get('Authorization', 'NO ENCONTRADO')[:50]}...")
                print(f"\nRespuesta del servidor:")
                print(resp.text[:500])
                print("="*60)
                print("\nSolución:")
                print("1. Ve a https://developer.clashroyale.com")
                print("2. Verifica que tu token esté activo")
                print("3. Si es necesario, genera un nuevo token")
                print("4. Actualiza el token en config/config.yaml")
                print("5. Asegúrate de que el token NO tenga espacios ni comillas extra")
                print("="*60 + "\n")
                resp.raise_for_status()
            
            # Manejar error 429 (Too Many Requests)
            if resp.status_code == 429:
                if retry_count < self.max_retries:
                    retry_after = int(resp.headers.get("Retry-After", self.retry_delay))
                    print(f"Rate limit alcanzado. Esperando {retry_after} segundos antes de reintentar...")
                    time.sleep(retry_after)
                    return self._get(path, params, retry_count + 1)
                else:
                    resp.raise_for_status()
            
            resp.raise_for_status()
            return resp.json()
            
        except requests.exceptions.HTTPError as e:
            # Re-lanzar el error con información adicional si es 403
            if hasattr(e, 'response') and e.response.status_code == 403:
                raise
            raise
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries and "429" in str(e):
                print(f"Error en request. Esperando {self.retry_delay} segundos antes de reintentar...")
                time.sleep(self.retry_delay)
                return self._get(path, params, retry_count + 1)
            raise

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

    def get_clans(self, min_score: int = None):
        """Get clans with optional minimum score filter"""
        params = {}
        if min_score is not None:
            params["minScore"] = min_score
        return self._get("clans", params=params)

    def test_connection(self):
        """Test API connection and token validity."""
        print("Testing API connection...")
        print(f"Base URL: {self.base_url}")
        print(f"Token present: {bool(self.token)}")
        if self.token:
            print(f"Token length: {len(self.token)}")
            print(f"Token preview: {self.token[:30]}...{self.token[-10:]}")
            print(f"Token has whitespace: {self.token != self.token.strip()}")
        else:
            print("ERROR: Token is None or empty!")
            print("\nTroubleshooting:")
            print("1. Check that config/config.yaml exists")
            print("2. Verify the token is set in config.yaml under 'api.token'")
            print("3. Check that the token value is not empty or None")
            return False
        print(f"Headers: {list(self.headers.keys())}")
        if self.headers:
            auth_header = self.headers.get('Authorization', '')
            print(f"Authorization header: {auth_header[:50]}...")
        
        try:
            # Try a simple endpoint
            result = self.get_cards()
            print("✓ Connection successful! Token is valid.")
            if isinstance(result, dict):
                items_count = len(result.get('items', []))
                support_count = len(result.get('supportItems', []))
                print(f"✓ Found {items_count} items and {support_count} support items")
            return True
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response.status_code == 403:
                print("✗ Connection failed: Token is invalid or expired")
            else:
                print(f"✗ Connection failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False