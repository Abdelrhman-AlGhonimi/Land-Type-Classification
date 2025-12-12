"""Retrieve satellite-like images from coordinates using free tile servers."""

import requests
from PIL import Image
import io
from typing import Tuple, Optional


class TileRetriever:
    """Retrieve satellite-like images from coordinates using free tile servers."""
    
    def __init__(self, tile_size: int = 512, zoom_level: int = 15):
        """
        Initialize tile retriever.
        
        Args:
            tile_size: Size of the retrieved image in pixels (default: 512)
            zoom_level: Zoom level for map tiles (default: 15, range: 1-19)
        """
        self.tile_size = tile_size
        self.zoom_level = zoom_level
    
    def deg2num(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """
        Convert lat/lon to tile coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
        
        Returns:
            Tuple of (tile_x, tile_y)
        """
        import math
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return tile_x, tile_y
    
    def num2deg(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
        """
        Convert tile coordinates to lat/lon of top-left corner.
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Zoom level
        
        Returns:
            Tuple of (latitude, longitude)
        """
        import math
        n = 2.0 ** zoom
        lon_deg = tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg
    
    def get_tile_url(self, tile_x: int, tile_y: int, zoom: int, provider: str = 'osm') -> str:
        """
        Get tile URL for a given tile coordinate and provider.
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Zoom level
            provider: Tile provider ('osm', 'cartodb', 'esri')
        
        Returns:
            Tile URL string
        """
        if provider == 'osm':
            # OpenStreetMap (free, no API key required)
            return f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
        elif provider == 'cartodb':
            # CartoDB Positron (free, no API key)
            return f"https://a.basemaps.cartocdn.com/light_all/{zoom}/{tile_x}/{tile_y}.png"
        elif provider == 'esri':
            # Esri World Imagery (free, no API key, satellite-like)
            return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def fetch_tile(self, lat: float, lon: float, provider: str = 'esri') -> Optional[Image.Image]:
        """
        Fetch a single tile image for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            provider: Tile provider ('osm', 'cartodb', 'esri')
        
        Returns:
            PIL Image or None if fetch failed
        """
        tile_x, tile_y = self.deg2num(lat, lon, self.zoom_level)
        url = self.get_tile_url(tile_x, tile_y, self.zoom_level, provider)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error fetching tile: {e}")
            return None
    
    def fetch_area(self, lat: float, lon: float, tiles_wide: int = 2, tiles_high: int = 2, 
                   provider: str = 'esri') -> Optional[Image.Image]:
        """
        Fetch a larger area by combining multiple tiles.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            tiles_wide: Number of tiles horizontally
            tiles_high: Number of tiles vertically
            provider: Tile provider
        
        Returns:
            Combined PIL Image or None if fetch failed
        """
        center_tile_x, center_tile_y = self.deg2num(lat, lon, self.zoom_level)
        
        # Calculate tile range
        start_x = center_tile_x - tiles_wide // 2
        end_x = center_tile_x + tiles_wide // 2 + (tiles_wide % 2)
        start_y = center_tile_y - tiles_high // 2
        end_y = center_tile_y + tiles_high // 2 + (tiles_high % 2)
        
        tiles = []
        for ty in range(start_y, end_y):
            row = []
            for tx in range(start_x, end_x):
                url = self.get_tile_url(tx, ty, self.zoom_level, provider)
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    tile_img = Image.open(io.BytesIO(response.content)).convert('RGB')
                    row.append(tile_img)
                except Exception as e:
                    print(f"Error fetching tile ({tx}, {ty}): {e}")
                    # Use a blank tile as fallback
                    blank_tile = Image.new('RGB', (256, 256), color=(200, 200, 200))
                    row.append(blank_tile)
            if row:
                tiles.append(row)
        
        if not tiles or not tiles[0]:
            return None
        
        # Combine tiles into single image
        tile_width = tiles[0][0].width
        tile_height = tiles[0][0].height
        
        combined_width = len(tiles[0]) * tile_width
        combined_height = len(tiles) * tile_height
        
        combined_image = Image.new('RGB', (combined_width, combined_height))
        
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                combined_image.paste(tile, (x * tile_width, y * tile_height))
        
        # Resize to desired output size
        if combined_image.size != (self.tile_size, self.tile_size):
            combined_image = combined_image.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
        
        return combined_image
    
    def get_image_from_coordinates(self, lat: float, lon: float, 
                                   use_satellite: bool = True) -> Optional[Image.Image]:
        """
        Get image from coordinates (main interface method).
        
        Args:
            lat: Latitude
            lon: Longitude
            use_satellite: If True, use satellite-like imagery (esri), else use OSM
        
        Returns:
            PIL Image or None if fetch failed
        """
        provider = 'esri' if use_satellite else 'osm'
        # Fetch a 2x2 tile area for better context
        return self.fetch_area(lat, lon, tiles_wide=2, tiles_high=2, provider=provider)

