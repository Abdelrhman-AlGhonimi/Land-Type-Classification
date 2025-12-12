"""Reverse geocoding using free APIs (OpenStreetMap Nominatim)."""

import requests
from typing import Dict, Optional
import time


class Geocoder:
    """Reverse geocoding using OpenStreetMap Nominatim (free, no API key required)."""
    
    def __init__(self, user_agent: str = "LandTypeClassification/1.0"):
        """
        Initialize geocoder.
        
        Args:
            user_agent: User agent string (required by Nominatim)
        """
        self.user_agent = user_agent
        self.base_url = "https://nominatim.openstreetmap.org/reverse"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Nominatim requires max 1 request per second
    
    def _rate_limit(self):
        """Enforce rate limiting for Nominatim API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def reverse_geocode(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Reverse geocode coordinates to get location information.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with location information or None if failed
        """
        self._rate_limit()
        
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1,
            'zoom': 18
        }
        
        headers = {
            'User-Agent': self.user_agent
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                return None
            
            # Extract relevant information
            address = data.get('address', {})
            
            location_info = {
                'display_name': data.get('display_name', 'Unknown'),
                'country': address.get('country', 'Unknown'),
                'country_code': address.get('country_code', ''),
                'state': address.get('state', '') or address.get('region', ''),
                'county': address.get('county', ''),
                'city': address.get('city', '') or address.get('town', '') or address.get('village', ''),
                'postcode': address.get('postcode', ''),
                'road': address.get('road', ''),
                'house_number': address.get('house_number', ''),
                'latitude': lat,
                'longitude': lon,
                'raw_address': address
            }
            
            return location_info
        
        except Exception as e:
            print(f"Error in reverse geocoding: {e}")
            return None
    
    def get_location_summary(self, lat: float, lon: float) -> Dict:
        """
        Get a formatted location summary.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with formatted location summary
        """
        location_info = self.reverse_geocode(lat, lon)
        
        if not location_info:
            return {
                'status': 'error',
                'message': 'Could not retrieve location information',
                'coordinates': {'lat': lat, 'lon': lon}
            }
        
        # Build summary string
        parts = []
        if location_info.get('city'):
            parts.append(location_info['city'])
        if location_info.get('county'):
            parts.append(location_info['county'])
        if location_info.get('state'):
            parts.append(location_info['state'])
        if location_info.get('country'):
            parts.append(location_info['country'])
        
        summary = ', '.join(parts) if parts else location_info.get('display_name', 'Unknown Location')
        
        return {
            'status': 'success',
            'summary': summary,
            'country': location_info.get('country', 'Unknown'),
            'region': location_info.get('state', ''),
            'city': location_info.get('city', ''),
            'county': location_info.get('county', ''),
            'coordinates': {'lat': lat, 'lon': lon},
            'full_display_name': location_info.get('display_name', ''),
            'raw_data': location_info
        }

