"""Rule-based land suitability recommendation engine."""

import json
import os
from typing import List, Dict, Optional


class RecommendationEngine:
    """Generate land suitability recommendations based on land type classification."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize recommendation engine.
        
        Args:
            config_path: Path to JSON configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'recommendations.json')
        
        self.config_path = config_path
        self.recommendations = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load recommendations configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Return default recommendations if config doesn't exist
                return self._get_default_recommendations()
        except Exception as e:
            print(f"Error loading recommendations config: {e}")
            return self._get_default_recommendations()
    
    def _get_default_recommendations(self) -> Dict:
        """Get default recommendations if config file is missing."""
        return {
            "desert": [
                {
                    "use": "Solar Farm",
                    "explanation": "Deserts receive abundant sunlight, making them ideal for large-scale solar energy generation."
                },
                {
                    "use": "Military Base",
                    "explanation": "Desert terrain provides strategic isolation and training grounds for military operations."
                },
                {
                    "use": "Airport Runway",
                    "explanation": "Flat desert terrain is suitable for airport infrastructure with minimal environmental impact."
                },
                {
                    "use": "Research Station",
                    "explanation": "Desert environments are ideal for climate and astronomical research facilities."
                }
            ],
            "water": [
                {
                    "use": "Fishing Operations",
                    "explanation": "Water bodies support commercial and recreational fishing activities."
                },
                {
                    "use": "Water Management Station",
                    "explanation": "Strategic location for water quality monitoring and resource management."
                },
                {
                    "use": "Marine Research",
                    "explanation": "Ideal for aquatic ecosystem studies and marine biodiversity research."
                },
                {
                    "use": "Recreational Water Sports",
                    "explanation": "Suitable for boating, swimming, and water-based tourism activities."
                }
            ],
            "forest": [
                {
                    "use": "Eco-Tourism",
                    "explanation": "Forests provide natural habitats for wildlife viewing and nature-based tourism."
                },
                {
                    "use": "Wildlife Protection Area",
                    "explanation": "Critical for biodiversity conservation and habitat preservation."
                },
                {
                    "use": "Sustainable Logging",
                    "explanation": "Managed forestry operations can support sustainable timber production."
                },
                {
                    "use": "Carbon Sequestration Project",
                    "explanation": "Forests are essential for carbon capture and climate change mitigation."
                }
            ],
            "urban": [
                {
                    "use": "Commercial Development",
                    "explanation": "Urban areas are prime locations for shopping malls and retail centers."
                },
                {
                    "use": "Residential Complex",
                    "explanation": "High-density housing development to accommodate growing populations."
                },
                {
                    "use": "Public Infrastructure",
                    "explanation": "Suitable for schools, hospitals, and community facilities."
                },
                {
                    "use": "Transportation Hub",
                    "explanation": "Ideal for bus stations, metro lines, and intermodal transport facilities."
                }
            ],
            "grassland": [
                {
                    "use": "Agriculture",
                    "explanation": "Fertile grasslands support crop cultivation and livestock grazing."
                },
                {
                    "use": "Sports Facilities",
                    "explanation": "Flat terrain is perfect for football fields, golf courses, and athletic tracks."
                },
                {
                    "use": "Wind Farm",
                    "explanation": "Open grasslands provide consistent wind patterns for renewable energy generation."
                },
                {
                    "use": "Grazing Land",
                    "explanation": "Natural pastures support sustainable livestock farming operations."
                }
            ],
            "mountain": [
                {
                    "use": "Ski Resort",
                    "explanation": "Mountainous terrain provides ideal conditions for winter sports and tourism."
                },
                {
                    "use": "Hydropower Plant",
                    "explanation": "Elevation differences enable efficient hydroelectric power generation."
                },
                {
                    "use": "Mining Operations",
                    "explanation": "Mountains often contain valuable mineral deposits for extraction."
                },
                {
                    "use": "Telecommunications Tower",
                    "explanation": "High elevation provides optimal signal coverage for communication networks."
                }
            ],
            "agricultural": [
                {
                    "use": "Crop Production",
                    "explanation": "Farmland is optimized for growing cereals, vegetables, and cash crops."
                },
                {
                    "use": "Agro-Tourism",
                    "explanation": "Educational farm visits and agricultural experience programs."
                },
                {
                    "use": "Greenhouse Complex",
                    "explanation": "Controlled environment agriculture for year-round production."
                },
                {
                    "use": "Agricultural Research Station",
                    "explanation": "Field testing and development of improved crop varieties."
                }
            ],
            "default": [
                {
                    "use": "General Development",
                    "explanation": "Area suitable for various development projects based on local planning regulations."
                },
                {
                    "use": "Environmental Assessment",
                    "explanation": "Conduct detailed environmental impact studies before development."
                },
                {
                    "use": "Infrastructure Planning",
                    "explanation": "Evaluate connectivity and utility access for potential projects."
                }
            ]
        }
    
    def get_recommendations(self, land_type: str, top_n: int = 5) -> List[Dict]:
        """
        Get recommendations for a given land type.
        
        Args:
            land_type: Predicted land type class name
            top_n: Maximum number of recommendations to return
        
        Returns:
            List of recommendation dictionaries with 'use' and 'explanation' keys
        """
        # Normalize land type name
        land_type_lower = land_type.lower()
        
        # Map specific classes to general categories
        category_mapping = {
            'desert': 'desert',
            'lake': 'water',
            'river': 'water',
            'harbor': 'water',
            'sea_ice': 'water',
            'wetland': 'water',
            'forest': 'forest',
            'chaparral': 'forest',
            'commercial_area': 'urban',
            'dense_residential': 'urban',
            'medium_residential': 'urban',
            'sparse_residential': 'urban',
            'industrial_area': 'urban',
            'parking_lot': 'urban',
            'stadium': 'urban',
            'meadow': 'grassland',
            'golf_course': 'grassland',
            'ground_track_field': 'grassland',
            'mountain': 'mountain',
            'circular_farmland': 'agricultural',
            'rectangular_farmland': 'agricultural',
            'beach': 'default',
            'bridge': 'default',
            'airport': 'default',
            'runway': 'default',
            'railway': 'default',
            'freeway': 'default'
        }
        
        # Get category
        category = category_mapping.get(land_type_lower, 'default')
        
        # Get recommendations for category
        recommendations = self.recommendations.get(category, self.recommendations.get('default', []))
        
        # Return top N recommendations
        return recommendations[:top_n]
    
    def format_recommendations(self, recommendations: List[Dict]) -> str:
        """
        Format recommendations as a readable string.
        
        Args:
            recommendations: List of recommendation dictionaries
        
        Returns:
            Formatted string
        """
        if not recommendations:
            return "No recommendations available."
        
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(f"{i}. **{rec['use']}**\n   {rec['explanation']}")
        
        return "\n\n".join(formatted)

