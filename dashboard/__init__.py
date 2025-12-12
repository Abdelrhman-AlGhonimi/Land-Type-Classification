"""Dashboard module for data visualization."""

try:
    from .visualizations import create_land_type_chart, create_input_type_chart, create_timeline_chart, create_map_view
    __all__ = ['create_land_type_chart', 'create_input_type_chart', 'create_timeline_chart', 'create_map_view']
except ImportError:
    # If plotly is not available, create dummy functions
    def create_land_type_chart(*args, **kwargs):
        return None
    def create_input_type_chart(*args, **kwargs):
        return None
    def create_timeline_chart(*args, **kwargs):
        return None
    def create_map_view(*args, **kwargs):
        return None
    __all__ = ['create_land_type_chart', 'create_input_type_chart', 'create_timeline_chart', 'create_map_view']

