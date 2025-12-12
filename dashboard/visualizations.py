"""Visualization functions for dashboard."""

import pandas as pd
from typing import List, Dict, Optional

# Lazy import plotly to handle missing dependency gracefully
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None


def create_land_type_chart(predictions: List[Dict], chart_type: str = 'pie') -> Optional:
    """
    Create a chart showing distribution of land types.
    
    Args:
        predictions: List of prediction dictionaries
        chart_type: Type of chart ('pie' or 'bar')
    
    Returns:
        Plotly figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    class_counts = df['prediction_class'].value_counts()
    
    if chart_type == 'pie':
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Land Type Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
    else:  # bar chart
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title='Land Type Distribution',
            labels={'x': 'Land Type', 'y': 'Count'},
            color=class_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(showlegend=False)
    
    fig.update_layout(height=400)
    return fig


def create_input_type_chart(predictions: List[Dict]) -> Optional:
    """
    Create a chart showing distribution by input type.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Plotly figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    type_counts = df['input_type'].value_counts()
    
    fig = px.bar(
        x=type_counts.index,
        y=type_counts.values,
        title='Predictions by Input Type',
        labels={'x': 'Input Type', 'y': 'Count'},
        color=type_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=300, showlegend=False)
    return fig


def create_timeline_chart(predictions: List[Dict]) -> Optional:
    """
    Create a timeline chart of predictions over time.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Plotly figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Group by date
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        title='Prediction Timeline',
        labels={'date': 'Date', 'count': 'Number of Predictions'},
        markers=True
    )
    fig.update_layout(height=300)
    return fig


def create_map_view(predictions: List[Dict]) -> Optional:
    """
    Create a map view showing coordinates of predictions.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Plotly figure object or None if plotly not available or no coordinates
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Filter predictions with coordinates
    coords_predictions = [
        p for p in predictions
        if p.get('latitude') is not None and p.get('longitude') is not None
    ]
    
    if not coords_predictions:
        return None
    
    df = pd.DataFrame(coords_predictions)
    
    # Create scatter mapbox
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color='prediction_class',
        size='confidence',
        hover_data=['prediction_class', 'confidence', 'location_name', 'timestamp'],
        title='Prediction Locations',
        zoom=2,
        height=500
    )
    
    fig.update_layout(
        mapbox_style='open-street-map',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def create_confidence_distribution(predictions: List[Dict]) -> Optional:
    """
    Create a histogram of confidence scores.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Plotly figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    
    fig = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title='Confidence Score Distribution',
        labels={'confidence': 'Confidence Score', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(height=300, showlegend=False)
    return fig

