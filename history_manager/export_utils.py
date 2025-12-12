"""Utilities for exporting history to Excel."""

import pandas as pd
from typing import List, Dict, Optional
import io

# Check if openpyxl is available
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def export_to_excel(predictions: List[Dict], filename: str = None) -> Optional[bytes]:
    """
    Export predictions to Excel format.
    
    Args:
        predictions: List of prediction dictionaries
        filename: Optional filename (not used, returns bytes)
    
    Returns:
        Excel file as bytes, or None if openpyxl is not available
    """
    if not OPENPYXL_AVAILABLE:
        return None
    
    if not predictions:
        # Return empty Excel file
        df = pd.DataFrame()
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()
    
    # Prepare data for export
    export_data = []
    for pred in predictions:
        row = {
            'ID': pred.get('id'),
            'Timestamp': pred.get('timestamp'),
            'Input Type': pred.get('input_type'),
            'Prediction Class': pred.get('prediction_class'),
            'Confidence': pred.get('confidence'),
            'Location Name': pred.get('location_name', ''),
            'Latitude': pred.get('latitude', ''),
            'Longitude': pred.get('longitude', ''),
            'Notes': pred.get('notes', '')
        }
        
        # Add top predictions as comma-separated string
        top_preds = pred.get('top_predictions', [])
        if top_preds:
            row['Top Predictions'] = ', '.join([f"{p[0]} ({p[1]:.3f})" for p in top_preds])
        else:
            row['Top Predictions'] = ''
        
        # Add recommendations as formatted string
        recommendations = pred.get('recommendations', [])
        if recommendations:
            rec_text = '; '.join([f"{r.get('use', '')}: {r.get('explanation', '')}" for r in recommendations])
            row['Recommendations'] = rec_text
        else:
            row['Recommendations'] = ''
        
        # Add metadata as JSON string if available
        metadata = pred.get('metadata', {})
        if metadata:
            import json
            row['Metadata'] = json.dumps(metadata)
        else:
            row['Metadata'] = ''
        
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Create Excel file in memory
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Predictions']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        return output.getvalue()
    except Exception as e:
        # If openpyxl is not available or other error, return None
        return None

