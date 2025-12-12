"""SQLite database manager for prediction history."""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager


class HistoryManager:
    """Manage persistent storage of prediction history using SQLite."""
    
    def __init__(self, db_path: str = "prediction_history.db"):
        """
        Initialize history manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    prediction_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    top_predictions TEXT,
                    metadata TEXT,
                    recommendations TEXT,
                    notes TEXT,
                    latitude REAL,
                    longitude REAL,
                    location_name TEXT
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def save_prediction(
        self,
        input_type: str,
        prediction_class: str,
        confidence: float,
        top_predictions: List[tuple] = None,
        metadata: Dict = None,
        recommendations: List[Dict] = None,
        notes: str = None,
        latitude: float = None,
        longitude: float = None,
        location_name: str = None
    ) -> int:
        """
        Save a prediction to history.
        
        Args:
            input_type: Type of input ('image', 'video', 'coordinates')
            prediction_class: Predicted land type class
            confidence: Prediction confidence score
            top_predictions: List of (class, confidence) tuples for top-k predictions
            metadata: Additional metadata dictionary
            recommendations: List of recommendation dictionaries
            notes: Optional user notes
            latitude: Latitude (for coordinate-based predictions)
            longitude: Longitude (for coordinate-based predictions)
            location_name: Location name (for coordinate-based predictions)
        
        Returns:
            ID of the saved record
        """
        timestamp = datetime.now().isoformat()
        
        # Serialize complex data to JSON
        top_predictions_json = json.dumps(top_predictions) if top_predictions else None
        metadata_json = json.dumps(metadata) if metadata else None
        recommendations_json = json.dumps(recommendations) if recommendations else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, input_type, prediction_class, confidence,
                    top_predictions, metadata, recommendations, notes,
                    latitude, longitude, location_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, input_type, prediction_class, confidence,
                top_predictions_json, metadata_json, recommendations_json, notes,
                latitude, longitude, location_name
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_all_predictions(self, limit: int = None) -> List[Dict]:
        """
        Get all predictions from history.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of prediction dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM predictions ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Get a single prediction by ID.
        
        Args:
            prediction_id: ID of the prediction
        
        Returns:
            Prediction dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None
    
    def update_prediction(
        self,
        prediction_id: int,
        prediction_class: str = None,
        confidence: float = None,
        top_predictions: List[tuple] = None,
        metadata: Dict = None,
        recommendations: List[Dict] = None,
        notes: str = None,
        location_name: str = None
    ) -> bool:
        """
        Update an existing prediction.
        
        Args:
            prediction_id: ID of the prediction to update
            prediction_class: New prediction class (optional)
            confidence: New confidence score (optional)
            top_predictions: New top predictions (optional)
            metadata: New metadata (optional)
            recommendations: New recommendations (optional)
            notes: New notes (optional)
            location_name: New location name (optional)
        
        Returns:
            True if update successful, False otherwise
        """
        updates = []
        values = []
        
        if prediction_class is not None:
            updates.append("prediction_class = ?")
            values.append(prediction_class)
        if confidence is not None:
            updates.append("confidence = ?")
            values.append(confidence)
        if top_predictions is not None:
            updates.append("top_predictions = ?")
            values.append(json.dumps(top_predictions))
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(json.dumps(metadata))
        if recommendations is not None:
            updates.append("recommendations = ?")
            values.append(json.dumps(recommendations))
        if notes is not None:
            updates.append("notes = ?")
            values.append(notes)
        if location_name is not None:
            updates.append("location_name = ?")
            values.append(location_name)
        
        if not updates:
            return False
        
        values.append(prediction_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE predictions SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Delete a prediction from history.
        
        Args:
            prediction_id: ID of the prediction to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_all_predictions(self) -> int:
        """
        Delete all predictions from history.
        
        Returns:
            Number of deleted records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            return count
    
    def filter_predictions(
        self,
        input_type: str = None,
        prediction_class: str = None,
        min_confidence: float = None,
        start_date: str = None,
        end_date: str = None,
        has_coordinates: bool = None
    ) -> List[Dict]:
        """
        Filter predictions by various criteria.
        
        Args:
            input_type: Filter by input type
            prediction_class: Filter by prediction class
            min_confidence: Minimum confidence threshold
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            has_coordinates: Filter by presence of coordinates
        
        Returns:
            List of filtered prediction dictionaries
        """
        conditions = []
        values = []
        
        if input_type:
            conditions.append("input_type = ?")
            values.append(input_type)
        if prediction_class:
            conditions.append("prediction_class = ?")
            values.append(prediction_class)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            values.append(min_confidence)
        if start_date:
            conditions.append("timestamp >= ?")
            values.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            values.append(end_date)
        if has_coordinates is not None:
            if has_coordinates:
                conditions.append("latitude IS NOT NULL AND longitude IS NOT NULL")
            else:
                conditions.append("(latitude IS NULL OR longitude IS NULL)")
        
        query = "SELECT * FROM predictions"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def search_predictions(self, search_term: str) -> List[Dict]:
        """
        Search predictions by text in various fields.
        
        Args:
            search_term: Text to search for
        
        Returns:
            List of matching prediction dictionaries
        """
        search_term = f"%{search_term}%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions
                WHERE prediction_class LIKE ?
                   OR location_name LIKE ?
                   OR notes LIKE ?
                ORDER BY timestamp DESC
            """, (search_term, search_term, search_term))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the prediction history.
        
        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_count = cursor.fetchone()[0]
            
            # Count by input type
            cursor.execute("""
                SELECT input_type, COUNT(*) as count
                FROM predictions
                GROUP BY input_type
            """)
            by_input_type = {row['input_type']: row['count'] for row in cursor.fetchall()}
            
            # Count by prediction class
            cursor.execute("""
                SELECT prediction_class, COUNT(*) as count
                FROM predictions
                GROUP BY prediction_class
                ORDER BY count DESC
            """)
            by_class = {row['prediction_class']: row['count'] for row in cursor.fetchall()}
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM predictions")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Records with coordinates
            cursor.execute("""
                SELECT COUNT(*) FROM predictions
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """)
            with_coordinates = cursor.fetchone()[0]
            
            return {
                'total_count': total_count,
                'by_input_type': by_input_type,
                'by_class': by_class,
                'avg_confidence': round(avg_confidence, 3),
                'with_coordinates': with_coordinates
            }
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert SQLite row to dictionary with deserialized JSON fields."""
        if not row:
            return None
        
        result = dict(row)
        
        # Deserialize JSON fields
        if result.get('top_predictions'):
            try:
                result['top_predictions'] = json.loads(result['top_predictions'])
            except:
                result['top_predictions'] = []
        
        if result.get('metadata'):
            try:
                result['metadata'] = json.loads(result['metadata'])
            except:
                result['metadata'] = {}
        
        if result.get('recommendations'):
            try:
                result['recommendations'] = json.loads(result['recommendations'])
            except:
                result['recommendations'] = []
        
        return result

