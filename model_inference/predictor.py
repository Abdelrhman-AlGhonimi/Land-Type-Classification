"""Prediction utilities for land type classification."""

import numpy as np
import torch
from PIL import Image
from .model_loader import load_model, get_transform, get_classes


class LandTypePredictor:
    """Predictor class for land type classification."""
    
    def __init__(self):
        """Initialize the predictor with model and classes."""
        self.classes = get_classes()
        self.model, self.device = load_model(num_classes=len(self.classes))
        self.transform = get_transform()
    
    def predict_image(self, image, top_k=5, conf_threshold=0.0):
        """
        Predict land type from a PIL Image.
        
        Args:
            image: PIL Image object (RGB)
            top_k: Number of top predictions to return
            conf_threshold: Minimum confidence threshold
        
        Returns:
            list: List of (class_name, confidence) tuples
        """
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get top-k predictions
        indices = np.argsort(-probs)[:top_k]
        results = [(self.classes[i], float(probs[i])) 
                   for i in indices if probs[i] >= conf_threshold]
        
        return results
    
    def predict_batch(self, images, top_k=5, conf_threshold=0.0):
        """
        Predict land types for a batch of images.
        
        Args:
            images: List of PIL Image objects
            top_k: Number of top predictions per image
            conf_threshold: Minimum confidence threshold
        
        Returns:
            list: List of prediction results for each image
        """
        results = []
        for image in images:
            pred = self.predict_image(image, top_k=top_k, conf_threshold=conf_threshold)
            results.append(pred)
        return results
    
    def aggregate_predictions(self, predictions, method='majority'):
        """
        Aggregate multiple predictions into a single result.
        
        Args:
            predictions: List of prediction results (each is list of (class, conf) tuples)
            method: Aggregation method ('majority' or 'weighted')
        
        Returns:
            tuple: (predicted_class, confidence, all_votes)
        """
        if not predictions:
            return None, 0.0, []
        
        if method == 'majority':
            # Count votes for each class
            class_votes = {}
            for pred_list in predictions:
                if pred_list:
                    top_class = pred_list[0][0]  # Get top prediction
                    class_votes[top_class] = class_votes.get(top_class, 0) + 1
            
            if not class_votes:
                return None, 0.0, []
            
            # Get majority class
            majority_class = max(class_votes, key=class_votes.get)
            confidence = class_votes[majority_class] / len(predictions)
            return majority_class, confidence, list(class_votes.items())
        
        elif method == 'weighted':
            # Weighted average of confidences
            class_scores = {}
            total_weight = 0
            
            for pred_list in predictions:
                if pred_list:
                    top_class, conf = pred_list[0]
                    class_scores[top_class] = class_scores.get(top_class, 0.0) + conf
                    total_weight += conf
            
            if not class_scores:
                return None, 0.0, []
            
            # Get class with highest weighted score
            best_class = max(class_scores, key=class_scores.get)
            confidence = class_scores[best_class] / total_weight if total_weight > 0 else 0.0
            return best_class, confidence, list(class_scores.items())
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

