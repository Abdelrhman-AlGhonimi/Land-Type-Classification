"""Model loading and preprocessing utilities."""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Configuration
MODEL_PATH = "best_efficientnet_model.pth"
IMAGE_SIZE = 224

# Default class names (45 classes from NWPU-RESISC45)
DEFAULT_CLASSES = [
    'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 
    'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 
    'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 
    'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 
    'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 
    'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 
    'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 
    'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 
    'terrace', 'thermal_power_station', 'wetland'
]


def load_model(num_classes: int = None):
    """
    Load the EfficientNetB0 model with fine-tuned weights.
    
    Args:
        num_classes: Number of classes. If None, inferred from dataset or defaults to 45.
    
    Returns:
        tuple: (model, device)
    """
    if num_classes is None:
        classes = get_classes()
        num_classes = len(classes)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please ensure 'best_efficientnet_model.pth' is in the project root directory."
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_name('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    )
    
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)
        return model, device
    except Exception as e:
        raise RuntimeError(
            f"Error loading model from {MODEL_PATH}:\n{str(e)}\n"
            f"Please ensure the model file is valid and matches the expected architecture."
        )


def get_classes():
    """
    Get the list of class names.
    First tries to infer from dataset directory, otherwise returns default list.
    
    Returns:
        list: List of class names
    """
    data_dir = os.path.join(os.getcwd(), 'NWPU-RESISC45')
    if os.path.isdir(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
        if len(classes) == 45:
            return classes
    return DEFAULT_CLASSES


def get_transform():
    """
    Get the image preprocessing transform pipeline.
    
    Returns:
        transforms.Compose: ImageNet normalization transform
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

