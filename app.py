import os
import io
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# -------- Config --------
MODEL_PATH = "best_efficientnet_model.pth"
IMAGE_SIZE = 224

# Class names inferred from dataset directory if available
DEFAULT_CLASSES = [
    'airplane','airport','baseball_diamond','basketball_court','beach','bridge','chaparral','church',
    'circular_farmland','cloud','commercial_area','dense_residential','desert','forest','freeway',
    'golf_course','ground_track_field','harbor','industrial_area','intersection','island','lake','meadow',
    'medium_residential','mobile_home_park','mountain','overpass','palace','parking_lot','railway',
    'railway_station','rectangular_farmland','river','roundabout','runway','sea_ice','ship','snowberg',
    'sparse_residential','stadium','storage_tank','tennis_court','terrace','thermal_power_station','wetland'
]

@st.cache_resource
def load_model(num_classes: int):
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
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model, device

@st.cache_data
def get_classes():
    # Attempt to infer from dataset folder if exists
    data_dir = os.path.join(os.getcwd(), 'NWPU-RESISC45')
    if os.path.isdir(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if len(classes) == 45:
            return classes
    return DEFAULT_CLASSES

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

st.set_page_config(page_title="Land Type Classification", page_icon="", layout="centered")
st.title(" Land Type Classification (EfficientNetB0)")
st.write("Upload a satellite image tile to get the predicted land cover class.")

classes = get_classes()
model, device = load_model(num_classes=len(classes))
transform = get_transform()

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"]) 

col1, col2 = st.columns(2)

with col1:
    topk = st.slider("Top-K predictions", 1, 10, 5)

with col2:
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.01)

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        indices = np.argsort(-probs)[:topk]
        results = [(classes[i], float(probs[i])) for i in indices if probs[i] >= conf_threshold]

    st.subheader("Predictions")
    for label, p in results:
        st.write(f"{label}: {p:.3f}")
        st.progress(min(1.0, p))

    if len(results) == 0:
        st.info("No prediction above the chosen confidence threshold.")

st.sidebar.header("About")
st.sidebar.markdown("This demo uses EfficientNetB0 fine-tuned on NWPU-RESISC45 (45 classes).")
st.sidebar.markdown("Model file expected at est_efficientnet_model.pth in project root.")
