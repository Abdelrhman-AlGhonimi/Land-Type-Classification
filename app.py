"""
Main Streamlit application for Land Type Classification System.
Upgraded geospatial intelligence system with multiple features.
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit_app.pages import (
    page_image_upload,
    page_video_upload,
    page_coordinates,
    page_history
)

# Page configuration
st.set_page_config(
    page_title="Land Type Classification System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🌍 Land Type Classification")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Image Upload", "Video Analysis", "Location Analysis", "History"],
    index=0
)

# Main content area
if page == "Image Upload":
    page_image_upload()
elif page == "Video Analysis":
    page_video_upload()
elif page == "Location Analysis":
    page_coordinates()
elif page == "History":
    page_history()

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This system uses **EfficientNetB0** fine-tuned on the **NWPU-RESISC45** dataset 
(45 land cover classes).

### Features:
- 📸 **Image Classification**: Upload and classify satellite images
- 🎥 **Video Analysis**: Frame-by-frame land type analysis
- 🌍 **Location Analysis**: Get images from coordinates and classify
- 💡 **Recommendations**: Land suitability suggestions
- 📊 **History**: Track your predictions

### Model:
- Architecture: EfficientNetB0
- Classes: 45 land types
- Input: 224×224 RGB images
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note**: All services use free APIs (no authentication required).")
