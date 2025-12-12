# 🌍 Land Type Classification System - Geospatial Intelligence Platform

A comprehensive deep-learning system for land cover classification from satellite imagery, upgraded with advanced geospatial intelligence features. The project implements transfer learning with EfficientNetB0 fine-tuned on NWPU-RESISC45 dataset, plus a full-featured web interface with video analysis, coordinate-based image retrieval, and land suitability recommendations.

## 📋 Features

### Core Functionality
- **Image Classification**: Upload satellite images and get land type predictions with confidence scores
- **Video Analysis**: Process video files frame-by-frame with configurable FPS extraction and prediction aggregation
- **Location-Based Analysis**: Enter coordinates to retrieve satellite imagery from free tile servers and classify land types
- **Geographic Information**: Automatic reverse geocoding to display country, region, city, and administrative details
- **Land Suitability Recommendations**: Rule-based engine providing 3-5 recommendations per land type with explanations
- **Persistent History Management**: Full CRUD operations for prediction history with SQLite database storage
- **Interactive Dashboard**: Visual analytics with charts, maps, and timeline views
- **Excel Export**: Export individual or bulk predictions to Excel format
- **Image Enhancement Pipeline**: Preprocessing to improve image quality before prediction (sharpness, brightness, contrast, saturation, denoising, upscaling)
- **Optimized UI Display**: Thumbnail previews and expandable full-size views to save space and improve layout
- **Video Frame Preview Grid**: Visual preview of extracted video frames before analysis

### Technical Highlights
- **Model**: EfficientNetB0 fine-tuned on NWPU-RESISC45 (45 classes)
- **Free APIs Only**: All external services use free, open-source APIs (no authentication required)
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Extensible**: Easy to add new land types, recommendations, and features

## 🗂️ Dataset

- **NWPU-RESISC45**: 45 land cover classes, ~31,500 images (~700 per class)
- Images standardized to 224×224 with ImageNet normalization
- Directory expected at `./NWPU-RESISC45/` (optional, class names auto-detected)

Example classes: `airplane, airport, beach, bridge, cloud, desert, forest, grassland, lake, mountain, urban, water, ...`

## 🏗️ Project Structure

```
.
├── app.py                          # Main Streamlit application
├── best_efficientnet_model.pth     # Saved fine-tuned model weights
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model_inference/                # Model loading and prediction
│   ├── __init__.py
│   ├── model_loader.py            # Model loading utilities
│   └── predictor.py                # Prediction interface
│
├── video_processing/               # Video frame extraction
│   ├── __init__.py
│   └── video_processor.py          # Video processing utilities
│
├── image_retrieval/                # Coordinate-based image fetching
│   ├── __init__.py
│   └── tile_retriever.py           # Free tile server integration
│
├── geolocation/                    # Reverse geocoding
│   ├── __init__.py
│   └── geocoder.py                 # OpenStreetMap Nominatim integration
│
├── recommendations/               # Land suitability engine
│   ├── __init__.py
│   └── recommendation_engine.py    # Rule-based recommendations
│
├── history_manager/                # History management system
│   ├── __init__.py
│   ├── history_db.py                # SQLite database operations
│   └── export_utils.py             # Excel export utilities
│
├── dashboard/                      # Visualization dashboard
│   ├── __init__.py
│   └── visualizations.py           # Plotly chart generation
│
├── preprocessing/                  # Image enhancement and preprocessing
│   ├── __init__.py
│   └── image_quality.py            # Image quality enhancement utilities
│
├── ui/                             # UI display utilities
│   ├── __init__.py
│   └── display_utils.py            # Thumbnail and preview rendering
│
├── streamlit_app/                  # Streamlit page modules
│   ├── __init__.py
│   └── pages.py                    # Individual page implementations
│
├── config/                         # Configuration files
│   └── recommendations.json        # Land suitability rules
│
└── prediction_history.db           # SQLite database (created automatically)
│
└── NWPU-RESISC45/                  # Dataset directory (optional)
    ├── airplane/
    ├── airport/
    ├── beach/
    └── ... (45 classes)
```

## 🚀 Quickstart

### 1. Installation

```bash
# Clone or download the project
cd Land-Type-Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup

Ensure `best_efficientnet_model.pth` is in the project root directory (same folder as `app.py`).

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 📖 Usage Guide

### Image Upload Page
1. Navigate to "Image Upload" in the sidebar
2. Upload a satellite image (JPG, PNG, BMP)
3. (Optional) Configure image enhancement settings:
   - Enhance Sharpness, Brightness, Contrast
   - Optional: Saturation, Noise Reduction, Upscaling
4. Adjust Top-K and confidence threshold sliders
5. View thumbnail preview with expandable full-size view
6. Optionally compare original vs enhanced images
7. View predictions and land suitability recommendations

### Video Analysis Page
1. Navigate to "Video Analysis"
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Set frames per second (FPS) for extraction (default: 1.0)
4. Choose aggregation method (majority voting or weighted average)
5. (Optional) Configure frame enhancement settings (same as image enhancement)
6. Click "Analyze Video" to process
7. View video preview and frame preview grid (thumbnail grid of extracted frames)
8. View overall prediction, frame-by-frame timeline, and recommendations

### Location Analysis Page
1. Navigate to "Location Analysis"
2. Enter latitude and longitude coordinates
3. Adjust zoom level (10-18, default: 15)
4. Toggle satellite imagery option
5. (Optional) Configure image enhancement settings
6. Click "Analyze Location"
7. View retrieved image (thumbnail with expandable full-size view)
8. Optionally compare original vs enhanced images
9. View predictions, location metadata, and recommendations

### History Management Page
- **View All Predictions**: Browse all saved predictions in an expandable table view
- **Filter & Search**: Filter by input type, land type, confidence threshold, or search by text
- **Edit Records**: Modify prediction class, confidence, notes, and location names
- **Delete Records**: Remove individual predictions or clear all history
- **Export to Excel**: Export individual predictions or bulk export all data
- **Interactive Dashboard**: 
  - Land type distribution (pie/bar charts)
  - Input type distribution
  - Prediction timeline
  - Map view of coordinate-based predictions
- **Statistics**: View total predictions, average confidence, and coordinate counts

## 💾 Historical Data Management

### Saving Predictions
All prediction pages (Image Upload, Video Analysis, Location Analysis) include a **"Save to History"** button that stores:
- Timestamp
- Input type (image/video/coordinates)
- Prediction class and confidence
- Top-K predictions
- Recommendations
- Metadata (coordinates, location info, video settings, etc.)
- User notes (editable)

### Database Storage
- Uses **SQLite** database (`prediction_history.db`) for persistent storage
- Automatically created on first use
- Thread-safe for multi-user scenarios
- No external database server required

### History Management Features
- **CRUD Operations**: Create, Read, Update, Delete predictions
- **Advanced Filtering**: By input type, land type, confidence, date range
- **Text Search**: Search across prediction classes, locations, and notes
- **Bulk Operations**: Export all data or clear entire history
- **Data Visualization**: Interactive charts and maps using Plotly

### Excel Export
- Export individual predictions or entire history
- Includes all prediction data, metadata, and recommendations
- Auto-formatted columns with proper widths
- Compatible with Excel and Google Sheets

## 🖼️ Image Enhancement & Preprocessing

### Image Quality Enhancement
The system includes an optional image enhancement pipeline that improves image quality before model prediction:

**Available Enhancements:**
- **Sharpness**: Increases image sharpness (default: 20% enhancement)
- **Brightness**: Adjusts brightness (default: 5% increase)
- **Contrast**: Enhances contrast (default: 10% increase)
- **Saturation**: Optional color saturation boost
- **Noise Reduction**: OpenCV-based denoising for noisy images
- **Upscaling**: Automatically upscales images smaller than 224×224 using LANCZOS resampling

**Usage:**
- All prediction pages include an expandable "Image Enhancement Settings" section
- Enable/disable individual enhancements as needed
- Enhanced images are used for prediction while original images are preserved
- Option to view side-by-side comparison of original vs enhanced images

### UI Display Optimization
- **Thumbnail Previews**: Images displayed at reduced size (300×300px max) to save space
- **Expandable Full-Size View**: Click to expand and view full-resolution images
- **Video Frame Grid**: Preview grid showing extracted video frames as thumbnails
- **Responsive Layout**: All displays maintain aspect ratio and adapt to screen size

## 🔧 Configuration

### Recommendations Engine

Edit `config/recommendations.json` to customize land suitability recommendations:

```json
{
  "desert": [
    {
      "use": "Solar Farm",
      "explanation": "Deserts receive abundant sunlight..."
    }
  ]
}
```

The engine automatically maps specific land types (e.g., `lake`, `river`) to general categories (`water`) for recommendations.

### Video Processing

Adjust FPS in the Streamlit interface or modify `VideoProcessor` defaults in `video_processing/video_processor.py`.

### Image Retrieval

Modify tile providers or zoom levels in `image_retrieval/tile_retriever.py`. Supported providers:
- **OpenStreetMap** (`osm`): Standard map tiles
- **CartoDB** (`cartodb`): Light map style
- **Esri World Imagery** (`esri`): Satellite-like imagery (default)

## 🧪 Model Details

### Architecture
- **Base Model**: EfficientNetB0 (ImageNet pre-trained)
- **Custom Head**: 
  - Dropout (0.3) → Linear (512) → ReLU → Dropout (0.2) → Linear (256) → ReLU → Dropout (0.1) → Linear (45)
- **Input**: 224×224 RGB images
- **Normalization**: ImageNet statistics

### Performance
- Validation Accuracy: ~96.3% (from training notebook)
- Classes: 45 land cover types
- Inference: GPU-accelerated (CUDA) when available

## 🌐 Free APIs Used

### Image Retrieval
- **OpenStreetMap Tiles**: `https://tile.openstreetmap.org/`
- **CartoDB Basemaps**: `https://a.basemaps.cartocdn.com/`
- **Esri World Imagery**: `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/`

All tile servers are free and require no API keys.

### Geolocation
- **OpenStreetMap Nominatim**: `https://nominatim.openstreetmap.org/`
  - Free reverse geocoding
  - Rate limit: 1 request/second (automatically handled)
  - No API key required

## 🛠️ Tech Stack

- **Python 3.8+**
- **PyTorch 2.0+**: Deep learning framework
- **EfficientNet-PyTorch**: Pre-trained model
- **Streamlit**: Web application framework
- **OpenCV**: Video processing
- **Pillow**: Image processing
- **Requests**: HTTP API calls
- **NumPy, Pandas**: Data manipulation
- **Matplotlib**: Static visualizations
- **Plotly**: Interactive dashboard visualizations
- **OpenPyXL**: Excel file generation
- **SQLite3**: Built-in database (Python standard library)
- **PIL/Pillow**: Image processing and enhancement
- **OpenCV**: Advanced image preprocessing (denoising, etc.)

## 📊 Results

From training notebook (`LTC.ipynb`):
- **ResNet50 Baseline**: ~81.9% validation accuracy
- **EfficientNetB0**: ~96.3% validation accuracy

Per-class metrics, confusion matrices, and comparative plots available in the notebook.

## 🔄 Extending the System

### Adding New Land Types
1. Retrain model with new classes (update dataset)
2. Update `DEFAULT_CLASSES` in `model_inference/model_loader.py`
3. Add mapping in `recommendations/recommendation_engine.py` if needed

### Adding New Recommendations
1. Edit `config/recommendations.json`
2. Add entries for new categories or update existing ones
3. Restart the application

### Adding New Tile Providers
1. Implement provider URL logic in `image_retrieval/tile_retriever.py`
2. Add provider option to `get_tile_url()` method
3. Update Streamlit interface if needed

## ⚠️ Notes

- **Model File**: Ensure `best_efficientnet_model.pth` exists in project root
- **Rate Limiting**: Nominatim API requires 1 request/second (handled automatically)
- **Video Processing**: Large videos may take time; consider adjusting FPS
- **Tile Retrieval**: Some coordinates may not have satellite imagery available
- **Database**: SQLite database (`prediction_history.db`) is created automatically in project root
- **History Persistence**: All saved predictions persist across app restarts
- **Export Files**: Excel exports are generated on-demand and downloaded to your device

## 🧭 Roadmap

- [x] Persistent prediction history (SQLite database)
- [x] Export prediction results to Excel
- [x] Interactive dashboard with visualizations
- [ ] Add Grad-CAM visualization for model interpretability
- [ ] Support for batch image processing
- [ ] Export to CSV/JSON formats
- [ ] Additional tile providers and imagery sources
- [ ] Real-time video stream processing
- [ ] Multi-model ensemble predictions
- [ ] History backup/restore functionality

## 📚 References

- **NWPU-RESISC45 Dataset**: http://www.escience.cn/people/gongcheng/NWPU-RESISC45.html
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **OpenStreetMap**: https://www.openstreetmap.org/
- **Nominatim API**: https://nominatim.org/
- **PyTorch**: https://pytorch.org/
- **Streamlit**: https://streamlit.io/

## 👥 Authors

**Abdelrhman Al Ghonimi**
- Email: abdelrhmanalghonimi@gmail.com
- GitHub: https://github.com/Abdelrhman-AlGhonimi
- LinkedIn: https://www.linkedin.com/in/abdelrhman-al-ghonimi-2005902a6/
- Portfolio: https://ghonimi.vercel.app/

**Samah Ashraf Saad Elmenady**
- Email: samahelmenady@gmail.com
- GitHub: https://github.com/samahelmenady
- LinkedIn: https://www.linkedin.com/in/samah-elmenady-9937182a2/
- Portfolio: https://samahelmenady-portfolio.vercel.app/

**Ziad Ahmed Sobhy Omran**
- Email: ziadomran814@gmail.com
- GitHub: https://github.com/ZIAD-OMRAN
- LinkedIn: https://www.linkedin.com/in/ziad-omran5
- Portfolio: https://ziad-omran.github.io/ziadomran_portfolio/

**Mohamed Osama Abdelaleem Elsheikh**
- Email: xmoosamax@gmail.com
- GitHub: https://github.com/m7mddosamaa
- LinkedIn: https://www.linkedin.com/in/muhammadelsheikhh/

**Yasmin Mohsen Abdelghafour Hammad**
- Email: yacmmeen@gmail.com
- GitHub: https://github.com/engyasmeen464574574558
- LinkedIn: https://www.linkedin.com/in/engyasminmohsen1315
- Portfolio: https://engyasmeen464574574558.github.io/PROTOFOLIO/

## 📄 License

This project is open-source and available for educational and research purposes.

---

**Built with ❤️ using PyTorch, Streamlit, and free open-source APIs.**
