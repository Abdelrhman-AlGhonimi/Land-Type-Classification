"""Individual page modules for Streamlit app."""

import streamlit as st
import io
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_inference import LandTypePredictor
from video_processing import VideoProcessor
from image_retrieval import TileRetriever
from geolocation import Geocoder
from recommendations import RecommendationEngine
from history_manager import HistoryManager
from preprocessing import ImageEnhancer
from ui.display_utils import display_image_with_expand, display_side_by_side, display_video_preview


def page_image_upload():
    """Image upload and prediction page."""
    st.header("📸 Image Classification")
    st.write("Upload a satellite image to classify the land type.")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.predictor = LandTypePredictor()
    
    predictor = st.session_state.predictor
    
    # Upload widget
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
    
    # Image enhancement options
    with st.expander("⚙️ Image Enhancement Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            enhance_sharpness = st.checkbox("Enhance Sharpness", value=True)
            enhance_brightness = st.checkbox("Enhance Brightness", value=True)
            enhance_contrast = st.checkbox("Enhance Contrast", value=True)
        with col2:
            enhance_saturation = st.checkbox("Enhance Saturation", value=False)
            denoise = st.checkbox("Noise Reduction", value=False)
            upscale_small = st.checkbox("Upscale Small Images", value=False)
    
    col1, col2 = st.columns(2)
    with col1:
        topk = st.slider("Top-K predictions", 1, 10, 5)
    with col2:
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.01)
    
    if uploaded is not None:
        original_image = Image.open(io.BytesIO(uploaded.read())).convert('RGB')
        
        # Initialize enhancer
        enhancer = ImageEnhancer(
            enhance_sharpness=enhance_sharpness,
            enhance_brightness=enhance_brightness,
            enhance_contrast=enhance_contrast,
            enhance_saturation=enhance_saturation,
            denoise=denoise,
            upscale_small=upscale_small
        )
        
        # Enhance image
        with st.spinner("Enhancing image..."):
            enhanced_image = enhancer.enhance(original_image)
        
        # Display images side by side
        st.subheader("📷 Image Preview")
        show_comparison = st.checkbox("Show Original vs Enhanced", value=False)
        
        if show_comparison:
            display_side_by_side(original_image, enhanced_image, "Original", "Enhanced (for prediction)")
        else:
            display_image_with_expand(enhanced_image, caption="Enhanced Image (used for prediction)", key="enhanced_img")
        
        # Run prediction on enhanced image
        with st.spinner("Predicting..."):
            results = predictor.predict_image(enhanced_image, top_k=topk, conf_threshold=conf_threshold)
        
        st.subheader("Predictions")
        if results:
            for label, conf in results:
                st.write(f"**{label}**: {conf:.3f}")
                st.progress(min(1.0, conf))
            
            # Show recommendations for top prediction
            if results:
                top_prediction = results[0][0]
                top_confidence = results[0][1]
                st.subheader("💡 Land Suitability Recommendations")
                rec_engine = RecommendationEngine()
                recommendations = rec_engine.get_recommendations(top_prediction, top_n=5)
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['use']}"):
                        st.write(rec['explanation'])
                
                # Save to History button
                st.markdown("---")
                if st.button("💾 Save to History", key="save_image_prediction"):
                    # Initialize history manager
                    if 'history_manager' not in st.session_state:
                        st.session_state.history_manager = HistoryManager()
                    
                    history_mgr = st.session_state.history_manager
                    prediction_id = history_mgr.save_prediction(
                        input_type='image',
                        prediction_class=top_prediction,
                        confidence=top_confidence,
                        top_predictions=results,
                        recommendations=recommendations,
                        metadata={'filename': uploaded.name if hasattr(uploaded, 'name') else 'uploaded_image'}
                    )
                    st.success(f"✅ Prediction saved to history! (ID: {prediction_id})")
        else:
            st.info("No prediction above the chosen confidence threshold.")


def page_video_upload():
    """Video upload and frame-by-frame analysis page."""
    st.header("🎥 Video Analysis")
    st.write("Upload a video to analyze land types frame by frame.")
    
    # Initialize components
    if 'predictor' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.predictor = LandTypePredictor()
    
    predictor = st.session_state.predictor
    video_processor = VideoProcessor()
    
    # Upload widget
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    
    col1, col2 = st.columns(2)
    with col1:
        fps = st.slider("Frames per second to extract", 0.1, 5.0, 1.0, 0.1)
    with col2:
        aggregation_method = st.selectbox("Aggregation method", ["majority", "weighted"])
    
    # Video enhancement options
    with st.expander("⚙️ Video Frame Enhancement Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            enhance_sharpness = st.checkbox("Enhance Sharpness", value=True, key="vid_sharp")
            enhance_brightness = st.checkbox("Enhance Brightness", value=True, key="vid_bright")
            enhance_contrast = st.checkbox("Enhance Contrast", value=True, key="vid_contrast")
        with col2:
            enhance_saturation = st.checkbox("Enhance Saturation", value=False, key="vid_sat")
            denoise = st.checkbox("Noise Reduction", value=False, key="vid_denoise")
            upscale_small = st.checkbox("Upscale Small Frames", value=False, key="vid_upscale")
    
    if uploaded_video is not None:
        # Show video preview (smaller)
        st.subheader("📹 Video Preview")
        st.video(uploaded_video, format="video/mp4")
        
        if st.button("Analyze Video"):
            with st.spinner("Processing video..."):
                # Process video
                temp_path, frames = video_processor.process_video_file(uploaded_video, fps=fps)
                
                if not frames:
                    st.error("No frames extracted from video.")
                    video_processor.cleanup_temp_file(temp_path)
                    return
                
                st.success(f"Extracted {len(frames)} frames")
                
                # Initialize enhancer
                enhancer = ImageEnhancer(
                    enhance_sharpness=enhance_sharpness,
                    enhance_brightness=enhance_brightness,
                    enhance_contrast=enhance_contrast,
                    enhance_saturation=enhance_saturation,
                    denoise=denoise,
                    upscale_small=upscale_small
                )
                
                # Show video frame preview grid
                display_video_preview(frames, video_name=uploaded_video.name if hasattr(uploaded_video, 'name') else "Video")
                
                # Predict for each frame (with enhancement)
                progress_bar = st.progress(0)
                all_predictions = []
                frame_prediction_lists = []
                
                for idx, (frame, timestamp) in enumerate(frames):
                    # Enhance frame before prediction
                    enhanced_frame = enhancer.enhance(frame)
                    
                    pred = predictor.predict_image(enhanced_frame, top_k=1, conf_threshold=0.0)
                    if pred:
                        class_name, confidence = pred[0]
                        all_predictions.append((timestamp, class_name, confidence))
                        frame_prediction_lists.append(pred)  # Store full prediction list
                    progress_bar.progress((idx + 1) / len(frames))
                
                # Aggregate predictions
                if frame_prediction_lists:
                    aggregated_class, aggregated_conf, votes = predictor.aggregate_predictions(
                        frame_prediction_lists, 
                        method=aggregation_method
                    )
                    
                    if aggregated_class:
                        st.subheader("Overall Video Prediction")
                        st.write(f"**Predicted Land Type**: {aggregated_class}")
                        st.write(f"**Confidence**: {aggregated_conf:.3f}")
                        
                        # Show recommendations
                        st.subheader("💡 Land Suitability Recommendations")
                        rec_engine = RecommendationEngine()
                        recommendations = rec_engine.get_recommendations(aggregated_class, top_n=5)
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"{i}. {rec['use']}"):
                                st.write(rec['explanation'])
                        
                        # Store video prediction data in session state for save button
                        st.session_state.video_prediction_data = {
                            'class': aggregated_class,
                            'confidence': aggregated_conf,
                            'top_predictions': [(aggregated_class, aggregated_conf)],
                            'recommendations': recommendations,
                            'frame_count': len(frames),
                            'aggregation_method': aggregation_method
                        }
                        
                        # Save to History button
                        st.markdown("---")
                        if st.button("💾 Save to History", key="save_video_prediction"):
                            # Initialize history manager
                            if 'history_manager' not in st.session_state:
                                st.session_state.history_manager = HistoryManager()
                            
                            history_mgr = st.session_state.history_manager
                            video_data = st.session_state.video_prediction_data
                            prediction_id = history_mgr.save_prediction(
                                input_type='video',
                                prediction_class=video_data['class'],
                                confidence=video_data['confidence'],
                                top_predictions=video_data['top_predictions'],
                                recommendations=video_data['recommendations'],
                                metadata={
                                    'frame_count': video_data['frame_count'],
                                    'aggregation_method': video_data['aggregation_method'],
                                    'fps': fps
                                }
                            )
                            st.success(f"✅ Prediction saved to history! (ID: {prediction_id})")
                    
                    # Timeline visualization
                    st.subheader("Frame-by-Frame Timeline")
                    if len(all_predictions) > 0:
                        df = pd.DataFrame(all_predictions, columns=['Timestamp', 'Class', 'Confidence'])
                        st.dataframe(df, width='stretch')
                        
                        # Simple timeline chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        timestamps = [p[0] for p in all_predictions]
                        classes = [p[1] for p in all_predictions]
                        ax.scatter(timestamps, [1] * len(timestamps), c=range(len(timestamps)), cmap='viridis')
                        ax.set_xlabel('Time (seconds)')
                        ax.set_ylabel('Frame')
                        ax.set_title('Frame Extraction Timeline')
                        st.pyplot(fig)
                    else:
                        st.warning("No predictions generated from video frames.")
                
                # Cleanup
                video_processor.cleanup_temp_file(temp_path)


def page_coordinates():
    """Coordinates-based image retrieval and analysis page."""
    st.header("🌍 Location-Based Analysis")
    st.write("Enter coordinates to retrieve satellite imagery and classify land type.")
    
    # Initialize components
    if 'predictor' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.predictor = LandTypePredictor()
    
    predictor = st.session_state.predictor
    tile_retriever = TileRetriever()
    geocoder = Geocoder()
    rec_engine = RecommendationEngine()
    
    # Input widgets
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=40.7128, format="%.6f", step=0.0001)
    with col2:
        longitude = st.number_input("Longitude", value=-74.0060, format="%.6f", step=0.0001)
    
    col1, col2 = st.columns(2)
    with col1:
        zoom_level = st.slider("Zoom Level", 10, 18, 15)
    with col2:
        use_satellite = st.checkbox("Use Satellite Imagery", value=True)
    
    # Image enhancement options
    with st.expander("⚙️ Image Enhancement Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            enhance_sharpness = st.checkbox("Enhance Sharpness", value=True, key="coord_sharp")
            enhance_brightness = st.checkbox("Enhance Brightness", value=True, key="coord_bright")
            enhance_contrast = st.checkbox("Enhance Contrast", value=True, key="coord_contrast")
        with col2:
            enhance_saturation = st.checkbox("Enhance Saturation", value=False, key="coord_sat")
            denoise = st.checkbox("Noise Reduction", value=False, key="coord_denoise")
            upscale_small = st.checkbox("Upscale Small Images", value=False, key="coord_upscale")
    
    tile_retriever.zoom_level = zoom_level
    
    # Initialize session state for coordinate analysis
    if 'coord_original_image' not in st.session_state:
        st.session_state.coord_original_image = None
    if 'coord_enhanced_image' not in st.session_state:
        st.session_state.coord_enhanced_image = None
    if 'coord_location_info' not in st.session_state:
        st.session_state.coord_location_info = None
    if 'coord_predictions' not in st.session_state:
        st.session_state.coord_predictions = None
    if 'coord_show_comparison' not in st.session_state:
        st.session_state.coord_show_comparison = False
    
    if st.button("Analyze Location"):
        with st.spinner("Retrieving image and location data..."):
            # Get image from coordinates
            original_image = tile_retriever.get_image_from_coordinates(
                latitude, longitude, use_satellite=use_satellite
            )
            
            # Get location information
            location_info = geocoder.get_location_summary(latitude, longitude)
        
        if original_image:
            # Initialize enhancer
            enhancer = ImageEnhancer(
                enhance_sharpness=enhance_sharpness,
                enhance_brightness=enhance_brightness,
                enhance_contrast=enhance_contrast,
                enhance_saturation=enhance_saturation,
                denoise=denoise,
                upscale_small=upscale_small
            )
            
            # Enhance image
            with st.spinner("Enhancing image..."):
                enhanced_image = enhancer.enhance(original_image)
            
            # Store images and data in session state
            st.session_state.coord_original_image = original_image
            st.session_state.coord_enhanced_image = enhanced_image
            st.session_state.coord_location_info = location_info
            st.session_state.coord_latitude = latitude
            st.session_state.coord_longitude = longitude
            st.session_state.coord_zoom_level = zoom_level
            st.session_state.coord_use_satellite = use_satellite
            
            # Run prediction on enhanced image
            with st.spinner("Classifying land type..."):
                results = predictor.predict_image(enhanced_image, top_k=5, conf_threshold=0.0)
            
            st.session_state.coord_predictions = results
            
            # Store coordinate prediction data in session state
            if results:
                top_prediction = results[0][0]
                recommendations = rec_engine.get_recommendations(top_prediction, top_n=5)
                
                st.session_state.coordinate_prediction_data = {
                    'class': top_prediction,
                    'confidence': results[0][1],
                    'top_predictions': results,
                    'recommendations': recommendations,
                    'latitude': latitude,
                    'longitude': longitude,
                    'location_info': location_info
                }
        else:
            st.error("Could not retrieve image for the specified coordinates. Please try different coordinates or zoom level.")
            # Clear session state on error
            st.session_state.coord_original_image = None
            st.session_state.coord_enhanced_image = None
            st.session_state.coord_location_info = None
            st.session_state.coord_predictions = None
    
    # Display results if available in session state
    if st.session_state.coord_original_image is not None and st.session_state.coord_enhanced_image is not None:
        # Display images
        st.subheader("📍 Retrieved Image")
        show_comparison = st.checkbox(
            "Show Original vs Enhanced", 
            value=st.session_state.coord_show_comparison, 
            key="coord_compare"
        )
        st.session_state.coord_show_comparison = show_comparison
        
        if show_comparison:
            display_side_by_side(
                st.session_state.coord_original_image, 
                st.session_state.coord_enhanced_image, 
                f"Original (Location: {st.session_state.coord_latitude:.6f}, {st.session_state.coord_longitude:.6f})",
                "Enhanced (for prediction)"
            )
        else:
            display_image_with_expand(
                st.session_state.coord_enhanced_image, 
                caption=f"Location: {st.session_state.coord_latitude:.6f}, {st.session_state.coord_longitude:.6f}",
                key="coord_img"
            )
        
        # Display predictions
        if st.session_state.coord_predictions:
            st.subheader("Land Type Prediction")
            results = st.session_state.coord_predictions
            
            for label, conf in results:
                st.write(f"**{label}**: {conf:.3f}")
                st.progress(min(1.0, conf))
            
            top_prediction = results[0][0]
            
            # Show location information
            st.subheader("📍 Location Information")
            location_info = st.session_state.coord_location_info
            if location_info and location_info.get('status') == 'success':
                st.write(f"**Location**: {location_info['summary']}")
                st.write(f"**Country**: {location_info['country']}")
                if location_info.get('region'):
                    st.write(f"**Region/State**: {location_info['region']}")
                if location_info.get('city'):
                    st.write(f"**City**: {location_info['city']}")
                if location_info.get('county'):
                    st.write(f"**County**: {location_info['county']}")
                st.write(f"**Coordinates**: {st.session_state.coord_latitude:.6f}, {st.session_state.coord_longitude:.6f}")
            else:
                st.warning("Could not retrieve detailed location information.")
                st.write(f"**Coordinates**: {st.session_state.coord_latitude:.6f}, {st.session_state.coord_longitude:.6f}")
            
            # Show recommendations
            st.subheader("💡 Land Suitability Recommendations")
            recommendations = rec_engine.get_recommendations(top_prediction, top_n=5)
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['use']}"):
                    st.write(rec['explanation'])
            
            # Update prediction data in session state
            st.session_state.coordinate_prediction_data = {
                'class': top_prediction,
                'confidence': results[0][1],
                'top_predictions': results,
                'recommendations': recommendations,
                'latitude': st.session_state.coord_latitude,
                'longitude': st.session_state.coord_longitude,
                'location_info': location_info
            }
            
            # Save to History button (always visible when data is available)
            st.markdown("---")
            if st.button("💾 Save to History", key="save_coordinate_prediction"):
                # Initialize history manager
                if 'history_manager' not in st.session_state:
                    st.session_state.history_manager = HistoryManager()
                
                history_mgr = st.session_state.history_manager
                coord_data = st.session_state.coordinate_prediction_data
                
                # Get location name
                location_name = None
                if coord_data.get('location_info') and coord_data['location_info'].get('status') == 'success':
                    location_name = coord_data['location_info'].get('summary', '')
                
                prediction_id = history_mgr.save_prediction(
                    input_type='coordinates',
                    prediction_class=coord_data['class'],
                    confidence=coord_data['confidence'],
                    top_predictions=coord_data['top_predictions'],
                    recommendations=coord_data['recommendations'],
                    latitude=coord_data['latitude'],
                    longitude=coord_data['longitude'],
                    location_name=location_name,
                    metadata={
                        'zoom_level': st.session_state.coord_zoom_level,
                        'use_satellite': st.session_state.coord_use_satellite,
                        'location_details': coord_data['location_info']
                    }
                )
                st.success(f"✅ Prediction saved to history! (ID: {prediction_id})")
                st.rerun()


def page_history():
    """Comprehensive prediction history management page."""
    st.header("📊 Prediction History Management")
    
    # Initialize history manager
    if 'history_manager' not in st.session_state:
        st.session_state.history_manager = HistoryManager()
    
    history_mgr = st.session_state.history_manager
    
    # Get all predictions
    all_predictions = history_mgr.get_all_predictions()
    
    # Check if plotly is available for dashboard
    try:
        import plotly
        plotly_available = True
    except ImportError:
        plotly_available = False
        if all_predictions:
            st.info("💡 **Tip**: Install plotly for interactive dashboard visualizations: `pip install plotly`")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Input type filter
    input_types = ['All'] + list(set([p['input_type'] for p in all_predictions]))
    selected_input_type = st.sidebar.selectbox("Input Type", input_types)
    
    # Prediction class filter
    all_classes = ['All'] + sorted(list(set([p['prediction_class'] for p in all_predictions])))
    selected_class = st.sidebar.selectbox("Land Type", all_classes)
    
    # Confidence filter
    min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0, 0.01)
    
    # Search
    search_term = st.sidebar.text_input("Search", "")
    
    # Apply filters
    filtered_predictions = all_predictions
    if selected_input_type != 'All':
        filtered_predictions = [p for p in filtered_predictions if p['input_type'] == selected_input_type]
    if selected_class != 'All':
        filtered_predictions = [p for p in filtered_predictions if p['prediction_class'] == selected_class]
    if min_confidence > 0:
        filtered_predictions = [p for p in filtered_predictions if p['confidence'] >= min_confidence]
    if search_term:
        filtered_predictions = history_mgr.search_predictions(search_term)
        # Apply other filters to search results
        if selected_input_type != 'All':
            filtered_predictions = [p for p in filtered_predictions if p['input_type'] == selected_input_type]
        if selected_class != 'All':
            filtered_predictions = [p for p in filtered_predictions if p['prediction_class'] == selected_class]
        if min_confidence > 0:
            filtered_predictions = [p for p in filtered_predictions if p['confidence'] >= min_confidence]
    
    # Statistics and Dashboard
    if all_predictions:
        stats = history_mgr.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", stats['total_count'])
        with col2:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
        with col3:
            st.metric("With Coordinates", stats['with_coordinates'])
        with col4:
            st.metric("Filtered Results", len(filtered_predictions))
        
        st.markdown("---")
        
        # Dashboard visualizations
        st.subheader("📈 Dashboard")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Land Types", "Input Types", "Timeline", "Map View"])
        
        with tab1:
            chart_type = st.radio("Chart Type", ["Pie", "Bar"], horizontal=True, key="land_type_chart")
            try:
                from dashboard.visualizations import create_land_type_chart
                fig = create_land_type_chart(all_predictions, chart_type.lower())
                if fig:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
            except ImportError:
                st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
        
        with tab2:
            try:
                from dashboard.visualizations import create_input_type_chart
                fig = create_input_type_chart(all_predictions)
                if fig:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
            except ImportError:
                st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
        
        with tab3:
            try:
                from dashboard.visualizations import create_timeline_chart
                fig = create_timeline_chart(all_predictions)
                if fig:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
            except ImportError:
                st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
        
        with tab4:
            try:
                from dashboard.visualizations import create_map_view
                fig = create_map_view(all_predictions)
                if fig:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No predictions with coordinates available for map view. Install plotly for visualization: `pip install plotly`")
            except ImportError:
                st.warning("⚠️ Plotly is not installed. Please install it with: `pip install plotly`")
        
        st.markdown("---")
    
    # Actions toolbar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📋 History Records ({len(filtered_predictions)} found)")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col3:
        if all_predictions:
            from history_manager.export_utils import export_to_excel, OPENPYXL_AVAILABLE
            if OPENPYXL_AVAILABLE:
                excel_data = export_to_excel(all_predictions)
                if excel_data:
                    st.download_button(
                        label="📥 Export All to Excel",
                        data=excel_data,
                        file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("⚠️ Excel export failed. Install openpyxl: `pip install openpyxl`")
            else:
                st.info("💡 Install openpyxl for Excel export: `pip install openpyxl`")
    
    # Display filtered predictions
    if filtered_predictions:
        for pred in filtered_predictions:
            with st.expander(
                f"ID {pred['id']}: {pred['prediction_class']} ({pred['confidence']:.3f}) - {pred['input_type']} - {pred['timestamp'][:19]}"
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Input Type**: {pred['input_type']}")
                    st.write(f"**Prediction**: {pred['prediction_class']}")
                    st.write(f"**Confidence**: {pred['confidence']:.3f}")
                    
                    if pred.get('location_name'):
                        st.write(f"**Location**: {pred['location_name']}")
                    if pred.get('latitude') and pred.get('longitude'):
                        st.write(f"**Coordinates**: {pred['latitude']:.6f}, {pred['longitude']:.6f}")
                    
                    st.write(f"**Timestamp**: {pred['timestamp']}")
                    
                    if pred.get('top_predictions'):
                        st.write("**Top Predictions**:")
                        for cls, conf in pred['top_predictions'][:3]:
                            st.write(f"  - {cls}: {conf:.3f}")
                    
                    if pred.get('recommendations'):
                        st.write("**Recommendations**:")
                        for rec in pred['recommendations'][:3]:
                            st.write(f"  - {rec.get('use', 'N/A')}")
                    
                    if pred.get('notes'):
                        st.write(f"**Notes**: {pred['notes']}")
                
                with col2:
                    # Edit button
                    if st.button("✏️ Edit", key=f"edit_{pred['id']}"):
                        st.session_state[f"editing_{pred['id']}"] = True
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{pred['id']}"):
                        if history_mgr.delete_prediction(pred['id']):
                            st.success(f"Deleted prediction {pred['id']}")
                            st.rerun()
                    
                    # Export single
                    from history_manager.export_utils import export_to_excel, OPENPYXL_AVAILABLE
                    if OPENPYXL_AVAILABLE:
                        excel_data = export_to_excel([pred])
                        if excel_data:
                            st.download_button(
                                label="📥 Export",
                                data=excel_data,
                                file_name=f"prediction_{pred['id']}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"export_{pred['id']}"
                            )
                        else:
                            st.caption("Export unavailable")
                    else:
                        st.caption("Install openpyxl for export")
                
                # Edit form
                if st.session_state.get(f"editing_{pred['id']}", False):
                    st.markdown("---")
                    st.subheader("Edit Prediction")
                    
                    new_class = st.text_input("Prediction Class", value=pred['prediction_class'], key=f"class_{pred['id']}")
                    new_confidence = st.number_input("Confidence", value=float(pred['confidence']), min_value=0.0, max_value=1.0, key=f"conf_{pred['id']}")
                    new_notes = st.text_area("Notes", value=pred.get('notes', ''), key=f"notes_{pred['id']}")
                    new_location = st.text_input("Location Name", value=pred.get('location_name', ''), key=f"loc_{pred['id']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save Changes", key=f"save_{pred['id']}"):
                            history_mgr.update_prediction(
                                pred['id'],
                                prediction_class=new_class,
                                confidence=new_confidence,
                                notes=new_notes,
                                location_name=new_location if new_location else None
                            )
                            st.success("Changes saved!")
                            st.session_state[f"editing_{pred['id']}"] = False
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_{pred['id']}"):
                            st.session_state[f"editing_{pred['id']}"] = False
                            st.rerun()
    else:
        st.info("No predictions found. Start making predictions to see them here!")
    
    # Clear all button (with confirmation)
    if all_predictions:
        st.markdown("---")
        if st.button("🗑️ Clear All History", type="primary"):
            if st.session_state.get('confirm_clear', False):
                deleted_count = history_mgr.delete_all_predictions()
                st.success(f"Deleted {deleted_count} predictions!")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm deletion of all history!")

