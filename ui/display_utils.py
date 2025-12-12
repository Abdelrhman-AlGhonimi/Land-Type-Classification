"""UI display utilities for images and videos."""

import streamlit as st
from PIL import Image
from typing import List, Tuple, Optional
import numpy as np


def create_thumbnail(image: Image.Image, max_width: int = 300, max_height: int = 300) -> Image.Image:
    """
    Create a thumbnail version of an image for display.
    
    Args:
        image: PIL Image
        max_width: Maximum width for thumbnail
        max_height: Maximum height for thumbnail
    
    Returns:
        Thumbnail PIL Image maintaining aspect ratio
    """
    thumbnail = image.copy()
    thumbnail.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return thumbnail


def display_image_with_expand(
    image: Image.Image,
    caption: str = "",
    thumbnail_size: Tuple[int, int] = (300, 300),
    key: Optional[str] = None
):
    """
    Display an image with thumbnail and expandable full-size view.
    
    Args:
        image: PIL Image to display
        caption: Caption for the image
        thumbnail_size: (width, height) for thumbnail display
        key: Unique key for Streamlit widget
    """
    # Create thumbnail
    thumbnail = create_thumbnail(image, thumbnail_size[0], thumbnail_size[1])
    
    # Display thumbnail
    st.image(thumbnail, caption=caption, width='stretch')
    
    # Add expandable section for full-size view
    with st.expander("🔍 View Full Size Image", expanded=False):
        st.image(image, caption=f"{caption} (Full Size)", width='stretch')


def create_video_preview_grid(
    frames: List[Tuple[Image.Image, float]],
    max_frames: int = 6,
    thumbnail_size: Tuple[int, int] = (150, 150)
) -> List[Image.Image]:
    """
    Create a grid of video frame thumbnails for preview.
    
    Args:
        frames: List of (PIL Image, timestamp) tuples
        max_frames: Maximum number of frames to show
        thumbnail_size: (width, height) for each thumbnail
    
    Returns:
        List of thumbnail PIL Images
    """
    # Select frames evenly spaced
    if len(frames) <= max_frames:
        selected_frames = frames
    else:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]
    
    # Create thumbnails
    thumbnails = []
    for frame, timestamp in selected_frames:
        thumb = create_thumbnail(frame, thumbnail_size[0], thumbnail_size[1])
        thumbnails.append(thumb)
    
    return thumbnails


def display_video_preview(
    frames: List[Tuple[Image.Image, float]],
    video_name: str = "Video",
    max_frames: int = 6
):
    """
    Display a preview grid of video frames.
    
    Args:
        frames: List of (PIL Image, timestamp) tuples
        video_name: Name of the video
        max_frames: Maximum number of frames to show in preview
    """
    if not frames:
        st.warning("No frames available for preview.")
        return
    
    st.subheader(f"📹 {video_name} Preview")
    
    # Create preview grid
    thumbnails = create_video_preview_grid(frames, max_frames=max_frames)
    
    # Display in columns
    num_cols = min(3, len(thumbnails))
    cols = st.columns(num_cols)
    
    for idx, thumb in enumerate(thumbnails):
        col_idx = idx % num_cols
        with cols[col_idx]:
            frame_idx = idx
            if len(frames) > max_frames:
                # Calculate approximate timestamp
                total_frames = len(frames)
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
                frame_idx = indices[idx]
            
            timestamp = frames[frame_idx][1] if frame_idx < len(frames) else 0.0
            st.image(thumb, caption=f"Frame {idx+1} (t={timestamp:.1f}s)", use_container_width=True)
    
    # Add expandable section for all frames
    with st.expander(f"🔍 View All {len(frames)} Frames", expanded=False):
        # Display more frames in a scrollable container
        num_cols_all = 4
        cols_all = st.columns(num_cols_all)
        
        for idx, (frame, timestamp) in enumerate(frames):
            col_idx = idx % num_cols_all
            with cols_all[col_idx]:
                thumb = create_thumbnail(frame, 120, 120)
                st.image(thumb, caption=f"Frame {idx+1} (t={timestamp:.1f}s)", width='stretch')


def display_side_by_side(
    original: Image.Image,
    enhanced: Image.Image,
    original_label: str = "Original",
    enhanced_label: str = "Enhanced",
    thumbnail_size: Tuple[int, int] = (300, 300)
):
    """
    Display original and enhanced images side by side.
    
    Args:
        original: Original PIL Image
        enhanced: Enhanced PIL Image
        original_label: Label for original image
        enhanced_label: Label for enhanced image
        thumbnail_size: (width, height) for display
    """
    col1, col2 = st.columns(2)
    
    with col1:
        thumb_orig = create_thumbnail(original, thumbnail_size[0], thumbnail_size[1])
        st.image(thumb_orig, caption=original_label, width='stretch')
        with st.expander("🔍 View Full Size", expanded=False):
            st.image(original, caption=f"{original_label} (Full Size)", width='stretch')
    
    with col2:
        thumb_enh = create_thumbnail(enhanced, thumbnail_size[0], thumbnail_size[1])
        st.image(thumb_enh, caption=enhanced_label, width='stretch')
        with st.expander("🔍 View Full Size", expanded=False):
            st.image(enhanced, caption=f"{enhanced_label} (Full Size)", width='stretch')

