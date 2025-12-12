"""Image quality enhancement and preprocessing utilities."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Tuple
import cv2


class ImageEnhancer:
    """Enhance image quality before model prediction."""
    
    def __init__(
        self,
        enhance_sharpness: bool = True,
        enhance_brightness: bool = True,
        enhance_contrast: bool = True,
        enhance_saturation: bool = False,
        denoise: bool = False,
        upscale_small: bool = False,
        min_size: int = 224
    ):
        """
        Initialize image enhancer.
        
        Args:
            enhance_sharpness: Apply sharpness enhancement
            enhance_brightness: Apply brightness adjustment
            enhance_contrast: Apply contrast adjustment
            denoise: Apply noise reduction
            upscale_small: Upscale images smaller than min_size
            min_size: Minimum size for upscaling (default: 224 for EfficientNet)
        """
        self.enhance_sharpness = enhance_sharpness
        self.enhance_brightness = enhance_brightness
        self.enhance_contrast = enhance_contrast
        self.enhance_saturation = enhance_saturation
        self.denoise = denoise
        self.upscale_small = upscale_small
        self.min_size = min_size
    
    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Apply all enabled enhancements to an image.
        
        Args:
            image: PIL Image to enhance
        
        Returns:
            Enhanced PIL Image
        """
        enhanced = image.copy()
        
        # Upscale if needed
        if self.upscale_small:
            width, height = enhanced.size
            if width < self.min_size or height < self.min_size:
                scale_factor = max(self.min_size / width, self.min_size / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                enhanced = enhanced.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply PIL-based enhancements
        if self.enhance_sharpness:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)  # 20% sharper
        
        if self.enhance_brightness:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.05)  # 5% brighter
        
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)  # 10% more contrast
        
        if self.enhance_saturation:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.1)  # 10% more saturation
        
        # Apply OpenCV-based denoising if needed
        if self.denoise:
            enhanced = self._denoise_image(enhanced)
        
        return enhanced
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """
        Apply noise reduction using OpenCV.
        
        Args:
            image: PIL Image
        
        Returns:
            Denoised PIL Image
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Apply denoising
        if len(img_array.shape) == 3:  # Color image
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        else:  # Grayscale
            denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        
        # Convert back to PIL
        return Image.fromarray(denoised)
    
    def enhance_with_params(
        self,
        image: Image.Image,
        sharpness_factor: float = 1.2,
        brightness_factor: float = 1.05,
        contrast_factor: float = 1.1,
        saturation_factor: float = 1.0
    ) -> Image.Image:
        """
        Enhance image with custom parameters.
        
        Args:
            image: PIL Image to enhance
            sharpness_factor: Sharpness enhancement factor (1.0 = no change)
            brightness_factor: Brightness enhancement factor (1.0 = no change)
            contrast_factor: Contrast enhancement factor (1.0 = no change)
            saturation_factor: Saturation enhancement factor (1.0 = no change)
        
        Returns:
            Enhanced PIL Image
        """
        enhanced = image.copy()
        
        # Upscale if needed
        if self.upscale_small:
            width, height = enhanced.size
            if width < self.min_size or height < self.min_size:
                scale_factor = max(self.min_size / width, self.min_size / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                enhanced = enhanced.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply enhancements with custom factors
        if self.enhance_sharpness and sharpness_factor != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness_factor)
        
        if self.enhance_brightness and brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness_factor)
        
        if self.enhance_contrast and contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast_factor)
        
        if self.enhance_saturation and saturation_factor != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation_factor)
        
        # Apply denoising
        if self.denoise:
            enhanced = self._denoise_image(enhanced)
        
        return enhanced
    
    @staticmethod
    def create_thumbnail(image: Image.Image, max_size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """
        Create a thumbnail version of an image for display.
        
        Args:
            image: PIL Image
            max_size: Maximum (width, height) for thumbnail
        
        Returns:
            Thumbnail PIL Image
        """
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image.copy()

