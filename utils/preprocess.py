"""
Image preprocessing module for Medical Imaging Assistant.
Handles image loading, resizing, normalization, and augmentation.
Supports multiple image formats including DICOM.
"""

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import io

# Optional DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


def load_image(uploaded_file):
    """
    Load image from uploaded file.
    Supports JPG, PNG, JPEG, and DICOM formats.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        numpy.ndarray: Loaded image array, or None if loading fails
    """
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Handle DICOM files
        if file_extension == 'dcm':
            if DICOM_AVAILABLE:
                # Read DICOM file
                dicom_data = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
                image = dicom_data.pixel_array
                
                # Normalize DICOM image
                if image.dtype != np.uint8:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                    image = image.astype(np.uint8)
                
                return image
            else:
                st.error("DICOM support requires pydicom. Install with: pip install pydicom")
                return None
        
        # Handle regular images
        else:
            # Reset file pointer
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            image = np.array(image)
            return image
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input.
    
    Steps:
    1. Convert to RGB if grayscale
    2. Resize to target size
    3. Normalize pixel values (0-1)
    4. Add batch dimension
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model
    """
    try:
        # Handle different image shapes
        if len(image.shape) == 2:
            # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Ensure we have 3 channels
        if len(image.shape) == 3 and image.shape[2] != 3:
            st.warning(f"Unexpected number of channels: {image.shape[2]}. Converting to RGB.")
            if image.shape[2] > 3:
                image = image[:, :, :3]
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add batch dimension (for model input)
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def get_image_info(image):
    """
    Get information about the image for display.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        dict: Image information including dimensions, channels, dtype
    """
    info = {
        'shape': image.shape,
        'dimensions': f"{image.shape[0]} x {image.shape[1]}",
        'channels': image.shape[2] if len(image.shape) == 3 else 1,
        'dtype': str(image.dtype),
        'min_value': float(image.min()),
        'max_value': float(image.max()),
        'mean_value': float(image.mean())
    }
    return info


def apply_augmentation(image, augmentation_type='none'):
    """
    Apply data augmentation to image (for demo purposes).
    
    Args:
        image (numpy.ndarray): Input image
        augmentation_type (str): Type of augmentation
        
    Returns:
        numpy.ndarray: Augmented image
    """
    if augmentation_type == 'none':
        return image
    
    try:
        if augmentation_type == 'rotate':
            angle = np.random.randint(-30, 30)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
            
        elif augmentation_type == 'flip_horizontal':
            return cv2.flip(image, 1)
            
        elif augmentation_type == 'brightness':
            brightness = np.random.uniform(0.7, 1.3)
            return np.clip(image * brightness, 0, 255).astype(np.uint8)
            
        elif augmentation_type == 'contrast':
            contrast = np.random.uniform(0.7, 1.3)
            mean = np.mean(image)
            return np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
            
        else:
            return image
            
    except Exception as e:
        st.warning(f"Augmentation failed: {str(e)}")
        return image


def validate_image(image):
    """
    Validate if the image is suitable for analysis.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (is_valid, message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image dimensions: {image.shape}"
    
    if image.size == 0:
        return False, "Empty image"
    
    # Check for valid pixel values
    if image.dtype == np.uint8:
        if image.max() > 255 or image.min() < 0:
            return False, "Pixel values out of range for uint8"
    
    return True, "Image is valid"
