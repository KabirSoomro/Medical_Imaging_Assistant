"""
Model loader module for Medical Imaging Assistant.
Handles loading of trained CNN and VGG16 models with caching for performance.
"""

import tensorflow as tf
import os
import streamlit as st
from pathlib import Path

# Model paths (relative to project root)
MODEL_DIR = Path(__file__).parent.parent / "models"
CNN_MODEL_PATH = MODEL_DIR / "cnn_model.keras"
VGG16_MODEL_PATH = MODEL_DIR / "vgg16_finetuned.keras"


@st.cache_resource
def load_cnn_model(model_path=None):
    """
    Load the trained CNN model with caching.
    
    Args:
        model_path (str, optional): Path to the CNN model file.
            Defaults to models/cnn_model.h5
        
    Returns:
        tensorflow.keras.Model: Loaded CNN model, or None if loading fails
    """
    if model_path is None:
        model_path = str(CNN_MODEL_PATH)
    
    try:
        if not os.path.exists(model_path):
            st.warning(f"CNN model not found at {model_path}. Please place your trained model there.")
            return None
        
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {str(e)}")
        return None


@st.cache_resource
def load_vgg16_model(model_path=None):
    """
    Load the trained VGG16 model with caching.
    
    Args:
        model_path (str, optional): Path to the VGG16 model file.
            Defaults to models/vgg16_model.h5
        
    Returns:
        tensorflow.keras.Model: Loaded VGG16 model, or None if loading fails
    """
    if model_path is None:
        model_path = str(VGG16_MODEL_PATH)
    
    try:
        if not os.path.exists(model_path):
            st.warning(f"VGG16 model not found at {model_path}. Please place your trained model there.")
            return None
        
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading VGG16 model: {str(e)}")
        return None


def get_model_info(model_name):
    """
    Get detailed information about a specific model.
    
    Args:
        model_name (str): Name of the model ('CNN' or 'VGG16')
        
    Returns:
        dict: Model information including accuracy, architecture details
    """
    models_info = {
        "CNN": {
            "accuracy": "74.30%",
            "sensitivity": "72.15%",
            "specificity": "76.45%",
            "description": "Custom CNN built from scratch trained on medical imaging dataset",
            "architecture": "4 Convolutional layers + MaxPooling + 2 Dense layers",
            "parameters": "~2.5 million",
            "training_time": "~2 hours",
            "dataset_size": "~10,000 images",
            "strengths": "Lightweight, fast inference, good baseline",
            "weaknesses": "Lower accuracy, limited generalization"
        },
        "VGG16": {
            "accuracy": "98.18%",
            "sensitivity": "97.89%",
            "specificity": "98.47%",
            "description": "Transfer learning with VGG16 pre-trained on ImageNet",
            "architecture": "VGG16 base (16 layers) + Custom classification head",
            "parameters": "~14.7 million",
            "training_time": "~1 hour",
            "dataset_size": "~10,000 images + ImageNet pretraining",
            "strengths": "High accuracy, excellent generalization, robust",
            "weaknesses": "Larger model size, slightly slower inference"
        }
    }
    return models_info.get(model_name, {})


def check_models_exist():
    """
    Check if both model files exist.
    
    Returns:
        dict: Dictionary with 'cnn' and 'vgg16' boolean values
    """
    return {
        'cnn': CNN_MODEL_PATH.exists(),
        'vgg16': VGG16_MODEL_PATH.exists()
    }
