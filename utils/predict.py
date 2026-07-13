"""
Prediction module for Medical Imaging Assistant.
Handles model inference, result processing, and confidence scoring.
"""

import numpy as np
import streamlit as st
import time
from utils.model_loader import load_cnn_model, load_vgg16_model
from utils.preprocess import preprocess_image, validate_image

# Disease classes mapping
DISEASE_CLASSES = [
    'Normal',
    'Pneumonia',
    'Tumor',
    'Fracture',
    'Cardiomegaly',
    'Pleural Effusion'
]

# Class descriptions for explainability
CLASS_DESCRIPTIONS = {
    'Normal': 'No significant abnormalities detected. All anatomical structures appear within normal limits.',
    'Pneumonia': 'Inflammation of the lungs typically caused by infection. Characterized by opacities in lung fields.',
    'Tumor': 'Abnormal growth of tissue that may be benign or malignant. Requires further investigation.',
    'Fracture': 'Break or crack in bone structure. May require immediate medical attention.',
    'Cardiomegaly': 'Enlargement of the heart. Can indicate various underlying cardiac conditions.',
    'Pleural Effusion': 'Accumulation of fluid in the pleural space around the lungs.'
}


def predict(image, model_type='VGG16', confidence_threshold=0.5):
    """
    Run prediction on input image using the specified model.
    
    Args:
        image (numpy.ndarray): Input image
        model_type (str): 'CNN' or 'VGG16'
        confidence_threshold (float): Minimum confidence for positive detection
        
    Returns:
        dict: Prediction results with probabilities, class, confidence, and metadata
    """
    # Validate image
    is_valid, message = validate_image(image)
    if not is_valid:
        st.error(f"Invalid image: {message}")
        return None
    
    # Load appropriate model
    start_time = time.time()
    
    if model_type == 'CNN':
        model = load_cnn_model()
        model_info = "Custom CNN (74.30% accuracy)"
    else:
        model = load_vgg16_model()
        model_info = "VGG16 (98.18% accuracy)"
    
    if model is None:
        st.error(f"Failed to load {model_type} model")
        return None
    
    # Preprocess image
    preprocessed = preprocess_image(image)
    if preprocessed is None:
        return None
    
    # Run prediction with progress
    with st.spinner(f'🔬 Running {model_type} inference...'):
        try:
            predictions = model.predict(preprocessed, verbose=0)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    # Process results
    raw_probs = predictions[0]
    
    if len(raw_probs) == 1:
        # Binary classification (Normal vs Pneumonia)
        p = float(raw_probs[0])
        probabilities = np.array([1.0 - p, p])
        active_classes = DISEASE_CLASSES[:2] # ['Normal', 'Pneumonia']
    else:
        # Multi-class classification
        probabilities = raw_probs
        active_classes = DISEASE_CLASSES

    predicted_class_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    predicted_class = active_classes[predicted_class_index]
    
    # Determine if abnormal (anything other than 'Normal')
    is_abnormal = predicted_class != 'Normal'
    
    # Check if confidence meets threshold
    meets_threshold = confidence >= confidence_threshold
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Create detailed result dictionary
    result = {
        'predicted_class': predicted_class,
        'class_index': predicted_class_index,
        'confidence': confidence,
        'probabilities': {
            active_classes[i]: float(probabilities[i]) 
            for i in range(len(active_classes))
        },
        'is_abnormal': is_abnormal,
        'meets_threshold': meets_threshold,
        'model_type': model_type,
        'model_info': model_info,
        'inference_time': inference_time,
        'confidence_threshold': confidence_threshold,
        'class_description': CLASS_DESCRIPTIONS.get(predicted_class, 'No description available.'),
        'severity': _get_severity(confidence),
        'recommendation': _get_recommendation(predicted_class, confidence, is_abnormal)
    }
    
    return result


def _get_severity(confidence):
    """
    Determine severity level based on confidence score.
    
    Args:
        confidence (float): Prediction confidence (0-1)
        
    Returns:
        str: Severity level with emoji
    """
    if confidence > 0.9:
        return "🟢 High Confidence - Strong evidence"
    elif confidence > 0.7:
        return "🟡 Moderate Confidence - Reasonable certainty"
    elif confidence > 0.5:
        return "🟠 Low Confidence - Requires caution"
    else:
        return "🔴 Very Low Confidence - Needs review"


def _get_recommendation(predicted_class, confidence, is_abnormal):
    """
    Generate recommendation based on prediction results.
    
    Args:
        predicted_class (str): Predicted disease class
        confidence (float): Confidence score
        is_abnormal (bool): Whether abnormality detected
        
    Returns:
        str: Recommendation text
    """
    if not is_abnormal:
        return "✅ Normal result. No immediate action required. Continue routine screening as recommended."
    
    if confidence > 0.9:
        return f"⚠️ {predicted_class} detected with high confidence. Recommend immediate clinical correlation and specialist consultation."
    elif confidence > 0.7:
        return f"⚠️ {predicted_class} suspected. Recommend radiology review and confirmatory tests."
    else:
        return f"⚠️ Possible {predicted_class} detected with low confidence. Strongly recommend human radiologist review and additional imaging."


def get_top_predictions(result, n=3):
    """
    Get top N predictions with their probabilities.
    
    Args:
        result (dict): Prediction results
        n (int): Number of top predictions to return
        
    Returns:
        list: Top N predictions as (class, probability) tuples
    """
    probabilities = result.get('probabilities', {})
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:n]


def compare_models(image):
    """
    Run prediction with both CNN and VGG16 models for comparison.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        dict: Results from both models
    """
    cnn_result = predict(image, model_type='CNN')
    vgg16_result = predict(image, model_type='VGG16')
    
    return {
        'cnn': cnn_result,
        'vgg16': vgg16_result
    }


def get_prediction_summary(result):
    """
    Get a concise summary of the prediction.
    
    Args:
        result (dict): Prediction results
        
    Returns:
        dict: Summary with key information
    """
    if result is None:
        return None
    
    summary = {
        'diagnosis': result.get('predicted_class', 'Unknown'),
        'confidence': f"{result.get('confidence', 0):.1%}",
        'status': 'Abnormal' if result.get('is_abnormal', False) else 'Normal',
        'model': result.get('model_type', 'Unknown'),
        'inference_time': f"{result.get('inference_time', 0):.2f}s",
        'severity': result.get('severity', 'Unknown'),
        'recommendation': result.get('recommendation', 'No recommendation available')
    }
    return summary
