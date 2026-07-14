"""
Medical Imaging Assistant - Streamlit Application
An AI-powered diagnostic support system for radiological image analysis.
Course: Introduction to Artificial Intelligence
Student: Ghulam Kabir (2K23/CSM/40)
"""

import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.model_loader import load_cnn_model, load_vgg16_model, get_model_info, check_models_exist
from utils.preprocess import load_image, preprocess_image, get_image_info
from utils.predict import predict, DISEASE_CLASSES, get_top_predictions, get_prediction_summary, compare_models
from utils.visualize import (
    create_confidence_chart,
    create_confidence_gauge,
    create_comparison_table,
    display_image_with_overlay,
    display_processing_steps,
    create_model_comparison_chart,
    display_prediction_summary
)
from utils.explainability import (
    generate_explanation,
    get_key_factors,
    get_simple_explanation,
    generate_report_text
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Medical Imaging Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS        
# ============================================================================
st.markdown("""
    <style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Result cards */
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-card-abnormal {
        background: #fef9f9;
        border-left: 4px solid #e74c3c;
    }
    .result-card-normal {
        background: #f0faf0;
        border-left: 4px solid #2ecc71;
    }
    
    /* Status boxes */
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    /* Metrics */
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'results' not in st.session_state:
    st.session_state.results = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'model_used' not in st.session_state:
    st.session_state.model_used = None

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class="main-header">
        <h1>🩺 Medical Imaging Assistant</h1>
        <p>AI-Powered Diagnostic Support for Radiological Analysis</p>
        <p style="font-size: 0.85rem; opacity: 0.8;">X-ray • MRI • CT Scan Analysis </p>
        <p style="font-size: 0.9rem; margin-top: 10px; font-weight: bold; background-color: rgba(255, 255, 255, 0.2); display: inline-block; padding: 5px 15px; border-radius: 20px;">
            🔍 Detects 2 Classes: Normal vs Pneumonia
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("<h2 style='margin-top: 0px; margin-bottom: 5px;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    # Model Selection
    with st.container():

        model_option = st.selectbox(
            "🤖 Model Selection",
            ["VGG16 (98.18% Accuracy)", "CNN (74.30% Accuracy)"],
            help="VGG16 uses transfer learning, CNN is custom-built from scratch"
        )
        model_type = "VGG16" if "VGG16" in model_option else "CNN"

    
    # Confidence Threshold
    with st.container():

        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Lower threshold = more sensitive (more alerts), Higher threshold = more specific (fewer false alerts)"
        )

    
    # Imaging Modality
    with st.container():

        modality = st.selectbox(
            "📷 Imaging Modality",
            ["X-ray", "MRI", "CT Scan"],
            help="Select the type of medical image being analyzed"
        )

    
    # Patient Information (Optional)
    with st.container():

        st.markdown("<div style='font-weight: 600; font-size: 14px; margin-bottom: 5px;'>👤 Patient Information</div>", unsafe_allow_html=True)
        patient_name = st.text_input("Patient Name", placeholder="e.g., kabir", label_visibility="collapsed")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
        patient_gender = st.selectbox("Gender", ["", "Male", "Female"])
        patient_symptoms = st.text_area("Symptoms", placeholder="e.g., cough, fever, chest pain", height=60)

    
    # Model Status
    with st.container():

        st.markdown("<div style='font-weight: 600; font-size: 14px; margin-bottom: 5px;'>📊 Model Status</div>", unsafe_allow_html=True)
        models_exist = check_models_exist()
        if models_exist['vgg16']:
            st.success("✅ VGG16 model loaded")
        else:
            st.warning("⚠️ VGG16 model not found")
        if models_exist['cnn']:
            st.success("✅ CNN model loaded")
        else:
            st.warning("⚠️ CNN model not found")


# ============================================================================
# MAIN AREA - IMAGE UPLOAD
# ============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Medical Image")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or browse to upload",
        type=["jpg", "jpeg", "png", "dcm"],
        help="Supported formats: JPG, PNG, JPEG, DICOM"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = load_image(uploaded_file)
        if image is not None:
            st.session_state.uploaded_image = image
            
            # Show image info
            info = get_image_info(image)
            st.caption(f"📐 {info['dimensions']} • {info['channels']} channels • {info['dtype']}")
            
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("⚡ Quick Actions")
    
    if uploaded_file is not None and st.session_state.uploaded_image is not None:
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing image..."):
                try:
                    # Show processing steps
                    steps = [
                        {"name": "Loading image", "status": "complete", "message": "Image loaded successfully"},
                        {"name": "Preprocessing", "status": "processing", "message": "Resizing and normalizing..."}
                    ]
                    
                    # Run prediction
                    result = predict(
                        st.session_state.uploaded_image,
                        model_type=model_type,
                        confidence_threshold=confidence_threshold
                    )
                    
                    if result:
                        st.session_state.results = result
                        st.session_state.model_used = model_type
                        st.success("✅ Analysis complete!")
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        # Reset button
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.results = None
            st.session_state.uploaded_image = None
            st.rerun()
    else:
        st.info("📤 Please upload an image to begin analysis")

# ============================================================================
# RESULTS SECTION
# ============================================================================
if st.session_state.results:
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    result = st.session_state.results
    model_used = st.session_state.model_used
    
    # ========================================================================
    # TABS
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Report",
        "📈 Visualizations",
        "🔍 Explainability",
        "📊 Comparison",
        "📜 History"
    ])
    
    # ========================================================================
    # TAB 1: REPORT
    # ========================================================================
    with tab1:
        st.markdown("### 📋 Diagnostic Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prediction", result.get('predicted_class', 'Unknown'))
        with col2:
            st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
        with col3:
            status = "⚠️ Abnormal" if result.get('is_abnormal', False) else "✅ Normal"
            st.metric("Status", status)
        with col4:
            st.metric("Inference Time", f"{result.get('inference_time', 0):.2f}s")
        
        # Detailed report
        st.markdown("---")
        
        # Generate full explanation
        explanation = generate_explanation(result, model_used)
        st.markdown(explanation)
        
        # Download report button
        patient_info = {
            'name': patient_name if patient_name else 'N/A',
            'age': patient_age if patient_age else 'N/A',
            'gender': patient_gender if patient_gender else 'N/A',
            'modality': modality
        }
        report_text = generate_report_text(result, patient_info)
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report_text,
            file_name=f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # ========================================================================
    # TAB 2: VISUALIZATIONS
    # ========================================================================
    with tab2:
        st.markdown("### 📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence gauge
            gauge = create_confidence_gauge(result.get('confidence', 0))
            st.plotly_chart(gauge, use_container_width=True)
        
        with col2:
            # Probability chart
            prob_chart = create_confidence_chart(result.get('probabilities', {}))
            if prob_chart:
                st.plotly_chart(prob_chart, use_container_width=True)
        
        # Image with overlay
        if st.session_state.uploaded_image is not None:
            st.markdown("### 🖼️ Image Analysis")
            st.caption("Heatmap shows regions the model focused on for decision")
            
            # Display original image
            display_image_with_overlay(st.session_state.uploaded_image)
    
    # ========================================================================
    # TAB 3: EXPLAINABILITY
    # ========================================================================
    with tab3:
        st.markdown("### 🔍 Explainability")
        st.caption("Understanding why the AI made this decision")
        
        # Simple explanation
        st.markdown("#### 💡 Quick Summary")
        simple_explanation = get_simple_explanation(result)
        if result.get('is_abnormal', False):
            st.warning(simple_explanation)
        else:
            st.success(simple_explanation)
        
        # Key factors
        st.markdown("#### 🔑 Key Factors")
        factors = get_key_factors(result)
        for factor in factors:
            st.markdown(f"- {factor}")
        
        # Model information
        st.markdown("#### 🤖 Model Information")
        model_info = get_model_info(model_used)
        if model_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", model_info.get('accuracy', 'N/A'))
                st.metric("Parameters", model_info.get('parameters', 'N/A'))
            with col2:
                st.metric("Sensitivity", model_info.get('sensitivity', 'N/A'))
                st.metric("Training Time", model_info.get('training_time', 'N/A'))
            with col3:
                st.metric("Specificity", model_info.get('specificity', 'N/A'))
                st.metric("Dataset Size", model_info.get('dataset_size', 'N/A'))
            
            st.write("**Architecture:**", model_info.get('architecture', 'N/A'))
            st.write("**Strengths:**", model_info.get('strengths', 'N/A'))
            st.write("**Weaknesses:**", model_info.get('weaknesses', 'N/A'))
    
    # ========================================================================
    # TAB 4: COMPARISON
    # ========================================================================
    with tab4:
        st.markdown("### 📊 Model Comparison: CNN vs VGG16")
        st.caption("Comparing both models on the same image")
        
        with st.spinner("Running comparison..."):
            # Run both models
            cnn_result = predict(st.session_state.uploaded_image, model_type='CNN', confidence_threshold=confidence_threshold)
            vgg16_result = predict(st.session_state.uploaded_image, model_type='VGG16', confidence_threshold=confidence_threshold)
            
            if cnn_result and vgg16_result:
                # Comparison table
                comparison_df = create_comparison_table(cnn_result, vgg16_result)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Visual comparison
                st.markdown("#### 📈 Confidence Comparison")
                comp_data = {
                    'Model': ['CNN', 'VGG16'],
                    'Confidence': [
                        cnn_result.get('confidence', 0),
                        vgg16_result.get('confidence', 0)
                    ],
                    'Accuracy': [0.7430, 0.9818]
                }
                comp_df = pd.DataFrame(comp_data)
                st.bar_chart(comp_df.set_index('Model'), height=300)
                
                # Insights
                st.markdown("#### 💡 Key Insights")
                if vgg16_result.get('confidence', 0) > cnn_result.get('confidence', 0):
                    st.success("✅ **VGG16 shows higher confidence** for this image, which is expected due to its 98.18% accuracy.")
                else:
                    st.info("🔍 **CNN shows higher confidence** for this image - interesting case for review.")
                
                st.info("""
                **Model Comparison:**
                - **VGG16:** 98.18% accuracy, better generalization, robust performance
                - **CNN:** 74.30% accuracy, faster inference, lightweight
                - **Recommendation:** Use VGG16 for production, CNN for quick screening
                """)
            else:
                st.error("Failed to run comparison. Please try again.")
    
    # ========================================================================
    # TAB 5: HISTORY
    # ========================================================================
    with tab5:
        st.markdown("### 📜 Analysis History")
        
        # Add current result to history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_used,
            'prediction': result.get('predicted_class', 'Unknown'),
            'confidence': f"{result.get('confidence', 0):.1%}",
            'status': '⚠️ Abnormal' if result.get('is_abnormal', False) else '✅ Normal'
        }
        st.session_state.history.append(history_entry)
        
        if st.session_state.history:
            # Display history table
            history_df = pd.DataFrame(st.session_state.history)
            
            # Reverse order (newest first)
            history_df = history_df.iloc[::-1]
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            st.markdown("#### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Scans", len(st.session_state.history))
            with col2:
                abnormal = sum(1 for h in st.session_state.history if 'Abnormal' in h['status'])
                st.metric("Abnormal Findings", abnormal)
            with col3:
                normal = len(st.session_state.history) - abnormal
                st.metric("Normal Findings", normal)
            with col4:
                # Average confidence
                conf_values = [float(h['confidence'].rstrip('%')) for h in st.session_state.history]
                avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No analysis history yet")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>🩺 Medical Imaging Assistant v1.0 | AI-Powered Diagnostic Support</p>
        <p style="font-size: 0.7rem; opacity: 0.7;">
            Ghulam Kabir (2K23/CSM/40)
        </p>
        <p style="font-size: 0.7rem; color: #e74c3c;">
            ⚠️ This is a supportive tool. Final diagnosis requires clinical correlation by a qualified radiologist.
        </p>
    </div>
""", unsafe_allow_html=True)
