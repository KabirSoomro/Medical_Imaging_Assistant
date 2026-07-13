"""
Visualization module for Medical Imaging Assistant.
Creates interactive charts, graphs, and visual elements for the UI.
Uses Plotly for interactive charts and Matplotlib for static displays.
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import io


def create_confidence_chart(probabilities):
    """
    Create an interactive bar chart showing disease probabilities.
    
    Args:
        probabilities (dict): Dictionary of disease: probability
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    if not probabilities:
        return None
    
    diseases = list(probabilities.keys())
    scores = list(probabilities.values())
    
    # Color coding based on probability
    colors = []
    for score in scores:
        if score < 0.3:
            colors.append('#2ecc71')  # Green - low probability
        elif score < 0.6:
            colors.append('#f1c40f')  # Yellow - medium probability
        elif score < 0.8:
            colors.append('#e67e22')  # Orange - high probability
        else:
            colors.append('#e74c3c')  # Red - very high probability
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=scores,
            marker_color=colors,
            text=[f'{score:.1%}' for score in scores],
            textposition='outside',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<br><extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='<b>Disease Probability Distribution</b>',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title='Disease Type',
        yaxis_title='Probability',
        yaxis=dict(
            range=[0, 1],
            tickformat='.0%',
            gridcolor='lightgray',
            gridwidth=1
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=11)
        ),
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Add horizontal line at threshold
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Threshold (50%)", 
                  annotation_position="bottom right")
    
    return fig


def create_confidence_gauge(confidence):
    """
    Create a gauge chart for confidence score.
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        plotly.graph_objects.Figure: Gauge chart
    """
    confidence_percent = confidence * 100
    
    # Determine color based on confidence
    if confidence > 0.8:
        gauge_color = "#2ecc71"  # Green
        status = "High Confidence"
    elif confidence > 0.5:
        gauge_color = "#f1c40f"  # Yellow
        status = "Medium Confidence"
    else:
        gauge_color = "#e74c3c"  # Red
        status = "Low Confidence"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title=dict(
            text=f"<b>Confidence Score</b><br><span style='font-size:12px;color:{gauge_color};'>{status}</span>",
            font=dict(size=16)
        ),
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge=dict(
            axis={
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "darkgray",
                'tickfont': {'size': 12}
            },
            bar={'color': gauge_color},
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                {'range': [0, 30], 'color': '#fdebd0'},
                {'range': [30, 70], 'color': '#fef9e7'},
                {'range': [70, 100], 'color': '#d5f5e3'}
            ],
            threshold={
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        )
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_comparison_table(cnn_result, vgg16_result):
    """
    Create a comparison table for CNN vs VGG16.
    
    Args:
        cnn_result (dict): CNN prediction results
        vgg16_result (dict): VGG16 prediction results
        
    Returns:
        pandas.DataFrame: Comparison dataframe
    """
    data = {
        'Metric': ['Accuracy', 'Predicted Class', 'Confidence', 'Inference Time', 'Status'],
        'CNN': [
            '74.30%',
            cnn_result.get('predicted_class', 'N/A'),
            f"{cnn_result.get('confidence', 0):.1%}",
            f"{cnn_result.get('inference_time', 0):.2f}s",
            '🟡 Abnormal' if cnn_result.get('is_abnormal', False) else '🟢 Normal'
        ],
        'VGG16': [
            '98.18%',
            vgg16_result.get('predicted_class', 'N/A'),
            f"{vgg16_result.get('confidence', 0):.1%}",
            f"{vgg16_result.get('inference_time', 0):.2f}s",
            '🔴 Abnormal' if vgg16_result.get('is_abnormal', False) else '🟢 Normal'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add styling
    def highlight_best(val):
        if val == '98.18%':
            return 'background-color: #2ecc71; color: white; font-weight: bold'
        elif val == '74.30%':
            return 'background-color: #f39c12; color: white;'
        return ''
    
    return df


def display_image_with_overlay(image, heatmap=None, boxes=None):
    """
    Display image with optional heatmap overlay and bounding boxes.
    
    Args:
        image (numpy.ndarray): Original image
        heatmap (numpy.ndarray, optional): Heatmap overlay
        boxes (list, optional): List of bounding boxes [x, y, w, h]
    """
    fig, axes = plt.subplots(1, 2 if heatmap is not None else 1, figsize=(14, 7))
    
    if heatmap is not None:
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Image with heatmap overlay
        axes[1].imshow(image)
        
        # Overlay heatmap
        axes[1].imshow(heatmap, alpha=0.4, cmap='jet')
        
        # Add bounding boxes if provided
        if boxes:
            for box in boxes:
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2], box[3],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[1].add_patch(rect)
        
        axes[1].set_title('Heatmap Overlay (Grad-CAM)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    else:
        # Single image display
        axes.imshow(image)
        axes.set_title('Uploaded Image', fontsize=14, fontweight='bold')
        axes.axis('off')
        st.pyplot(fig)
        plt.close()


def display_processing_steps(steps):
    """
    Display processing steps with status indicators.
    
    Args:
        steps (list): List of step dictionaries with 'name', 'status', 'message'
    """
    for step in steps:
        status_icon = {
            'pending': '⏳',
            'processing': '🔄',
            'complete': '✅',
            'error': '❌'
        }.get(step.get('status', 'pending'), '⏳')
        
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"<h3>{status_icon}</h3>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{step.get('name', 'Step')}**")
            if step.get('message'):
                st.caption(step.get('message'))


def create_model_comparison_chart(cnn_accuracy, vgg16_accuracy):
    """
    Create a comparison chart for model accuracies.
    
    Args:
        cnn_accuracy (float): CNN accuracy (0-1)
        vgg16_accuracy (float): VGG16 accuracy (0-1)
        
    Returns:
        plotly.graph_objects.Figure: Comparison chart
    """
    fig = go.Figure(data=[
        go.Bar(
            name='Accuracy',
            x=['CNN', 'VGG16'],
            y=[cnn_accuracy * 100, vgg16_accuracy * 100],
            marker_color=['#f39c12', '#2ecc71'],
            text=[f'{cnn_accuracy:.1%}', f'{vgg16_accuracy:.1%}'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='<b>Model Accuracy Comparison</b>',
        xaxis_title='Model',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig


def create_metrics_dashboard(metrics):
    """
    Create a metrics dashboard with key performance indicators.
    
    Args:
        metrics (dict): Dictionary of metrics
    """
    cols = st.columns(len(metrics))
    
    for col, (key, value) in zip(cols, metrics.items()):
        with col:
            st.metric(
                label=key.replace('_', ' ').title(),
                value=value
            )


def display_prediction_summary(result):
    """
    Display a visual summary of prediction results.
    
    Args:
        result (dict): Prediction results
    """
    if not result:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status indicator
        if result.get('is_abnormal', False):
            st.error(f"🔴 **{result.get('predicted_class', 'Unknown')}**")
        else:
            st.success(f"✅ **{result.get('predicted_class', 'Normal')}**")
        
        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
        st.metric("Inference Time", f"{result.get('inference_time', 0):.2f}s")
    
    with col2:
        st.info(f"**Model:** {result.get('model_type', 'Unknown')}")
        st.info(f"**Severity:** {result.get('severity', 'Unknown')}")
        st.info(f"**Threshold Met:** {'✅ Yes' if result.get('meets_threshold', False) else '❌ No'}")
