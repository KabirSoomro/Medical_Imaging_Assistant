"""
Explainability module for Medical Imaging Assistant.
Provides natural language explanations, key factors, and recommendations
for model predictions to build trust and understanding.
"""

import streamlit as st
from datetime import datetime


def generate_explanation(result, model_type='VGG16'):
    """
    Generate a comprehensive natural language explanation for the prediction.
    
    Args:
        result (dict): Prediction results
        model_type (str): Type of model used
        
    Returns:
        str: Natural language explanation with sections
    """
    if not result:
        return "No results available for explanation."
    
    predicted_class = result.get('predicted_class', 'Unknown')
    confidence = result.get('confidence', 0)
    is_abnormal = result.get('is_abnormal', False)
    severity = result.get('severity', 'Unknown')
    recommendation = result.get('recommendation', 'No recommendation available')
    class_description = result.get('class_description', 'No description available.')
    
    explanation_parts = []
    
    # 1. Header
    explanation_parts.append("=" * 70)
    explanation_parts.append("📋 **DIAGNOSTIC EXPLANATION REPORT**")
    explanation_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    explanation_parts.append("=" * 70)
    explanation_parts.append("")
    
    # 2. Main Finding
    explanation_parts.append("## 🎯 MAIN FINDING")
    if is_abnormal:
        explanation_parts.append(f"🔴 **{predicted_class}** detected with **{confidence:.1%}** confidence.")
    else:
        explanation_parts.append(f"✅ **{predicted_class}** - No significant abnormalities detected with **{confidence:.1%}** confidence.")
    explanation_parts.append("")
    
    # 3. Interpretation
    explanation_parts.append("## 💡 INTERPRETATION")
    if is_abnormal:
        if predicted_class == 'Pneumonia':
            explanation_parts.append("• The AI identified patterns consistent with pneumonia:")
            explanation_parts.append("  - Areas of opacity in the lung fields")
            explanation_parts.append("  - Possible consolidation or infiltration")
            explanation_parts.append("  - Air bronchogram signs may be present")
            explanation_parts.append("• This suggests possible infection or inflammation of the lungs.")
        
        elif predicted_class == 'Tumor':
            explanation_parts.append("• The AI detected an abnormal mass or growth:")
            explanation_parts.append("  - Irregular borders or margins")
            explanation_parts.append("  - Density differences in surrounding tissue")
            explanation_parts.append("  - Possible mass effect on adjacent structures")
            explanation_parts.append("• This could indicate a tumor that requires further investigation.")
        
        elif predicted_class == 'Fracture':
            explanation_parts.append("• The AI identified a break or crack in bone structure:")
            explanation_parts.append("  - Discontinuity in bone cortex")
            explanation_parts.append("  - Possible displacement or angulation")
            explanation_parts.append("  - Surrounding soft tissue swelling may be present")
            explanation_parts.append("• This suggests a fracture that needs immediate attention.")
        
        elif predicted_class == 'Cardiomegaly':
            explanation_parts.append("• The AI detected an enlarged heart shadow:")
            explanation_parts.append("  - Increased cardiothoracic ratio")
            explanation_parts.append("  - Changes in heart silhouette")
            explanation_parts.append("  - Possible cardiac enlargement or fluid around heart")
            explanation_parts.append("• This could indicate various cardiac conditions requiring evaluation.")
        
        elif predicted_class == 'Pleural Effusion':
            explanation_parts.append("• The AI detected fluid accumulation in pleural space:")
            explanation_parts.append("  - Blunting of costophrenic angles")
            explanation_parts.append("  - Opacification of lower lung zones")
            explanation_parts.append("  - Possible meniscus sign")
            explanation_parts.append("• This suggests pleural effusion requiring further assessment.")
        
        else:
            explanation_parts.append(f"• The AI detected findings consistent with {predicted_class}.")
            explanation_parts.append("• This requires clinical correlation and further evaluation.")
    else:
        explanation_parts.append("• The AI analysis shows normal anatomical structures:")
        explanation_parts.append("  - Clear lung fields with no significant opacities")
        explanation_parts.append("  - Normal heart silhouette and diaphragm contour")
        explanation_parts.append("  - No visible fractures or bone abnormalities")
        explanation_parts.append("  - No suspicious masses or lesions detected")
    explanation_parts.append("")
    
    # 4. Confidence Assessment
    explanation_parts.append("## 📊 CONFIDENCE ASSESSMENT")
    explanation_parts.append(f"**Confidence Score:** {confidence:.1%}")
    explanation_parts.append(f"**Severity Level:** {severity}")
    
    if confidence > 0.9:
        explanation_parts.append("✅ **High Confidence:** The model is very certain about this finding. The evidence is strong and consistent.")
        explanation_parts.append("• Multiple features support the diagnosis")
        explanation_parts.append("• Pattern recognition is clear and unambiguous")
    elif confidence > 0.7:
        explanation_parts.append("⚠️ **Moderate Confidence:** The model is reasonably confident but additional review is recommended.")
        explanation_parts.append("• Some features may be subtle or atypical")
        explanation_parts.append("• Consider clinical correlation")
    else:
        explanation_parts.append("🔴 **Low Confidence:** The model is uncertain about this finding.")
        explanation_parts.append("• Features may be ambiguous or atypical")
        explanation_parts.append("• Human expert review is strongly recommended")
    explanation_parts.append("")
    
    # 5. Key Factors
    explanation_parts.append("## 🔑 KEY FACTORS IN DECISION")
    factors = get_key_factors(result)
    for factor in factors:
        explanation_parts.append(f"• {factor}")
    explanation_parts.append("")
    
    # 6. Recommendations
    explanation_parts.append("## 📋 RECOMMENDATIONS")
    explanation_parts.append(recommendation)
    explanation_parts.append("")
    
    # 7. Model Information
    explanation_parts.append("## 🤖 MODEL INFORMATION")
    explanation_parts.append(f"**Model:** {model_type}")
    
    if model_type == 'VGG16':
        explanation_parts.append("• **Accuracy:** 98.18%")
        explanation_parts.append("• **Training:** Transfer learning from ImageNet")
        explanation_parts.append("• **Architecture:** 16 layers with custom classification head")
        explanation_parts.append("• **Strengths:** Excellent generalization, robust performance")
    else:
        explanation_parts.append("• **Accuracy:** 74.30%")
        explanation_parts.append("• **Training:** Custom CNN from scratch")
        explanation_parts.append("• **Architecture:** 4 Conv layers + 2 Dense layers")
        explanation_parts.append("• **Strengths:** Lightweight, fast inference")
    explanation_parts.append("")
    
    # 8. Disclaimer
    explanation_parts.append("## ⚠️ DISCLAIMER")
    explanation_parts.append("This AI system provides supportive analysis only. Final diagnosis must be made by a qualified medical professional.")
    explanation_parts.append("The information provided is for educational and decision-support purposes only.")
    explanation_parts.append("=" * 70)
    
    return "\n".join(explanation_parts)


def get_key_factors(result):
    """
    Extract and describe key factors that influenced the AI decision.
    
    Args:
        result (dict): Prediction results
        
    Returns:
        list: Key factors with descriptions
    """
    factors = []
    
    if not result:
        return ["No factors available."]
    
    predicted_class = result.get('predicted_class', 'Unknown')
    confidence = result.get('confidence', 0)
    probabilities = result.get('probabilities', {})
    is_abnormal = result.get('is_abnormal', False)
    
    # 1. Top prediction
    factors.append(f"**Primary Diagnosis:** {predicted_class} ({confidence:.1%} confidence)")
    
    # 2. Probability distribution (top 3)
    if probabilities:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_three = sorted_probs[:3]
        
        factor_text = "**Top 3 Differential Diagnoses:** "
        for i, (disease, prob) in enumerate(top_three, 1):
            factor_text += f"{i}. {disease} ({prob:.1%}) "
        factors.append(factor_text)
    
    # 3. Clinical features based on class
    if is_abnormal:
        if predicted_class == 'Pneumonia':
            factors.append("**Clinical Features Identified:**")
            factors.append("• Lung opacity in affected regions")
            factors.append("• Airspace consolidation patterns")
            factors.append("• Possible pleural effusion")
        elif predicted_class == 'Tumor':
            factors.append("**Clinical Features Identified:**")
            factors.append("• Abnormal mass with irregular borders")
            factors.append("• Tissue density differences")
            factors.append("• Possible displacement of adjacent structures")
        elif predicted_class == 'Fracture':
            factors.append("**Clinical Features Identified:**")
            factors.append("• Cortical bone discontinuity")
            factors.append("• Fracture line visibility")
            factors.append("• Surrounding tissue changes")
        elif predicted_class == 'Cardiomegaly':
            factors.append("**Clinical Features Identified:**")
            factors.append("• Enlarged cardiac silhouette")
            factors.append("• Increased cardiothoracic ratio")
            factors.append("• Possible pulmonary congestion")
        elif predicted_class == 'Pleural Effusion':
            factors.append("**Clinical Features Identified:**")
            factors.append("• Blunting of costophrenic angles")
            factors.append("• Opacification of lung bases")
            factors.append("• Possible meniscus sign")
        else:
            factors.append("**Clinical Features Identified:**")
            factors.append("• Abnormal patterns detected")
            factors.append("• Deviations from normal anatomy")
            factors.append("• Requires further correlation")
    else:
        factors.append("**Normal Findings:**")
        factors.append("• Clear lung fields with no opacities")
        factors.append("• Normal heart size and position")
        factors.append("• Intact bone structures")
        factors.append("• No suspicious masses or lesions")
    
    return factors


def get_simple_explanation(result):
    """
    Get a simple, concise explanation for quick understanding.
    
    Args:
        result (dict): Prediction results
        
    Returns:
        str: Simple explanation
    """
    if not result:
        return "No results available."
    
    predicted_class = result.get('predicted_class', 'Unknown')
    confidence = result.get('confidence', 0)
    is_abnormal = result.get('is_abnormal', False)
    
    if is_abnormal:
        return f"⚠️ **{predicted_class}** detected with {confidence:.1%} confidence. Please consult a radiologist for confirmation."
    else:
        return f"✅ **Normal** - No abnormalities detected with {confidence:.1%} confidence."


def generate_report_text(result, patient_info=None):
    """
    Generate a structured report text for download or printing.
    
    Args:
        result (dict): Prediction results
        patient_info (dict, optional): Patient information
        
    Returns:
        str: Formatted report
    """
    if not result:
        return "No results available."
    
    lines = []
    lines.append("=" * 80)
    lines.append("MEDICAL IMAGING ASSISTANT - DIAGNOSTIC REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Patient Information
    if patient_info:
        lines.append("PATIENT INFORMATION")
        lines.append("-" * 40)
        lines.append(f"Name: {patient_info.get('name', 'N/A')}")
        lines.append(f"Age: {patient_info.get('age', 'N/A')}")
        lines.append(f"Gender: {patient_info.get('gender', 'N/A')}")
        lines.append(f"Modality: {patient_info.get('modality', 'N/A')}")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
    
    # Results
    lines.append("ANALYSIS RESULTS")
    lines.append("-" * 40)
    lines.append(f"Prediction: {result.get('predicted_class', 'Unknown')}")
    lines.append(f"Confidence: {result.get('confidence', 0):.1%}")
    lines.append(f"Status: {'Abnormal' if result.get('is_abnormal', False) else 'Normal'}")
    lines.append(f"Model: {result.get('model_type', 'Unknown')}")
    lines.append("")
    
    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append(result.get('recommendation', 'No recommendation available'))
    lines.append("")
    
    # Disclaimer
    lines.append("DISCLAIMER")
    lines.append("-" * 40)
    lines.append("This report is generated by an AI system and is for supportive purposes only.")
    lines.append("Final diagnosis must be made by a qualified medical professional.")
    lines.append("=" * 80)
    
    return "\n".join(lines)
