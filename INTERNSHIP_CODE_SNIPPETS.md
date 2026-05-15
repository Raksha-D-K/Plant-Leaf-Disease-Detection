# Plant Disease Detection System - Important Code Snippets
**Internship Report - Core Implementation Components**

---

## 1. Module Imports and Dependencies

```python
# Core libraries for the plant disease detection system
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import os
from dotenv import load_dotenv

# Custom modules for specialized functionality
from multi_leaf_analyzer import MultiLeafAnalyzer
from leaf_health_analyzer import LeafHealthAnalyzer
from plantnet_leaf_verifier import PlantNetLeafVerifier
from maskrcnn_segmentation import segment_with_maskrcnn
from disease_chatbot import create_disease_chatbot_interface

# Load environment variables for API keys
load_dotenv()
```

**Explanation:** These imports provide the foundation for the entire plant disease detection system, including machine learning frameworks, image processing, web interface, and specialized analysis modules.

---

## 2. Disease Database Configuration

```python
# Comprehensive plant disease knowledge base
DISEASE_DB = {
    "Apple Scab": {
        "pesticides": ["Myclobutanil", "Mancozeb", "Sulfur"],
        "desc": "Dark olive-green spots on leaves and fruit. Common in humid conditions.",
        "risk": "Moderate",
        "color": "#ef4444",
        "action": "Apply fungicide during dormant season and monitor weather conditions."
    },
    "Tomato Early Blight": {
        "pesticides": ["Chlorothalonil", "Copper fungicides", "Mancozeb"],
        "desc": "Dark brown spots with concentric rings on lower leaves first.",
        "risk": "High",
        "color": "#f97316", 
        "action": "Remove infected leaves, apply fungicide, ensure proper spacing."
    },
    # ... more disease entries
}
```

**Explanation:** This database serves as the knowledge base containing disease information, treatment recommendations, and risk assessments for various plant diseases.

---

## 3. Image Upload and Preprocessing

```python
# File upload interface with validation
uploaded_file = st.file_uploader(
    "Drop image here...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file)
    st.image(image, width=600, caption="Target Specimen")
    
    # Store image in session state for processing
    st.session_state.current_image = image
```

**Explanation:** This code handles user image uploads through the Streamlit interface, validates file types, and prepares images for analysis.

---

## 4. Leaf Verification System

```python
# PlantNet API integration for leaf verification
class PlantNetLeafVerifier:
    def __init__(self):
        self.api_key = os.getenv("PLANTNET_API_KEY")
        self.api_url = "https://my-api.plantnet.org/v2/identify"
    
    def verify_leaf_with_plantnet(self, image):
        """Verify if uploaded image is actually a leaf"""
        try:
            # Try PlantNet API first
            result = self._try_plantnet_api(image)
            if result['success']:
                return result
        except Exception as e:
            print(f"PlantNet API failed: {e}")
        
        # Fallback to computer vision
        return self._simple_fallback_detection(image)
    
    def _simple_fallback_detection(self, image):
        """Fallback leaf detection using color analysis"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        
        # Detect green leaf colors
        green_mask = cv2.inRange(img_cv, 
            np.array([20, 40, 40]), 
            np.array([100, 255, 255])
        )
        
        # Calculate leaf coverage
        leaf_pixels = cv2.countNonZero(green_mask)
        total_pixels = img_cv.shape[0] * img_cv.shape[1]
        leaf_ratio = leaf_pixels / total_pixels
        
        # Text detection to reject documents
        has_text = self._detect_text_simple(img_cv)
        
        if has_text:
            return {'success': False, 'message': "Document detected - not a leaf"}
        elif leaf_ratio > 0.03:
            return {'success': True, 'message': "Leaf detected", 'confidence': leaf_ratio}
        else:
            return {'success': False, 'message': "No leaf detected"}
```

**Explanation:** This system verifies that uploaded images contain actual leaves using both the PlantNet API and computer vision techniques, preventing analysis of non-leaf images.

---

## 5. AI Model Prediction

```python
# Real-time disease prediction using trained model
def predict_disease(image):
    """Predict plant disease from leaf image"""
    try:
        # Load trained model
        model = tf.keras.models.load_model("trained_model.h5")
        
        # Preprocess image
        img_array = np.array(image)
        img_resized = tf.image.resize(img_array, [128, 128])
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Map to disease names
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy',
            # ... more classes
        ]
        
        predicted_class = class_names[predicted_class_idx]
        disease_name = predicted_class.replace('___', ' ').replace('_', ' ')
        
        return {
            'disease': disease_name,
            'confidence': confidence,
            'all_probabilities': predictions[0].tolist()
        }
        
    except Exception as e:
        return {'error': f"Prediction failed: {e}"}
```

**Explanation:** This function uses the trained deep learning model to predict plant diseases from leaf images, providing confidence scores and probability distributions.

---

## 6. Image Segmentation with Mask R-CNN

```python
# Leaf segmentation using Mask R-CNN
def segment_with_maskrcnn(image):
    """Segment individual leaves from complex images"""
    try:
        # Load Mask R-CNN model
        model = load_maskrcnn_model()
        
        # Preprocess for segmentation
        img_array = np.array(image)
        results = model.detect([img_array], verbose=0)
        
        # Extract masks and bounding boxes
        r = results[0]
        masks = r['masks']
        boxes = r['rois']
        class_ids = r['class_ids']
        
        # Create segmented visualization
        segmented_image = img_array.copy()
        for i in range(len(masks)):
            if class_ids[i] == 1:  # Leaf class
                mask = masks[:, :, i]
                segmented_image[mask] = [255, 0, 0]  # Red overlay
        
        return {
            'segmented_image': segmented_image,
            'leaf_count': len([cid for cid in class_ids if cid == 1]),
            'masks': masks,
            'boxes': boxes
        }
        
    except Exception as e:
        return {'error': f"Segmentation failed: {e}"}
```

**Explanation:** This code uses Mask R-CNN to identify and segment individual leaves from complex plant images, enabling detailed analysis of multiple leaves in a single image.

---

## 7. Multi-Leaf Analysis

```python
# Analysis of multiple leaves in single image
class MultiLeafAnalyzer:
    def __init__(self):
        self.leaf_health_thresholds = {
            'healthy': (0, 5),
            'mild': (5, 15),
            'moderate': (15, 30),
            'severe': (30, 100)
        }
    
    def detect_individual_leaves(self, image):
        """Detect separate leaves in multi-leaf image"""
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Multiple green color ranges for flexibility
        green_ranges = [
            (np.array([20, 20, 20]), np.array([100, 255, 255])),
            (np.array([25, 40, 40]), np.array([90, 255, 255])),
            (np.array([35, 30, 30]), np.array([85, 255, 255]))
        ]
        
        # Combine all green masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find individual leaf contours
        contours, _ = cv2.findContours(combined_mask, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        leaf_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Filter by size
                x, y, w, h = cv2.boundingRect(contour)
                leaf_regions.append({'bbox': (x, y, w, h), 'contour': contour})
        
        return leaf_regions
    
    def analyze_multiple_leaves(self, image):
        """Analyze health of all detected leaves"""
        leaf_regions = self.detect_individual_leaves(image)
        leaf_analyses = []
        
        for i, region in enumerate(leaf_regions):
            # Extract individual leaf
            x, y, w, h = region['bbox']
            leaf_img = image.crop((x, y, x+w, y+h))
            
            # Analyze individual leaf health
            health_score = self._calculate_leaf_health(leaf_img)
            disease_status = self._determine_health_status(health_score)
            
            leaf_analyses.append({
                'leaf_id': i,
                'bbox': region['bbox'],
                'health_score': health_score,
                'status': disease_status
            })
        
        return {
            'total_leaves': len(leaf_analyses),
            'leaf_analyses': leaf_analyses,
            'overall_health': self._calculate_overall_health(leaf_analyses)
        }
```

**Explanation:** This system detects and analyzes multiple leaves within a single image, providing individual health assessments for each leaf and an overall plant health summary.

---

## 8. Leaf Health Analysis

```python
# Comprehensive leaf health assessment
class LeafHealthAnalyzer:
    def __init__(self):
        self.color_ranges = {
            'healthy_green': (np.array([35, 40, 40]), np.array([85, 255, 255])),
            'yellowing': (np.array([15, 40, 40]), np.array([35, 255, 255])),
            'brown': (np.array([8, 40, 40]), np.array([25, 255, 255])),
            'dry': (np.array([0, 0, 50]), np.array([180, 50, 100]))
        }
    
    def analyze_leaf_health(self, image):
        """Complete leaf health analysis"""
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Color analysis
        color_metrics = self._analyze_leaf_colors(hsv)
        
        # Disease pattern detection
        disease_patterns = self._detect_disease_patterns(hsv)
        
        # Health score calculation
        health_score = self._calculate_leaf_health_score(color_metrics, disease_patterns)
        
        # Disease prediction
        disease_type = self._predict_disease_type(color_metrics, disease_patterns)
        
        return {
            'health_score': health_score,
            'status': self._get_health_status(health_score),
            'color_metrics': color_metrics,
            'disease_patterns': disease_patterns,
            'predicted_disease': disease_type
        }
    
    def _analyze_leaf_colors(self, hsv):
        """Analyze color distribution in leaf"""
        metrics = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            pixels = cv2.countNonZero(mask)
            metrics[color_name] = (pixels / total_pixels) * 100
        
        return metrics
    
    def _calculate_leaf_health_score(self, color_metrics, patterns):
        """Calculate overall leaf health score (0-100)"""
        score = 100
        
        # Penalize yellowing
        yellow_penalty = color_metrics.get('yellowing', 0) * 2
        score -= yellow_penalty
        
        # Penalize brown spots
        brown_penalty = color_metrics.get('brown', 0) * 3
        score -= brown_penalty
        
        # Penalize disease patterns
        if patterns.get('spots', 0) > 0:
            score -= patterns['spots'] * 0.5
        if patterns.get('irregular', 0) > 0:
            score -= patterns['irregular'] * 0.3
        
        return max(0, min(100, score))
```

**Explanation:** This module performs detailed health analysis by examining color distributions, detecting disease patterns, and calculating comprehensive health scores for individual leaves.

---

## 9. Chatbot Integration

```python
# AI-powered disease chatbot
class DiseaseChatbot:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("api_key"),
            base_url="https://openrouter.ai/api/v1"
        )
    
    def create_chatbot_interface(self):
        """Create interactive chatbot interface"""
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle user input
        if prompt := st.chat_input("Ask about plant diseases..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                response = self._generate_response(prompt)
                st.markdown(response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def _generate_response(self, prompt):
        """Generate AI response using OpenAI"""
        system_prompt = """
        You are an expert agricultural chatbot specializing in plant diseases. 
        You have knowledge of various plant diseases, their symptoms, treatments, 
        and prevention methods. Be helpful, accurate, and provide practical advice.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {e}"
```

**Explanation:** This chatbot provides users with interactive assistance for plant disease questions, leveraging AI to give accurate and helpful advice.

---

## 10. Model Training Code

```python
# Improved plant disease model training
class ImprovedPlantDiseaseTrainer:
    def __init__(self, img_size=224, batch_size=32):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        
        # Data augmentation pipeline
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip('horizontal_and_vertical'),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2)
        ])
    
    def create_transfer_learning_model(self):
        """Create model with transfer learning"""
        # Use EfficientNetB0 as base model
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build complete model
        inputs = layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        x = self.data_augmentation(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(38, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def train_model(self, train_dataset, val_dataset):
        """Train the model with two-phase approach"""
        model = self.create_transfer_learning_model()
        
        # Phase 1: Train with frozen base model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history_phase1 = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
        
        # Phase 2: Fine-tune with unfrozen layers
        model.layers[1].trainable = True  # Unfreeze base model
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        history_phase2 = model.fit(train_dataset, validation_data=val_dataset, epochs=20)
        
        return model, history_phase1, history_phase2
```

**Explanation:** This training code implements modern deep learning techniques including transfer learning, data augmentation, and two-phase training to create an accurate plant disease detection model.

---

## 11. Testing and Evaluation

```python
# Model evaluation and testing
def evaluate_model(model, test_dataset):
    """Comprehensive model evaluation"""
    # Get predictions
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    precision = np.diagonal(cm) / np.sum(cm, axis=0)
    recall = np.diagonal(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        'accuracy': accuracy,
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1_score),
        'confusion_matrix': cm,
        'classification_report': report
    }

# Test with unknown images
def test_unknown_images(model, image_paths):
    """Test model with unknown leaf images"""
    results = []
    for img_path in image_paths:
        # Load and preprocess
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_batch)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        results.append({
            'image': img_path,
            'predicted_class': class_names[predicted_class],
            'confidence': confidence,
            'top_3': np.argsort(predictions[0])[-3:][::-1]
        })
    
    return results
```

**Explanation:** This testing code evaluates model performance using standard metrics and tests the model's ability to handle unknown leaf images, providing comprehensive assessment of model reliability.

---

## 12. Main Application Interface

```python
# Main Streamlit application
def main():
    # Page navigation
    menu = st.sidebar.selectbox("Navigation", [
        "🏠 Home Dashboard", 
        "Leaf Analysis", 
        "Disease Chatbot",
        "Fleet Dashboard"
    ])
    
    if menu == "Leaf Analysis":
        st.header("Plant Disease Analysis")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file and st.button("Analyze"):
            image = Image.open(uploaded_file)
            
            # Step 1: Verify it's a leaf
            verifier = PlantNetLeafVerifier()
            leaf_check = verifier.verify_leaf_with_plantnet(image)
            
            if not leaf_check['success']:
                st.error("Not a valid leaf image!")
                st.warning(leaf_check['message'])
                return
            
            # Step 2: Disease prediction
            prediction = predict_disease(image)
            
            # Step 3: Display results
            st.success(f"Disease Detected: {prediction['disease']}")
            st.progress(int(prediction['confidence'] * 100), 
                       text=f"Confidence: {prediction['confidence']:.1%}")
            
            # Step 4: Additional analyses
            if st.checkbox("Detailed Health Analysis"):
                health_results = analyze_leaf_health(image)
                st.json(health_results)
            
            if st.checkbox("Multi-Leaf Analysis"):
                multi_results = analyze_multiple_leaves(image)
                st.json(multi_results)
    
    elif menu == "Disease Chatbot":
        create_disease_chatbot_interface()

if __name__ == "__main__":
    main()
```

**Explanation:** This main application code integrates all components into a user-friendly interface, providing seamless navigation between different analysis tools and features.

---

## Summary

This codebase demonstrates a comprehensive plant disease detection system incorporating:

- **Computer Vision**: Image processing, segmentation, and feature extraction
- **Deep Learning**: Transfer learning with EfficientNet for disease classification
- **API Integration**: PlantNet API for plant verification
- **Multi-Modal Analysis**: Health scoring, pattern detection, and multi-leaf analysis
- **Interactive Interface**: Streamlit-based web application with chatbot
- **Robust Testing**: Comprehensive evaluation and validation procedures

The system successfully addresses real-world agricultural challenges through advanced AI techniques and user-friendly design.
