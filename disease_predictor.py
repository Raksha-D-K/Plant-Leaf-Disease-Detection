import tensorflow as tf
import numpy as np
from PIL import Image
import os

class DiseasePredictor:
    def __init__(self, model_path="model_working_copy.keras"):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        self.load_model()

    def load_model(self):
        """Load the trained Keras model - using updated plant-diseases-detection-cnn-97-acc model"""
        model_paths = [
            "updated_plant_model.h5",
            "model_working_copy.keras",
            "trained_model.keras"
        ]
        
        for path in model_paths:
            if not os.path.exists(path):
                continue
            
            try:
                print(f"🔄 Attempting to load updated model from: {path}")
                self.model = tf.keras.models.load_model(path, custom_objects=None)
                print(f"✅ Updated disease prediction model (97% accuracy) loaded from {path}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load from {path}: {e}")
                continue
        
        print("❌ Failed to load the updated model. Available model files:")
        for path in model_paths:
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"   {exists} {path}")

    def preprocess_image(self, image):
        """
        Preprocess image for model prediction
        - Resize to 150x150 (as per plant-diseases-detection-cnn-97-acc model)
        - Normalize to [0,1]
        - Add batch dimension
        """
        if isinstance(image, Image.Image):
            # Convert PIL to numpy array
            img_array = np.array(image)
        else:
            img_array = image.copy()

        # Ensure RGB format
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel

        # Resize to 150x150 (updated model input size)
        img_resized = tf.image.resize(img_array, [150, 150])

        # Normalize to [0,1]
        img_normalized = img_resized / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch

    def predict_disease(self, image):
        """
        Predict disease from leaf image using the trained model

        Args:
            image: PIL Image or numpy array

        Returns:
            dict: {
                'disease': predicted_class_name,
                'confidence': prediction_confidence,
                'all_predictions': dict of all class probabilities
            }
        """
        if self.model is None:
            return {
                'disease': 'Model not loaded',
                'confidence': 0.0,
                'all_predictions': {}
            }

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)

            # Get the predicted class index and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])

            # Get predicted class name
            predicted_class = self.class_names[predicted_class_idx]

            # Create dictionary of all predictions
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = float(predictions[0][i])

            return {
                'disease': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'disease': 'Prediction failed',
                'confidence': 0.0,
                'all_predictions': {}
            }

# Global predictor instance
_predictor = None

def get_disease_predictor():
    """Get or create the global disease predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor

def predict_plant_disease(image):
    """
    Convenience function to predict plant disease from image

    Args:
        image: PIL Image or numpy array

    Returns:
        dict: prediction results
    """
    predictor = get_disease_predictor()
    return predictor.predict_disease(image)