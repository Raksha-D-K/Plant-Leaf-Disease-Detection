from disease_predictor import predict_plant_disease
from PIL import Image
import numpy as np

# Create a dummy test image
test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

# Make a prediction
result = predict_plant_disease(test_img)
print(f"Prediction: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")
print("✅ Prediction system working!")
