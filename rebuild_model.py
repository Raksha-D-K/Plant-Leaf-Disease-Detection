import tensorflow as tf
import os
import json

# Load the model.weights.h5 and metadata to reconstruct the model
model_dir = "model_working_copy.keras"

print("🔄 Reconstructing model from updated weights...")

try:
    # Read the config to understand the model structure
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Config loaded: {config}")
    
    # The config contains the model architecture
    if "config" in config:
        model_config = config["config"]
        # Recreate model from config
        model = tf.keras.Sequential.from_config(model_config)
        print("✅ Model architecture reconstructed from config")
        
        # Load weights
        weights_path = os.path.join(model_dir, "model.weights.h5")
        model.load_weights(weights_path)
        print("✅ Weights loaded")
        
        # Save as standalone .h5 file
        model.save("updated_plant_model.h5")
        print("✅ Model saved to updated_plant_model.h5")
except Exception as e:
    print(f"⚠️ Reconstruction failed: {e}")
    import traceback
    traceback.print_exc()
