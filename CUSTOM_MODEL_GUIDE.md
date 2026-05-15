# Using Your Trained YOLOv8 Model

## 🚀 Quick Setup

### 1. Place Your Model File
Put your trained YOLOv8 model in the project directory:
```
d:/skill/skill/
├── best.pt                    # Your trained model
├── yolov8n_custom.pt         # Alternative model name
└── app.py                    # Main application
```

### 2. Supported Model Classes
The system automatically recognizes these plant disease classes:
- `apple_scab` → Apple Scab
- `apple_healthy` → Apple Healthy  
- `apple_rust` → Apple Cedar Rust
- `corn_rust` → Corn Common Rust
- `potato_early_blight` → Potato Early Blight
- `potato_healthy` → Potato Healthy
- `tomato_early_blight` → Tomato Early Blight
- `tomato_yellow_virus` → Tomato Yellow Curl Virus
- `tomato_healthy` → Tomato Healthy
- `strawberry_scorch` → Strawberry Leaf Scorch

### 3. Using the Custom Model

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Select Detection Method:**
   - Choose "YOLO Detection" or "Combined AI Analysis"

3. **Enable Custom Model:**
   - ✅ Check "Use Custom Trained Model"
   - Enter your model filename (e.g., `best.pt`)
   - Adjust confidence threshold if needed

4. **Upload & Analyze:**
   - Upload your plant leaf image
   - Click "EXECUTE ANALYSIS"

## 🎯 Expected Results

### With Custom Trained Model:
- ✅ Accurate disease detection
- ✅ Proper disease names
- ✅ Risk level assessment
- ✅ Bounding boxes on detected diseases

### Example Output:
```
🎯 YOLO Detection Results
✅ Found 2 objects

Detected: Apple Scab
Confidence: 0.87

Detected: Apple Healthy  
Confidence: 0.92
```

## 🔧 Model Requirements

### Training Data Format:
- Images: `.jpg`, `.jpeg`, `.png`
- Labels: YOLO format (`.txt` files)
- Classes: Plant disease names

### Model Architecture:
- YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, or YOLOv8x
- Input size: 640x640 (recommended)
- Classes: Plant diseases only

## 🛠️ Troubleshooting

### Model Not Found:
```
Error: Model not loaded
```
- Check model filename spelling
- Ensure model file is in project directory
- Verify model is valid `.pt` file

### No Detections:
```
No objects detected with current confidence threshold
```
- Lower confidence threshold (try 0.3)
- Check if model classes match image content
- Ensure proper image lighting and focus

### Unknown Classes:
```
Detected: unknown_class
```
- Add class mapping in `yolo_detector.py`
- Update `_map_to_disease_db()` method

## 📝 Training Tips

### Dataset Recommendations:
- Minimum 100 images per class
- Balanced class distribution
- Various lighting conditions
- Different disease stages

### Augmentation:
- Rotation: ±15°
- Brightness: ±20%
- Contrast: ±15%
- Saturation: ±10%

## 🎨 Custom Class Mapping

To add new disease classes, edit `yolo_detector.py`:

```python
def _map_to_disease_db(self, class_name):
    disease_mappings = {
        'your_custom_class': {
            'name': 'Your Disease Name',
            'risk': 'High',
            'description': 'Disease description here'
        },
        # Add more mappings...
    }
```

## 📊 Performance Metrics

### Good Performance Indicators:
- ✅ Confidence > 0.8 for clear cases
- ✅ Precision > 85%
- ✅ Recall > 80%
- ✅ Inference time < 2 seconds

### Optimization Tips:
- Use YOLOv8n for speed
- Use YOLOv8m for accuracy
- Batch size: 16-32
- Epochs: 50-100

## 🚀 Next Steps

1. **Test with validation images**
2. **Fine-tune confidence threshold**
3. **Add missing class mappings**
4. **Monitor performance on real data**
5. **Retrain with new data if needed**

---

**Need help?** Check the console output for detailed error messages and model loading information.
