# YOLO Detection Troubleshooting Guide

## 🔍 "No Objects Detected" - Common Causes & Solutions

### **Issue**: "No objects detected with current confidence threshold"

---

## 🚀 Quick Solutions (Try These First)

### 1. **Lower Confidence Threshold**
The most common cause is confidence threshold being too high.

**Current Setting**: 0.5 (50%)
**Try**: 0.3 (30%) or 0.2 (20%)

**How**:
- In the app, move the "Confidence Threshold" slider to 0.3
- Try again with your image

### 2. **Use Custom Trained Model**
The default YOLOv8n model is trained on general objects (people, cars, etc.), not plant diseases.

**Solution**:
- ✅ Check "Use Custom Trained Model"
- Enter your model filename (e.g., `best.pt`)
- Your model should detect plant-specific diseases

### 3. **Check Image Quality**
Poor image quality can prevent detection.

**Requirements**:
- Clear, well-lit images
- Disease symptoms visible
- Proper focus
- Minimum resolution: 640x640

---

## 🔧 Detailed Troubleshooting

### **Step 1: Check Model Loading**
Run the app and check the console output:

```
✅ Good Output:
Custom YOLO model loaded from: best.pt
YOLO model loaded on cpu
Model classes: ['apple_scab', 'apple_healthy', 'tomato_early_blight']

❌ Bad Output:
Using pretrained YOLOv8n model
Model classes: ['person', 'bicycle', 'car', 'motorcycle']
```

### **Step 2: Check Detection Process**
With debugging enabled, you'll see:

```
✅ Good Output:
Image shape: (640, 480, 3)
Confidence threshold: 0.5
Using custom model: True
Found 2 boxes in this result
Detection - Class: 0, Confidence: 0.876
Detection - Class: 1, Confidence: 0.234
Total detections found: 2
Detections above threshold: 1

❌ Bad Output:
Image shape: (640, 480, 3)
Confidence threshold: 0.5
Using custom model: False
Total detections found: 0
Detections above threshold: 0
No detections above confidence threshold.
```

### **Step 3: Verify Model Classes**
Your model should have plant disease classes:

```
✅ Expected Classes:
['apple_scab', 'apple_healthy', 'tomato_early_blight', 'potato_healthy']

❌ Wrong Classes:
['person', 'car', 'truck', 'bicycle']
```

---

## 🎯 Specific Solutions

### **Solution A: Using Pretrained Model**
If you're using the default YOLOv8n model:

1. **Lower confidence to 0.1**
2. **Upload images with common objects** (people, cars, etc.) to test
3. **Expect generic object detection**, not plant diseases

### **Solution B: Using Custom Model**
If you have a trained model:

1. **Ensure model file exists** (e.g., `best.pt`)
2. **Check model classes** match your training data
3. **Verify model was trained on plant diseases**

### **Solution C: Model Training Issues**
If your custom model doesn't work:

1. **Check training data quality**
2. **Verify model export format** (should be `.pt`)
3. **Test with training images**
4. **Re-train with more data if needed**

---

## 🖼️ Image Requirements

### **Good Images**:
- ✅ Clear disease symptoms
- ✅ Good lighting
- ✅ Proper focus
- ✅ Single leaf or multiple leaves
- ✅ High resolution

### **Bad Images**:
- ❌ Blurry or out of focus
- ❌ Poor lighting
- ❌ No visible symptoms
- ❌ Very small resolution
- ❌ Heavily processed/filtered

---

## 📊 Testing Different Scenarios

### **Test 1: Pretrained Model**
```
Method: YOLO Detection
Custom Model: ❌ unchecked
Confidence: 0.1
Expected: Generic objects (person, car, etc.)
```

### **Test 2: Custom Model**
```
Method: YOLO Detection  
Custom Model: ✅ checked
Model Path: best.pt
Confidence: 0.5
Expected: Plant diseases
```

### **Test 3: Low Confidence**
```
Method: YOLO Detection
Custom Model: ❌ unchecked  
Confidence: 0.1
Expected: More detections (possibly false positives)
```

---

## 🛠️ Advanced Troubleshooting

### **Check Model File**:
```python
from ultralytics import YOLO
model = YOLO('best.pt')
print(f"Model classes: {list(model.names.values())}")
```

### **Test Direct Inference**:
```python
from PIL import Image
image = Image.open('your_test_image.jpg')
results = model(image, conf=0.5)
print(f"Raw detections: {len(results[0].boxes)}")
```

### **Verify Image Format**:
- Supported: `.jpg`, `.jpeg`, `.png`
- RGB format preferred
- No transparency channels

---

## 🎯 Expected Behavior

### **With Custom Trained Model**:
```
✅ Found 2 objects

Detected: Apple Scab
Confidence: 0.87

Detected: Apple Healthy  
Confidence: 0.92
```

### **With Pretrained Model**:
```
✅ Found 1 object

Detected: person
Confidence: 0.76
```

---

## 📞 Need More Help?

1. **Check console output** for debugging information
2. **Try different confidence thresholds**
3. **Test with different images**
4. **Verify your custom model file**
5. **Check the model classes** match your expectations

The debugging output will show you exactly what's happening during detection!
