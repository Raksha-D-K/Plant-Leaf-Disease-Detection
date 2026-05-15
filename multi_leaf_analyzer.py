import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class MultiLeafAnalyzer:
    def __init__(self):
        self.leaf_health_thresholds = {
            'healthy': (0, 5),      # 0-5% disease coverage
            'mild': (5, 15),        # 5-15% disease coverage  
            'moderate': (15, 30),   # 15-30% disease coverage
            'severe': (30, 100)    # >30% disease coverage
        }
    
    def detect_individual_leaves(self, image):
        """
        Detect individual leaves in a multi-leaf image using flexible detection
        Returns: list of leaf regions with their bounding boxes
        """
        # Convert to grayscale and preprocess
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert to HSV for better leaf detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define broader green color ranges for more flexible leaf detection
        green_ranges = [
            (np.array([20, 20, 20]), np.array([100, 255, 255])),    # Very broad green
            (np.array([25, 40, 40]), np.array([90, 255, 255])),     # Standard green
            (np.array([35, 30, 30]), np.array([85, 255, 255])),     # Light green
            (np.array([30, 50, 50]), np.array([80, 255, 255])),     # Medium green
            (np.array([15, 50, 50]), np.array([95, 255, 255])),     # Yellow-green
        ]
        
        # Try different detection strategies
        all_masks = []
        
        # Strategy 1: Combine all green ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower_green, upper_green in green_ranges:
            mask = cv2.inRange(hsv, lower_green, upper_green)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        all_masks.append(combined_mask)
        
        # Strategy 2: Use only the most common green range
        standard_mask = cv2.inRange(hsv, green_ranges[1][0], green_ranges[1][1])
        all_masks.append(standard_mask)
        
        # Strategy 3: Use LAB color space for better green detection
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        lab_green_lower = np.array([0, 120, 0])
        lab_green_upper = np.array([255, 255, 255])
        lab_mask = cv2.inRange(lab, lab_green_lower, lab_green_upper)
        all_masks.append(lab_mask)
        
        # Try each mask and pick the best one
        best_leaf_regions = []
        best_mask = combined_mask
        
        for mask_idx, test_mask in enumerate(all_masks):
            # Apply morphological operations
            kernel_small = np.ones((3,3), np.uint8)
            kernel_medium = np.ones((5,5), np.uint8)
            
            # Remove small noise
            test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_OPEN, kernel_small)
            # Fill gaps
            test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel_medium)
            # Smooth edges
            test_mask = cv2.GaussianBlur(test_mask, (3, 3), 0)
            
            # Threshold to get binary mask
            _, test_mask = cv2.threshold(test_mask, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # More lenient filtering for this attempt
            min_area = 200   # Very small minimum area
            max_area = (img_array.shape[0] * img_array.shape[1]) * 0.5  # Max 50% of image area
            leaf_regions = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Very lenient area filtering
                if min_area <= area <= max_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Very lenient aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:  # Very broad range
                        
                        # Lenient solidity check
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        if solidity > 0.1:  # Very low threshold
                            # Create leaf mask
                            leaf_mask = np.zeros(test_mask.shape, dtype=np.uint8)
                            cv2.fillPoly(leaf_mask, [contour], 255)
                            
                            leaf_regions.append({
                                'leaf_id': i,
                                'bbox': (x, y, w, h),
                                'area': area,
                                'contour': contour,
                                'mask': leaf_mask,
                                'aspect_ratio': aspect_ratio,
                                'solidity': solidity
                            })
            
            # If this mask found more leaves, use it
            if len(leaf_regions) > len(best_leaf_regions):
                best_leaf_regions = leaf_regions
                best_mask = test_mask
        
        # If still no leaves found, try the entire image as one region
        if not best_leaf_regions:
            h, w = img_array.shape[:2]
            full_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            # Create a single leaf region covering the entire image
            best_leaf_regions = [{
                'leaf_id': 0,
                'bbox': (0, 0, w, h),
                'area': w * h,
                'contour': np.array([[0,0], [w,0], [w,h], [0,h]]),
                'mask': full_mask,
                'aspect_ratio': w/h if h > 0 else 1,
                'solidity': 1.0
            }]
            best_mask = full_mask
        
        # Sort by area (largest first)
        best_leaf_regions.sort(key=lambda x: x['area'], reverse=True)
        
        # Limit to reasonable number of leaves
        max_leaves = 15
        best_leaf_regions = best_leaf_regions[:max_leaves]
        
        return best_leaf_regions, best_mask
    
    def analyze_leaf_health(self, image, leaf_region, disease_mask=None, plant_type=None):
        """
        Analyze health of individual leaf region
        """
        x, y, w, h = leaf_region['bbox']
        leaf_id = leaf_region['leaf_id']
        
        # Convert PIL Image to NumPy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Extract leaf region
        leaf_image = img_array[y:y+h, x:x+w]
        
        # Color-based health analysis
        health_metrics = self._analyze_leaf_color(leaf_image, None)
        
        # Predict disease with consistent plant type
        if 'green_ratio' in health_metrics and 'yellow_ratio' in health_metrics and 'brown_ratio' in health_metrics:
            disease_prediction = self._predict_leaf_disease(
                health_metrics['green_ratio'], 
                health_metrics['yellow_ratio'], 
                health_metrics['brown_ratio'], 
                health_metrics['mean_hue'],
                plant_type
            )
            health_metrics['disease_prediction'] = disease_prediction
        
        # Disease-specific analysis if disease mask is provided
        if disease_mask is not None:
            disease_metrics = self._analyze_disease_coverage(
                disease_mask[y:y+h, x:x+w], leaf_mask_region
            )
            health_metrics.update(disease_metrics)
        
        # Add leaf metadata
        health_metrics['leaf_id'] = leaf_id
        health_metrics['bbox'] = leaf_region['bbox']
        health_metrics['area'] = leaf_region['area']
        
        return health_metrics
    
    def _analyze_leaf_color(self, leaf_image, leaf_mask):
        """
        Analyze leaf health based on color characteristics with realistic scoring
        """
        # Convert to HSV
        hsv = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2HSV)
        
        # Apply mask to get only leaf pixels (if mask provided)
        if leaf_mask is not None:
            leaf_pixels_hsv = hsv[leaf_mask > 0]
        else:
            # Use all pixels if no mask provided
            leaf_pixels_hsv = hsv.reshape(-1, 3)
        
        if len(leaf_pixels_hsv) == 0:
            return {'health_score': 50, 'status': 'unknown', 'disease_prediction': 'Unknown'}
        
        # Calculate color statistics
        mean_hue = np.mean(leaf_pixels_hsv[:, 0])
        mean_saturation = np.mean(leaf_pixels_hsv[:, 1])
        mean_value = np.mean(leaf_pixels_hsv[:, 2])
        
        # Analyze color distribution for disease symptoms - fixed brown detection
        green_ratio = np.sum((leaf_pixels_hsv[:, 0] >= 30) & (leaf_pixels_hsv[:, 0] <= 90)) / len(leaf_pixels_hsv)
        yellow_ratio = np.sum((leaf_pixels_hsv[:, 0] >= 20) & (leaf_pixels_hsv[:, 0] <= 35) & 
                            (leaf_pixels_hsv[:, 1] > 50)) / len(leaf_pixels_hsv)
        # Fixed brown detection - broader hue range and proper value range
        brown_ratio = np.sum((leaf_pixels_hsv[:, 0] >= 8) & (leaf_pixels_hsv[:, 0] <= 30) & 
                           (leaf_pixels_hsv[:, 1] >= 20) & (leaf_pixels_hsv[:, 2] <= 120)) / len(leaf_pixels_hsv)
        
        # Calculate health score with realistic scoring for healthy leaves
        health_score = 50  # Base score (neutral)
        
        # Green color bonus (much higher for very healthy leaves)
        if green_ratio > 0.95:  # Nearly perfect green ratio (extremely healthy)
            health_score += (green_ratio - 0.95) * 400  # Max +20 points
        elif green_ratio > 0.9:  # Extremely high green ratio (very healthy)
            health_score += (green_ratio - 0.9) * 200  # Max +10 points
        elif green_ratio > 0.8:  # Very high green ratio (healthy)
            health_score += (green_ratio - 0.8) * 50   # Max +5 points
        elif green_ratio < 0.5:  # Low green ratio (unhealthy)
            health_score -= (0.5 - green_ratio) * 40  # Significant penalty
        
        # Yellow stress penalty (more aggressive)
        if yellow_ratio > 0.05:  # More than 5% yellow
            health_score -= yellow_ratio * 80  # Heavy penalty
        
        # Brown disease penalty (very aggressive)
        if brown_ratio > 0.02:  # More than 2% brown
            health_score -= brown_ratio * 120  # Very heavy penalty
        
        # Saturation factor
        if mean_saturation < 50:  # Low saturation indicates stress
            health_score -= (50 - mean_saturation) * 0.8
        
        # Value (brightness) factor
        if mean_value < 70:  # Dark leaves can indicate issues
            health_score -= (70 - mean_value) * 0.5
        
        # Clamp to valid range
        health_score = max(0, min(100, health_score))
        
        # Determine health status
        if health_score >= 85:
            status = 'healthy'
        elif health_score >= 70:
            status = 'mild_stress'
        elif health_score >= 50:
            status = 'moderate_stress'
        else:
            status = 'severe_stress'
        
        # Predict leaf-specific disease
        disease_prediction = self._predict_leaf_disease(green_ratio, yellow_ratio, brown_ratio, mean_hue)
        
        return {
            'health_score': health_score,
            'status': status,
            'mean_hue': mean_hue,
            'mean_saturation': mean_saturation,
            'mean_value': mean_value,
            'green_ratio': green_ratio,
            'yellow_ratio': yellow_ratio,
            'brown_ratio': brown_ratio,
            'disease_prediction': disease_prediction
        }
    
    def _detect_plant_type_for_image(self, img_array):
        """
        Detect plant type for the entire image (all leaves from same plant)
        Enhanced detection for different plant types (Apple, Tomato, Pepper, etc.)
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Analyze multiple characteristics for better plant detection
        h_channel = hsv[:, :, 0]  # Hue
        s_channel = hsv[:, :, 1]  # Saturation
        v_channel = hsv[:, :, 2]  # Value
        
        # Calculate color distribution statistics
        mean_hue = np.mean(h_channel)
        mean_saturation = np.mean(s_channel)
        mean_value = np.mean(v_channel)
        
        # Analyze green content with different ranges
        green_pixels_broad = np.sum((h_channel >= 25) & (h_channel <= 95))
        green_pixels_narrow = np.sum((h_channel >= 35) & (h_channel <= 85))
        total_pixels = h_channel.size
        
        green_ratio_broad = green_pixels_broad / total_pixels
        green_ratio_narrow = green_pixels_narrow / total_pixels
        
        # Analyze red/orange content (for tomatoes)
        red_pixels = np.sum((h_channel >= 0) & (h_channel <= 10)) + np.sum((h_channel >= 170) & (h_channel <= 180))
        red_ratio = red_pixels / total_pixels
        
        # Analyze yellow-green content (for apple leaves)
        yellow_green_pixels = np.sum((h_channel >= 25) & (h_channel <= 45))
        yellow_green_ratio = yellow_green_pixels / total_pixels
        
        # Analyze leaf shape characteristics (using edge detection)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # Enhanced plant type detection logic for 4 plant types
        if green_ratio_broad > 0.7 and mean_saturation > 60 and edge_density > 0.15:
            # High green, high saturation, moderate edge density - Apple leaves
            if yellow_green_ratio > 0.3:
                return "Apple"
            else:
                return "Tomato"
        elif green_ratio_broad > 0.6 and mean_saturation > 50:
            # Good green content, moderate saturation
            if red_ratio > 0.1:
                return "Tomato"  # Tomatoes often have some red/orange tint
            elif yellow_green_ratio > 0.25:
                return "Apple"   # Apple leaves have yellow-green tint
            elif mean_hue > 40 and mean_hue < 60:
                return "Corn"    # Corn leaves have more yellow-green hue
            elif mean_hue < 30 and edge_density < 0.12:
                return "Potato"  # Potato leaves are broader, less edge density
            elif mean_hue < 35 and mean_saturation < 55:
                return "Potato"  # Potato leaves have lower saturation
            else:
                return "Pepper"
        elif green_ratio_broad > 0.4:
            # Moderate green content
            if mean_hue > 45 and mean_hue < 65:
                return "Corn"    # Corn has distinct yellow-green hue
            elif mean_hue < 30 and edge_density < 0.10:
                return "Potato"  # Potato leaves are broader
            elif mean_hue < 35:
                return "Pepper"
            else:
                return "General Plant"
        else:
            # Low green content
            if mean_hue > 50 and mean_hue < 70:
                return "Corn"    # Even stressed corn has yellow tint
            else:
                return "General Plant"
    
    def _predict_leaf_disease(self, green_ratio, yellow_ratio, brown_ratio, mean_hue, plant_type=None):
        """
        Predict specific disease for individual leaf based on color analysis
        Uses consistent plant type for all leaves from same plant
        """
        # Use provided plant type or detect it once for the entire plant
        if plant_type is None:
            # This should only be called once per plant analysis
            if green_ratio > 0.7:
                plant_type = "Tomato"
            elif green_ratio > 0.5:
                plant_type = "Pepper"
            else:
                plant_type = "General Plant"
        
        # Use scoring system like traditional analysis for consistency
        disease_score = 0
        
        # Check for definite disease symptoms (higher thresholds)
        if brown_ratio > 0.12:  # High brown content
            disease_score += 3
        elif brown_ratio > 0.06:  # Some brown content
            disease_score += 1
        
        if yellow_ratio > 0.25:  # High yellow content
            disease_score += 3
        elif yellow_ratio > 0.15:  # Some yellow content
            disease_score += 1
        
        # Check for healthy symptoms
        if green_ratio > 0.80:  # Very high green content
            disease_score -= 2  # Negative score for healthy
        
        # Determine disease based on score and plant type
        if disease_score >= 2:  # Definitely diseased
            if plant_type == "Apple":
                if brown_ratio > 0.12:
                    return f"{plant_type} Apple Scab" if mean_hue < 12 else f"{plant_type} Fire Blight"
                elif yellow_ratio > 0.25:
                    return f"{plant_type} Chlorosis" if green_ratio < 0.4 else f"{plant_type} Nutrient Deficiency"
                else:
                    return f"{plant_type} Leaf Spot Disease"
            elif plant_type == "Tomato":
                if brown_ratio > 0.12:
                    return f"{plant_type} Early Blight" if mean_hue < 12 else f"{plant_type} Leaf Spot Disease"
                elif yellow_ratio > 0.25:
                    return f"{plant_type} Yellow Leaf Curl" if green_ratio < 0.3 else f"{plant_type} Nutrient Deficiency"
                else:
                    return f"{plant_type} Bacterial Blight"
            elif plant_type == "Corn":
                if brown_ratio > 0.12:
                    return f"{plant_type} Northern Leaf Blight" if mean_hue < 12 else f"{plant_type} Gray Leaf Spot"
                elif yellow_ratio > 0.25:
                    return f"{plant_type} Stewart's Wilt" if green_ratio < 0.3 else f"{plant_type} Nutrient Deficiency"
                else:
                    return f"{plant_type} Common Rust"
            elif plant_type == "Potato":
                if brown_ratio > 0.12:
                    return f"{plant_type} Late Blight" if mean_hue < 12 else f"{plant_type} Early Blight"
                elif yellow_ratio > 0.25:
                    return f"{plant_type} Yellowing Disease" if green_ratio < 0.3 else f"{plant_type} Nutrient Deficiency"
                else:
                    return f"{plant_type} Leaf Spot"
            else:
                return f"{plant_type} Leaf Spot Disease"
        elif disease_score >= 1:  # Mild symptoms
            if plant_type == "Apple":
                return f"{plant_type} Mild Symptoms"
            elif plant_type == "Tomato":
                return f"{plant_type} Mild Symptoms"
            elif plant_type == "Corn":
                return f"{plant_type} Mild Symptoms"
            elif plant_type == "Potato":
                return f"{plant_type} Mild Symptoms"
            else:
                return f"{plant_type} Mild Symptoms"
        else:  # Healthy (score < 1)
            return f"{plant_type} - Healthy"
    
    def _analyze_disease_coverage(self, disease_mask, leaf_mask):
        """
        Calculate disease coverage within leaf area
        """
        # Calculate leaf area
        leaf_area = np.sum(leaf_mask > 0)
        
        if leaf_area == 0:
            return {'disease_coverage': 0, 'disease_status': 'healthy'}
        
        # Calculate disease area
        disease_area = np.sum((disease_mask > 0) & (leaf_mask > 0))
        disease_coverage = (disease_area / leaf_area) * 100
        
        # Determine disease status
        if disease_coverage <= 5:
            disease_status = 'healthy'
        elif disease_coverage <= 15:
            disease_status = 'mild_infection'
        elif disease_coverage <= 30:
            disease_status = 'moderate_infection'
        else:
            disease_status = 'severe_infection'
        
        return {
            'disease_coverage': disease_coverage,
            'disease_status': disease_status,
            'disease_area': disease_area,
            'leaf_area': leaf_area
        }
    
    def calculate_overall_plant_health(self, leaf_health_results):
        """
        Calculate overall plant health from individual leaf analyses
        """
        if not leaf_health_results:
            return {
                'overall_health_score': 0,
                'overall_status': 'unknown',
                'total_leaves': 0,
                'healthy_leaves': 0,
                'stressed_leaves': 0,
                'recommendations': []
            }
        
        # Aggregate metrics
        total_leaves = len(leaf_health_results)
        health_scores = [leaf['health_score'] for leaf in leaf_health_results]
        
        # Count leaves by status
        healthy_count = sum(1 for leaf in leaf_health_results if leaf['status'] == 'healthy')
        stressed_count = total_leaves - healthy_count
        
        # Calculate overall health score (weighted average)
        if 'disease_coverage' in leaf_health_results[0]:
            # If disease analysis is available, weight by disease coverage
            disease_coverages = [leaf.get('disease_coverage', 0) for leaf in leaf_health_results]
            avg_disease_coverage = np.mean(disease_coverages)
            
            # Adjust health score based on disease coverage
            base_health_score = np.mean(health_scores)
            disease_penalty = min(30, avg_disease_coverage * 0.5)  # Max 30 point penalty
            overall_health_score = max(0, base_health_score - disease_penalty)
        else:
            overall_health_score = np.mean(health_scores)
        
        # Determine overall status
        if overall_health_score >= 80:
            overall_status = 'healthy'
        elif overall_health_score >= 60:
            overall_status = 'mildly_stressed'
        elif overall_health_score >= 40:
            overall_status = 'moderately_stressed'
        else:
            overall_status = 'severely_stressed'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(leaf_health_results, overall_status)
        
        return {
            'overall_health_score': overall_health_score,
            'overall_status': overall_status,
            'total_leaves': total_leaves,
            'healthy_leaves': healthy_count,
            'stressed_leaves': stressed_count,
            'avg_health_score': np.mean(health_scores),
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, leaf_health_results, overall_status):
        """
        Generate plant care recommendations based on analysis
        """
        recommendations = []
        
        if overall_status == 'healthy':
            recommendations.append("Plant appears healthy. Continue regular monitoring and care.")
        elif overall_status == 'mildly_stressed':
            recommendations.append("Mild stress detected. Check watering and light conditions.")
            recommendations.append("Monitor for pest activity and environmental changes.")
        elif overall_status == 'moderately_stressed':
            recommendations.append("Moderate stress detected. Immediate attention recommended.")
            recommendations.append("Check soil moisture, nutrient levels, and potential disease symptoms.")
        else:  # severely_stressed
            recommendations.append("Severe stress detected. Urgent intervention required.")
            recommendations.append("Isolate plant if disease suspected. Consult plant care specialist.")
        
        # Specific recommendations based on disease coverage
        disease_coverages = [leaf.get('disease_coverage', 0) for leaf in leaf_health_results]
        avg_disease_coverage = np.mean(disease_coverages) if disease_coverages else 0
        
        if avg_disease_coverage > 20:
            recommendations.append("High disease coverage detected. Consider fungicide treatment.")
        elif avg_disease_coverage > 10:
            recommendations.append("Moderate disease symptoms. Early treatment recommended.")
        
        return recommendations
    
    def create_multi_leaf_visualization(self, image, leaf_regions, leaf_health_results):
        """
        Create visualization showing individual leaf analyses
        """
        # Convert PIL Image to NumPy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Create annotated image
        annotated_image = img_array.copy()
        
        # Draw bounding boxes and health status for each leaf
        for leaf_result in leaf_health_results:
            leaf_id = leaf_result['leaf_id']
            bbox = leaf_result['bbox']
            health_score = leaf_result['health_score']
            status = leaf_result['status']
            
            x, y, w, h = bbox
            
            # Color based on health status
            colors = {
                'healthy': (0, 255, 0),
                'mild_stress': (255, 255, 0),
                'moderate_stress': (255, 165, 0),
                'severe_stress': (0, 0, 255)
            }
            color = colors.get(status, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            
            # No text labels - just bounding boxes
            # This keeps the visualization clean without cluttering text
        
        return annotated_image

# Utility function for Streamlit integration
def analyze_multiple_leaves(image, disease_mask=None, debug=False):
    """
    Main function for multi-leaf analysis
    Ensures all leaves from same plant get consistent plant type
    """
    analyzer = MultiLeafAnalyzer()
    
    # Convert PIL Image to NumPy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Detect individual leaves
    leaf_regions, green_mask = analyzer.detect_individual_leaves(img_array)
    
    if not leaf_regions:
        return {"error": "No leaves detected in the image"}
    
    # Detect plant type once for the entire image (all leaves from same plant)
    plant_type = analyzer._detect_plant_type_for_image(img_array)
    
    # Analyze each leaf with consistent plant type
    leaf_health_results = []
    for region in leaf_regions:
        health_metrics = analyzer.analyze_leaf_health(img_array, region, disease_mask, plant_type)
        leaf_health_results.append(health_metrics)
    
    # Calculate overall plant health
    overall_health = analyzer.calculate_overall_plant_health(leaf_health_results)
    
    # Create visualization
    annotated_image = analyzer.create_multi_leaf_visualization(img_array, leaf_regions, leaf_health_results)
    
    return {
        'annotated_image': annotated_image,
        'leaf_regions': leaf_regions,
        'leaf_health_results': leaf_health_results,
        'overall_health': overall_health,
        'green_mask': green_mask,
        'plant_type': plant_type  # Add plant type to results
    }

def simple_fallback_analysis(image):
    """
    Simple fallback analysis that always works by analyzing the entire image
    """
    analyzer = MultiLeafAnalyzer()
    
    # Convert to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    h, w = img_array.shape[:2]
    
    # Create a single region covering the entire image
    full_region = {
        'leaf_id': 0,
        'bbox': (0, 0, w, h),
        'area': w * h,
        'contour': np.array([[0,0], [w,0], [w,h], [0,h]]),
        'mask': np.ones((h, w), dtype=np.uint8) * 255,
        'aspect_ratio': w/h if h > 0 else 1,
        'solidity': 1.0
    }
    
    # Analyze this region
    health_metrics = analyzer.analyze_leaf_health(image, full_region)
    
    # Calculate overall health
    overall_health = analyzer.calculate_overall_plant_health([health_metrics])
    
    # Create simple visualization (no text labels)
    annotated_image = img_array.copy()
    cv2.rectangle(annotated_image, (0, 0), (w, h), (0, 255, 0), 3)
    # No text labels - clean visualization
    
    return {
        'annotated_image': annotated_image,
        'leaf_regions': [full_region],
        'leaf_health_results': [health_metrics],
        'overall_health': overall_health,
        'green_mask': np.ones((h, w), dtype=np.uint8) * 255,
        'fallback_used': True,
        'message': 'Used fallback analysis - entire image treated as one leaf region'
    }
