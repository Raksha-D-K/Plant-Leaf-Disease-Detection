import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class LeafHealthAnalyzer:
    def __init__(self):
        self.healthy_color_ranges = {
            'green_healthy': {
                'lower': np.array([30, 20, 20]),  # Wider range for healthy green
                'upper': np.array([90, 255, 255])
            },
            'yellow_stress': {
                'lower': np.array([15, 50, 50]),
                'upper': np.array([35, 255, 255])
            },
            'brown_disease': {
                'lower': np.array([8, 30, 30]),
                'upper': np.array([25, 255, 255])
            }
        }
    
    def analyze_leaf_image(self, image):
        """
        Analyze leaf image for disease symptoms
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Detect individual leaves first
        leaf_regions = self._detect_individual_leaves(hsv)
        
        # Analyze each leaf separately
        leaf_analyses = []
        for i, region in enumerate(leaf_regions):
            leaf_analysis = self._analyze_individual_leaf(hsv, region, i)
            leaf_analyses.append(leaf_analysis)
        
        # Calculate overall health from all leaves
        overall_health = self._calculate_overall_health(leaf_analyses)
        
        # Overall color analysis
        overall_color_metrics = self._analyze_leaf_colors(hsv)
        
        # Overall disease patterns
        overall_patterns = self._detect_disease_patterns(hsv)
        
        # Detect plant type
        plant_type = self._detect_plant_type(img_array, hsv)
        
        # Determine overall disease type with plant name
        disease_prediction = self._predict_disease_type(overall_color_metrics, overall_patterns, plant_type)
        
        # Create visualization with detected leaves
        annotated_image = self._create_leaf_visualization(img_array, leaf_analyses)
        
        return {
            'health_score': overall_health['score'],
            'status': overall_health['status'],
            'disease_prediction': disease_prediction,
            'plant_type': plant_type,
            'color_analysis': overall_color_metrics,
            'patterns': overall_patterns,
            'leaf_count': len(leaf_regions),
            'leaf_analyses': leaf_analyses,
            'annotated_image': annotated_image,
            'recommendations': self._get_recommendations(overall_health['score'], disease_prediction)
        }
    
    def _detect_individual_leaves(self, hsv_image):
        """
        Detect individual leaves in the image using color and contour analysis
        """
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate leaves (green areas)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # Find contours (potential leaves)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to get meaningful leaves
        min_area = 1000  # Minimum leaf area
        leaf_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                leaf_regions.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': area
                })
        
        # Sort by area (largest first)
        leaf_regions.sort(key=lambda x: x['area'], reverse=True)
        
        return leaf_regions[:10]  # Limit to top 10 leaves
    
    def _analyze_individual_leaf(self, hsv_image, region, leaf_id):
        """
        Analyze a single leaf region
        """
        x, y, w, h = region['bbox']
        
        # Extract leaf region
        leaf_region = hsv_image[y:y+h, x:x+w]
        
        # Analyze colors in this leaf
        leaf_colors = self._analyze_leaf_colors(leaf_region)
        
        # Detect patterns in this leaf
        leaf_patterns = self._detect_disease_patterns(leaf_region)
        
        # Calculate leaf health score
        leaf_score = self._calculate_leaf_health_score(leaf_colors, leaf_patterns)
        
        return {
            'leaf_id': leaf_id + 1,
            'bbox': region['bbox'],
            'area': region['area'],
            'health_score': leaf_score,
            'status': self._get_health_status(leaf_score),
            'color_analysis': leaf_colors,
            'patterns': leaf_patterns
        }
    
    def _calculate_leaf_health_score(self, color_metrics, patterns):
        """
        Calculate health score for individual leaf (more realistic scoring for actual leaves)
        """
        score = 75  # Start at healthy baseline for verified leaves
        
        # Healthy green color bonus
        green_pct = color_metrics.get('green_healthy', 0)
        if green_pct > 70:
            score += (green_pct - 70) * 0.5  # Bonus for very healthy leaves
        elif green_pct > 50:
            score += (green_pct - 50) * 0.3
        else:
            score -= (50 - green_pct) * 0.4
        
        # Yellow stress penalty (less harsh)
        yellow_pct = color_metrics.get('yellow_stress', 0)
        if yellow_pct > 10:
            score -= yellow_pct * 0.8
        elif yellow_pct > 5:
            score -= yellow_pct * 0.4
        
        # Brown disease penalty (less harsh)
        brown_pct = color_metrics.get('brown_disease', 0)
        if brown_pct > 5:
            score -= brown_pct * 1.5
        elif brown_pct > 2:
            score -= brown_pct * 0.8
        
        # Pattern penalties (much less harsh)
        if patterns.get('spots', 0) > 5:
            score -= patterns['spots'] * 2
        elif patterns.get('spots', 0) > 2:
            score -= patterns['spots'] * 1
        
        if patterns.get('irregular', 0) > 10:
            score -= patterns['irregular'] * 1
        elif patterns.get('irregular', 0) > 5:
            score -= patterns['irregular'] * 0.5
        
        if patterns.get('yellowing', 0) > 3:
            score -= patterns['yellowing'] * 3
        elif patterns.get('yellowing', 0) > 1:
            score -= patterns['yellowing'] * 1
        
        # Ensure minimum score for verified leaves
        score = max(25, min(100, score))
        
        # If we have any green color at all, ensure minimum healthy score
        if green_pct > 20:
            score = max(40, score)
        
        return score
    
    def _calculate_overall_health(self, leaf_analyses):
        """
        Calculate overall plant health from individual leaf analyses
        """
        if not leaf_analyses:
            return {'score': 50, 'status': 'Unknown'}
        
        # Weight by leaf size (larger leaves have more impact)
        total_weight = 0
        weighted_score = 0
        
        for leaf in leaf_analyses:
            weight = leaf['area']
            total_weight += weight
            weighted_score += leaf['health_score'] * weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 50
        
        return {
            'score': overall_score,
            'status': self._get_health_status(overall_score),
            'leaf_count': len(leaf_analyses),
            'healthy_leaves': sum(1 for leaf in leaf_analyses if leaf['health_score'] > 70),
            'stressed_leaves': sum(1 for leaf in leaf_analyses if leaf['health_score'] <= 70)
        }
    
    def _analyze_leaf_colors(self, hsv_image):
        """
        Analyze color distribution in leaf image
        """
        metrics = {}
        
        for color_name, color_range in self.healthy_color_ranges.items():
            mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
            # Fix percentage calculation - divide by total pixels (3 channels)
            total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
            percentage = (np.count_nonzero(mask) / total_pixels) * 100
            metrics[color_name] = percentage
        
        return metrics
    
    def _detect_disease_patterns(self, hsv_image):
        """
        Detect disease patterns in leaf image
        """
        patterns = {}
        
        # Convert to grayscale for pattern detection
        gray = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Detect spots using blob detection
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count different types of contours
        spots = 0
        irregular = 0
        yellowing = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Reasonable threshold for disease spots
                # Check if circular (spot)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:
                        spots += 1
                    else:
                        irregular += 1
        
        # Detect yellowing areas
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellowing = len([c for c in yellow_contours if cv2.contourArea(c) > 100])
        
        patterns['spots'] = spots
        patterns['irregular'] = irregular
        patterns['yellowing'] = yellowing
        
        return patterns
    
    def _detect_plant_type(self, img_array, hsv_image):
        """
        Detect plant type based on leaf characteristics
        Enhanced detection for Apple, Tomato, Pepper, etc.
        """
        # Analyze color distribution
        color_metrics = self._analyze_leaf_colors(hsv_image)
        green_ratio = color_metrics.get('green_healthy', 0)
        
        # Convert to HSV for detailed analysis
        h_channel = hsv_image[:, :, 0]
        s_channel = hsv_image[:, :, 1]
        
        # Calculate color statistics
        mean_hue = np.mean(h_channel)
        mean_saturation = np.mean(s_channel)
        
        # Analyze yellow-green content (for apple leaves)
        yellow_green_pixels = np.sum((h_channel >= 25) & (h_channel <= 45))
        total_pixels = h_channel.size
        yellow_green_ratio = (yellow_green_pixels / total_pixels) * 100  # Convert to percentage
        
        # Enhanced plant type detection for 4 plant types
        if green_ratio > 70 and yellow_green_ratio > 25 and mean_saturation > 60:
            # High green, high yellow-green ratio, high saturation - Apple
            return "Apple"
        elif green_ratio > 70 and mean_saturation > 50:
            # High green, moderate saturation
            if mean_hue > 40 and mean_hue < 60:
                return "Corn"    # Corn leaves have more yellow-green hue
            elif mean_hue < 30:
                return "Potato"  # Potato leaves are broader
            elif mean_hue < 35 and mean_saturation < 55:
                return "Potato"  # Potato leaves have lower saturation
            else:
                return "Tomato"
        elif green_ratio > 50:
            # Moderate green
            if mean_hue > 45 and mean_hue < 65:
                return "Corn"    # Corn has distinct yellow-green hue
            elif mean_hue < 30:
                return "Potato"  # Potato leaves are broader
            else:
                return "Pepper"
        else:
            # Lower green ratio
            if mean_hue > 50 and mean_hue < 70:
                return "Corn"    # Even stressed corn has yellow tint
            else:
                return "General Plant"
    
    def _predict_disease_type(self, color_metrics, patterns, plant_type="Tomato"):
        """
        Predict specific disease type based on color and pattern analysis with plant name
        """
        # Disease database with specific symptoms - balanced thresholds
        disease_symptoms = {
            "Leaf Spot Disease": {
                "brown_threshold": 10,  # Balanced threshold
                "spots_threshold": 1,   # Lower threshold to detect actual spots
                "description": "Circular brown spots on leaves"
            },
            "Early Blight": {
                "brown_threshold": 12,  # Balanced threshold
                "irregular_threshold": 8, # Balanced threshold
                "description": "Target-like brown spots with yellow rings"
            },
            "Powdery Mildew": {
                "white_threshold": 15,
                "description": "White powdery coating on leaf surface"
            },
            "Yellow Leaf Disease": {
                "yellow_threshold": 25,  # Balanced threshold
                "description": "Extensive yellowing of leaf tissue"
            },
            "Bacterial Blight": {
                "brown_threshold": 8,   # Balanced threshold
                "irregular_threshold": 10, # Balanced threshold
                "description": "Irregular brown lesions with yellow halos"
            },
            "Nutrient Deficiency": {
                "yellow_threshold": 15,  # Balanced threshold
                "brown_threshold": 3,   # Balanced threshold
                "description": "Yellowing between leaf veins"
            },
            "Healthy Leaf": {
                "green_threshold": 70,
                "description": "No significant disease symptoms"
            }
        }
        
        brown_pct = color_metrics.get('brown_disease', 0)
        yellow_pct = color_metrics.get('yellow_stress', 0)
        green_pct = color_metrics.get('green_healthy', 0)
        spots = patterns.get('spots', 0)
        irregular = patterns.get('irregular', 0)
        
        # Match symptoms to diseases
        best_match = "Unknown Condition"
        best_score = 0
        
        for disease, symptoms in disease_symptoms.items():
            score = 0
            
            # Check brown symptoms
            if "brown_threshold" in symptoms and brown_pct >= symptoms["brown_threshold"]:
                score += 2
            
            # Check yellow symptoms
            if "yellow_threshold" in symptoms and yellow_pct >= symptoms["yellow_threshold"]:
                score += 2
            
            # Check green symptoms (for healthy)
            if "green_threshold" in symptoms and green_pct >= symptoms["green_threshold"]:
                score += 3
            
            # Check spots
            if "spots_threshold" in symptoms and spots >= symptoms["spots_threshold"]:
                score += 2
            
            # Check irregular patterns
            if "irregular_threshold" in symptoms and irregular >= symptoms["irregular_threshold"]:
                score += 2
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = disease
        
        # If no strong match, provide general assessment
        if best_score < 2:
            if green_pct > 75 and spots == 0 and irregular == 0 and yellow_pct < 5:
                best_match = "Healthy Leaf"  # Explicit healthy check
            elif brown_pct > 5:
                best_match = "Leaf Damage"
            elif yellow_pct > 10:
                best_match = "Leaf Stress"
            else:
                best_match = "Mild Symptoms"
        
        # Add plant name to disease prediction
        if best_match == "Healthy Leaf":
            return f"{plant_type} - Healthy"
        elif best_match == "Unknown Condition":
            return f"{plant_type} - General Condition"
        else:
            return f"{plant_type} {best_match}"
    
    def _get_health_status(self, score):
        """
        Get health status based on score (adjusted for new scoring system)
        """
        if score >= 80:
            return "Healthy"
        elif score >= 65:
            return "Mild Stress"
        elif score >= 45:
            return "Moderate Issues"
        else:
            return "Severe Problems"
    
    def _get_recommendations(self, health_score, disease_prediction):
        """
        Get care recommendations based on analysis
        """
        recommendations = []
        
        # Check for healthy status first
        if "Healthy" in disease_prediction:
            recommendations.extend([
                "Continue regular monitoring",
                "Maintain proper care routine",
                "Preventative measures recommended",
                "Ensure optimal growing conditions",
                "Monitor for pests and diseases"
            ])
        # Check for specific disease types (plant-agnostic)
        elif "Leaf Spot" in disease_prediction or "Spot" in disease_prediction:
            recommendations.extend([
                "Apply fungicide treatment",
                "Remove affected leaves",
                "Improve air circulation",
                "Avoid overhead watering",
                "Ensure proper spacing between plants"
            ])
        elif "Yellow" in disease_prediction or "Yellowing" in disease_prediction:
            recommendations.extend([
                "Check soil nutrients",
                "Apply balanced fertilizer",
                "Ensure proper watering",
                "Test soil pH levels",
                "Consider nutrient deficiency treatment"
            ])
        elif "Blight" in disease_prediction:
            recommendations.extend([
                "Apply systemic fungicide",
                "Remove infected plant parts",
                "Disinfect tools",
                "Improve air circulation",
                "Avoid wetting foliage"
            ])
        elif "Bacterial" in disease_prediction:
            recommendations.extend([
                "Apply copper-based fungicide",
                "Remove infected plant parts",
                "Avoid working with wet plants",
                "Ensure proper drainage",
                "Practice crop rotation"
            ])
        elif "Scab" in disease_prediction:
            recommendations.extend([
                "Apply fungicide spray",
                "Remove fallen leaves",
                "Prune affected branches",
                "Ensure good air circulation",
                "Apply dormant season treatment"
            ])
        elif "Rust" in disease_prediction:
            recommendations.extend([
                "Apply rust-specific fungicide",
                "Remove infected leaves",
                "Reduce humidity",
                "Improve air circulation",
                "Avoid overhead irrigation"
            ])
        elif "Mildew" in disease_prediction:
            recommendations.extend([
                "Apply sulfur-based fungicide",
                "Improve air circulation",
                "Reduce humidity",
                "Remove affected plant parts",
                "Ensure proper spacing"
            ])
        elif "Wilt" in disease_prediction:
            recommendations.extend([
                "Remove infected plants",
                "Control insect vectors",
                "Apply soil fumigation",
                "Practice crop rotation",
                "Use resistant varieties"
            ])
        elif "Mild Symptoms" in disease_prediction:
            recommendations.extend([
                "Monitor for changes",
                "Check environmental conditions",
                "Consider plant inspection",
                "Review watering and fertilization",
                "Look for early disease signs"
            ])
        else:
            recommendations.extend([
                "Monitor plant health regularly",
                "Check for pest activity",
                "Ensure proper growing conditions",
                "Consider professional consultation",
                "Review care practices"
            ])
        
        return recommendations
    
    def _create_leaf_visualization(self, image, leaf_analyses):
        """
        Create visualization showing detected leaves with health status (no text labels)
        """
        result_image = image.copy()
        
        for leaf in leaf_analyses:
            x, y, w, h = leaf['bbox']
            health_score = leaf['health_score']
            status = leaf['status']
            
            # Color based on health status
            if status == 'Healthy':
                color = (0, 255, 0)  # Green
            elif status == 'Mild Stress':
                color = (255, 255, 0)  # Yellow
            elif status == 'Moderate Issues':
                color = (255, 165, 0)  # Orange
            else:
                color = (255, 0, 0)  # Red
            
            # Draw bounding box only (no text labels)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        
        return result_image

def analyze_leaf_health(image):
    """
    Main function for leaf health analysis
    """
    analyzer = LeafHealthAnalyzer()
    results = analyzer.analyze_leaf_image(image)
    
    return results
