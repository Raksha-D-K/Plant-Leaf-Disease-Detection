import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class LeafHealthAnalyzer:
    def __init__(self):
        self.healthy_color_ranges = {
            'green_healthy': {
                'lower': np.array([25, 40, 40]),
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
        
        # Determine overall disease type
        disease_prediction = self._predict_disease_type(overall_color_metrics, overall_patterns)
        
        # Create visualization with detected leaves
        annotated_image = self._create_leaf_visualization(img_array, leaf_analyses)
        
        return {
            'health_score': overall_health['score'],
            'status': overall_health['status'],
            'disease_prediction': disease_prediction,
            'color_analysis': overall_color_metrics,
            'patterns': overall_patterns,
            'leaf_count': len(leaf_regions),
            'leaf_analyses': leaf_analyses,
            'annotated_image': annotated_image,
            'recommendations': self._get_recommendations(overall_health['score'], disease_prediction)
        }
    
    def _create_leaf_visualization(self, image, leaf_analyses):
        """
        Create visualization showing detected leaves with health status
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
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"Leaf {leaf['leaf_id']}: {health_score:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                        (x + label_size[0], y), color, -1)
            cv2.putText(result_image, label, (x, y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
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
        Calculate health score for individual leaf (more realistic scoring)
        """
        score = 50  # Start at neutral
        
        # Healthy green color bonus
        green_pct = color_metrics.get('green_healthy', 0)
        if green_pct > 60:
            score += (green_pct - 60) * 0.5
        else:
            score -= (60 - green_pct) * 0.8
        
        # Yellow stress penalty
        yellow_pct = color_metrics.get('yellow_stress', 0)
        if yellow_pct > 5:
            score -= yellow_pct * 1.2
        
        # Brown disease penalty
        brown_pct = color_metrics.get('brown_disease', 0)
        if brown_pct > 2:
            score -= brown_pct * 2.5
        
        # Pattern penalties
        if patterns.get('spots', 0) > 2:
            score -= patterns['spots'] * 4
        
        if patterns.get('irregular', 0) > 5:
            score -= patterns['irregular'] * 2
        
        if patterns.get('yellowing', 0) > 1:
            score -= patterns['yellowing'] * 6
        
        return max(0, min(100, score))
    
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
            percentage = np.count_nonzero(mask) / (hsv_image.shape[0] * hsv_image.shape[1]) * 100
            metrics[color_name] = percentage
        
        return metrics
    
    def _detect_disease_patterns(self, hsv_image):
        """
        Detect common disease patterns
        """
        patterns = {}
        
        # Detect spots (circular patterns)
        gray = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Find circular patterns (spots)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        if circles is not None:
            patterns['spots'] = len(circles[0])
        else:
            patterns['spots'] = 0
        
        # Detect irregular patterns (blight, rust)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        irregular_contours = [c for c in contours if cv2.contourArea(c) > 100]
        patterns['irregular'] = len(irregular_contours)
        
        # Detect yellowing patterns
        yellow_mask = cv2.inRange(hsv_image, self.healthy_color_ranges['yellow_stress']['lower'], 
                                self.healthy_color_ranges['yellow_stress']['upper'])
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patterns['yellowing'] = len([c for c in yellow_contours if cv2.contourArea(c) > 50])
        
        return patterns
    
    def _calculate_health_score(self, color_metrics, patterns):
        """
        Calculate overall leaf health score (0-100)
        """
        # Start with base score
        score = 100
        
        # Penalize for unhealthy colors
        if color_metrics.get('yellow_stress', 0) > 10:
            score -= color_metrics['yellow_stress'] * 0.5
        
        if color_metrics.get('brown_disease', 0) > 5:
            score -= color_metrics['brown_disease'] * 2
        
        # Penalize for disease patterns
        if patterns.get('spots', 0) > 5:
            score -= patterns['spots'] * 3
        
        if patterns.get('irregular', 0) > 10:
            score -= patterns['irregular'] * 1.5
        
        if patterns.get('yellowing', 0) > 3:
            score -= patterns['yellowing'] * 4
        
        # Bonus for healthy green color
        if color_metrics.get('green_healthy', 0) > 60:
            score += 10
        
    return max(0, min(100, score))
    
def _predict_disease_type(self, color_metrics, patterns):
        """
        Predict specific disease type based on color and pattern analysis
        """
        # Disease database with specific symptoms
        disease_symptoms = {
            "Leaf Spot Disease": {
                "brown_threshold": 8,
                "spots_threshold": 3,
                "description": "Circular brown spots on leaves"
            },
            "Early Blight": {
                "brown_threshold": 12,
                "irregular_threshold": 8,
                "description": "Target-like brown spots with yellow rings"
            },
            "Powdery Mildew": {
                "white_threshold": 15,
                "description": "White powdery coating on leaf surface"
            },
            "Yellow Leaf Disease": {
                "yellow_threshold": 25,
                "description": "Extensive yellowing of leaf tissue"
            },
            "Bacterial Blight": {
                "brown_threshold": 6,
                "irregular_threshold": 12,
                "description": "Irregular brown lesions with yellow halos"
            },
            "Nutrient Deficiency": {
                "yellow_threshold": 15,
                "brown_threshold": 3,
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
            if brown_pct > 5:
                best_match = "Leaf Damage"
            elif yellow_pct > 10:
                best_match = "Leaf Stress"
            else:
                best_match = "Mild Symptoms"
        
        return best_match
    
    def _get_recommendations(self, health_score, disease_prediction):
        """
        Get care recommendations based on analysis
        """
        recommendations = []
        
        if disease_prediction == "Leaf Spot Disease":
            recommendations.extend([
                "Apply fungicide treatment",
                "Remove affected leaves",
                "Improve air circulation"
            ])
        elif disease_prediction == "Yellow Leaf Disease":
            recommendations.extend([
                "Check soil nutrients",
                "Apply balanced fertilizer",
                "Ensure proper watering"
            ])
        elif disease_prediction == "Early Blight":
            recommendations.extend([
                "Apply systemic fungicide",
                "Remove infected plant parts",
                "Disinfect tools"
            ])
        elif disease_prediction == "Healthy Leaf":
            recommendations.extend([
                "Continue regular monitoring",
                "Maintain proper care routine",
                "Preventative measures recommended"
            ])
        else:
            recommendations.extend([
                "Monitor for changes",
                "Check environmental conditions",
                "Consider plant inspection"
            ])
        
        return recommendations

def analyze_leaf_health(image):
    """
    Main function for leaf health analysis
    """
    analyzer = LeafHealthAnalyzer()
    results = analyzer.analyze_leaf_image(image)
    
    return results
