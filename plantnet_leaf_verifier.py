import requests
import json
import os
from PIL import Image
import io
import base64

class PlantNetLeafVerifier:
    def __init__(self):
        self.api_key = os.getenv("PLANTNET_API_KEY")
        if not self.api_key:
            raise ValueError("PLANTNET_API_KEY not found in environment variables")
        self.api_url = "https://api.plantnet.org/v2/identify"
    
    def verify_leaf_with_plantnet(self, image, language="en"):
        """
        Verify if the uploaded image is actually a leaf using PlantNet API
        """
        # Try PlantNet API first
        api_result = self._try_plantnet_api(image, language)
        
        # If API succeeds, return the result
        if api_result['success']:
            return api_result
        else:
            # If API fails, use simple fallback
            print("DEBUG: PlantNet API failed, using fallback detection")
            return self._simple_fallback_detection(image)
    
    def _try_plantnet_api(self, image, language="en"):
        """
        Try to identify the plant species using PlantNet API
        """
        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Prepare the API request
            files = {'images': ('leaf.jpg', img_byte_arr, 'image/jpeg')}
            
            # Basic organ data - we're specifically looking for leaves
            data = {
                'organs': 'leaf',
                'lang': language,
                'include-related-images': 'false'
            }
            
            headers = {
                'Api-Key': self.api_key
            }
            
            # Make the API call
            response = requests.post(self.api_url, files=files, data=data, headers=headers)
            
            print(f"DEBUG: PlantNet API Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"DEBUG: PlantNet Response Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    if 'results' in result and len(result['results']) > 0:
                        # Get the top result
                        top_result = result['results'][0]
                        confidence = top_result.get('score', 0)
                        species = top_result.get('scientificNameWithoutAuthor', 'Unknown')
                        common_name = top_result.get('commonNames', ['Unknown'])[0] if top_result.get('commonNames') else 'Unknown'
                        
                        print(f"DEBUG: PlantNet confidence = {confidence:.3f} ({confidence:.1%})")
                        print(f"DEBUG: PlantNet species = {species}")
                        print(f"DEBUG: PlantNet common name = {common_name}")
                        
                        # Only accept if it's clearly a plant species with good confidence
                        if confidence > 0.2 and species != 'Unknown' and self._is_valid_plant_species(species, common_name):
                            return {
                                'success': True,
                                'confidence': confidence,
                                'species': species,
                                'common_name': common_name,
                                'method': 'plantnet_api',
                                'message': f"Leaf detected with {confidence:.1%} confidence. Identified as {common_name}."
                            }
                        else:
                            return {
                                'success': False,
                                'confidence': confidence,
                                'method': 'plantnet_api',
                                'message': "upload the leaf images only"
                            }
                    else:
                        return {
                            'success': False,
                            'method': 'plantnet_api',
                            'message': "upload the leaf images only"
                        }
                except Exception as e:
                    print(f"DEBUG: PlantNet JSON Parsing Error: {e}")
                    return {
                        'success': False,
                        'method': 'plantnet_api',
                        'message': "upload the leaf images only"
                    }
            else:
                print(f"DEBUG: PlantNet API Error: {response.status_code}")
                return {
                    'success': False,
                    'method': 'plantnet_api',
                    'message': "upload the leaf images only"
                }
                
        except Exception as e:
            print(f"DEBUG: PlantNet API Exception: {e}")
            return {
                'success': False,
                'method': 'plantnet_api',
                'message': "upload the leaf images only"
            }
    
    def _is_valid_plant_species(self, species, common_name):
        """
        Validate that the detected species is actually a plant
        """
        try:
            # Convert to lowercase for comparison
            species_lower = species.lower() if species else ""
            common_name_lower = common_name.lower() if common_name else ""
            
            # Keywords that indicate non-plant objects
            non_plant_keywords = [
                'human', 'person', 'people', 'face', 'man', 'woman', 'child',
                'card', 'document', 'paper', 'text', 'photo', 'image',
                'object', 'item', 'product', 'device', 'machine',
                'animal', 'dog', 'cat', 'bird', 'insect', 'bug',
                'building', 'house', 'car', 'vehicle', 'road', 'street',
                'clothing', 'shirt', 'pants', 'dress', 'shoe'
            ]
            
            # Check if any non-plant keywords are in the results
            for keyword in non_plant_keywords:
                if keyword in species_lower or keyword in common_name_lower:
                    print(f"DEBUG: Non-plant keyword detected: {keyword}")
                    return False
            
            # Check for valid plant indicators
            plant_indicators = [
                'plant', 'leaf', 'tree', 'flower', 'herb', 'grass', 'shrub',
                'cactus', 'succulent', 'fern', 'moss', 'vine', 'weed',
                'crop', 'agriculture', 'garden', 'botanical', 'flora'
            ]
            
            # If we find plant indicators, it's likely a plant
            for indicator in plant_indicators:
                if indicator in species_lower or indicator in common_name_lower:
                    print(f"DEBUG: Plant indicator detected: {indicator}")
                    return True
            
            # Check for common plant families/genus (these indicate real plants)
            plant_families = [
                'acer', 'betula', 'quercus', 'pinus', 'rosa', 'citrus',
                'solanum', 'malus', 'prunus', 'vitis', 'zea', 'oryza',
                'triticum', 'helianthus', 'brassica', 'allium', 'daucus'
            ]
            
            for family in plant_families:
                if family in species_lower:
                    print(f"DEBUG: Plant family detected: {family}")
                    return True
            
            # If species name looks scientific (contains spaces, capital letters, etc.)
            # and doesn't contain non-plant keywords, assume it's a plant
            if ' ' in species and len(species.split()) >= 2:
                # Scientific names typically have genus and species
                print(f"DEBUG: Scientific name format detected")
                return True
            
            # Default to False if we can't determine
            print(f"DEBUG: Unable to validate as plant species")
            return False
            
        except Exception as e:
            print(f"DEBUG: Species validation error: {e}")
            return False
    
    def _simple_fallback_detection(self, image):
        """
        Simple fallback detection when PlantNet API fails
        """
        try:
            import cv2
            import numpy as np
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and OpenCV format
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for leaves (including diseased colors)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            lower_yellow = np.array([15, 40, 40])
            upper_yellow = np.array([35, 255, 255])
            lower_red = np.array([0, 40, 40])
            upper_red = np.array([15, 255, 255])
            lower_brown = np.array([8, 40, 40])
            upper_brown = np.array([25, 255, 255])
            
            # Create masks for all leaf colors
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Combine all leaf color masks
            combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, red_mask)
            combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
            
            # Calculate leaf pixel ratio
            leaf_pixels = cv2.countNonZero(combined_mask)
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            leaf_ratio = leaf_pixels / total_pixels
            
            print(f"DEBUG: Fallback - leaf ratio = {leaf_ratio:.3f} ({leaf_ratio:.1%})")
            
            # Check for text in the image
            has_text = self._detect_text_simple(img_cv)
            print(f"DEBUG: Fallback - has text = {has_text}")
            
            # Reject if there's text (document detected)
            if has_text:
                return {
                    'success': False,
                    'method': 'fallback',
                    'message': "upload the leaf images only"
                }
            
            # Simple detection: if we have reasonable leaf color and no text, accept it
            if leaf_ratio > 0.03:  # Only 3% needed for fallback
                confidence = min(leaf_ratio * 100, 80)
                return {
                    'success': True,
                    'confidence': confidence / 100,
                    'method': 'fallback',
                    'message': f"Leaf detected with {confidence:.1f}% confidence using fallback detection."
                }
            else:
                return {
                    'success': False,
                    'method': 'fallback',
                    'message': "upload the leaf images only"
                }
                
        except Exception as e:
            print(f"DEBUG: Fallback detection error: {e}")
            return {
                'success': False,
                'message': "upload the leaf images only"
            }
    
    def _detect_text_simple(self, img_cv):
        """
        Simple text detection for fallback method
        """
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Use morphological operations to detect text-like patterns
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text typically has specific aspect ratios and sizes
                if 2 < aspect_ratio < 15 and 30 < area < 10000:
                    text_like_contours += 1
            
            # If we find many text-like contours, it's likely a document
            has_text = text_like_contours > 3
            print(f"DEBUG: Text-like contours found: {text_like_contours}")
            return has_text
            
        except Exception as e:
            print(f"DEBUG: Simple text detection error: {e}")
            return False
    
    def _shape_based_leaf_detection(self, image):
        """
        Simple shape-based leaf detection using computer vision
        """
        try:
            import cv2
            import numpy as np
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and OpenCV format
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for leaves (including diseased colors)
            # Green healthy leaves
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # Yellow/brown diseased leaves
            lower_yellow = np.array([15, 40, 40])
            upper_yellow = np.array([35, 255, 255])
            
            # Red/brown diseased leaves  
            lower_red = np.array([0, 40, 40])
            upper_red = np.array([15, 255, 255])
            
            # Brown/dark diseased leaves
            lower_brown = np.array([8, 40, 40])
            upper_brown = np.array([25, 255, 255])
            
            # Create masks for all leaf colors
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Combine all leaf color masks
            combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, red_mask)
            combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
            
            # Calculate leaf pixel ratio (all colors)
            leaf_pixels = cv2.countNonZero(combined_mask)
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            leaf_ratio = leaf_pixels / total_pixels
            
            print(f"DEBUG: Leaf pixel ratio = {leaf_ratio:.3f} ({leaf_ratio:.1%})")
            
            # Convert to grayscale for shape analysis
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (leaf-like objects) - more strict
            min_area = 2000  # Higher minimum area to reject small objects
            max_area = total_pixels * 0.8  # Lower maximum area to reject background
            
            leaf_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Check aspect ratio (more strict for leaf detection)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Stricter aspect ratio for leaves (0.4 to 3)
                    if 0.4 < aspect_ratio < 3:
                        # Additional check: contour should be reasonably compact
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            compactness = 4 * np.pi * area / (perimeter * perimeter)
                            # Compactness should be reasonable for leaves (not too irregular)
                            if compactness > 0.1:
                                leaf_contours.append(contour)
            
            print(f"DEBUG: Found {len(leaf_contours)} leaf-like contours")
            
            # Check for text and unnatural shapes
            has_text = self._detect_text(gray)
            has_unnatural_shapes = self._detect_unnatural_shapes(leaf_contours)
            
            print(f"DEBUG: Has text = {has_text}")
            print(f"DEBUG: Has unnatural shapes = {has_unnatural_shapes}")
            
            # Determine if it's a leaf based on simple criteria
            is_leaf = False
            confidence = 0.0
            
            # Reject if there's text or unnatural shapes
            if has_text or has_unnatural_shapes:
                is_leaf = False
                confidence = 0.0
            # Accept if there are basic leaf colors
            elif leaf_ratio > 0.05:
                is_leaf = True
                confidence = min(leaf_ratio * 100, 90)
            else:
                is_leaf = False
                confidence = 0.0
            
            print(f"DEBUG: Leaf detection result = {is_leaf}, confidence = {confidence:.1f}%")
            
            if is_leaf:
                return {
                    'success': True,
                    'confidence': confidence / 100,
                    'method': 'shape_based',
                    'message': f"Leaf detected with {confidence:.1f}% confidence using shape analysis."
                }
            else:
                return {
                    'success': False,
                    'confidence': confidence / 100,
                    'method': 'shape_based',
                    'message': "upload the leaf images only"
                }
                
        except Exception as e:
            print(f"DEBUG: Shape detection error: {e}")
            return {
                'success': False,
                'message': "upload the leaf images only"
            }
    
    def _detect_text(self, gray):
        """
        Detect if there's text in the image (indicating document/ID card)
        """
        try:
            import cv2
            import numpy as np
            
            # Use morphological operations to detect text-like patterns
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find text-like regions
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Count contours that look like text
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text typically has specific aspect ratios and sizes
                if 2 < aspect_ratio < 10 and 20 < area < 5000:
                    text_like_contours += 1
            
            # If we find many text-like contours, it's likely a document
            has_text = text_like_contours > 5
            return has_text
            
        except Exception as e:
            print(f"DEBUG: Text detection error: {e}")
            return False
    
    def _detect_unnatural_shapes(self, contours):
        """
        Detect if there are unnatural geometric shapes (rectangles, perfect circles)
        """
        try:
            import cv2
            import numpy as np
            
            unnatural_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Check for perfect rectangles
                    x, y, w, h = cv2.boundingRect(contour)
                    rect_area = w * h
                    rect_similarity = area / rect_area if rect_area > 0 else 0
                    
                    # Check for perfect circles
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # If shape is very rectangular or very circular, it's unnatural
                    if rect_similarity > 0.9 or circularity > 0.85:
                        unnatural_count += 1
            
            # If we find unnatural shapes, it's likely not a leaf
            has_unnatural = unnatural_count > 0
            return has_unnatural
            
        except Exception as e:
            print(f"DEBUG: Unnatural shape detection error: {e}")
            return False
    
    def _fallback_leaf_verification(self, image):
        """
        Basic fallback verification if PlantNet API fails
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic color analysis for leaf detection
            import numpy as np
            
            img_array = np.array(image)
            
            # Check for green color presence (typical for leaves)
            green_pixels = np.sum((img_array[:, :, 1] > img_array[:, :, 0]) & 
                                 (img_array[:, :, 1] > img_array[:, :, 2]))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            green_ratio = green_pixels / total_pixels
            
            # Basic shape and texture checks
            gray = np.mean(img_array, axis=2)
            
            # Check if image has reasonable variance (not uniform)
            variance = np.var(gray)
            
            # If we have enough green color and reasonable variance, it's likely a leaf (more lenient)
            if green_ratio > 0.08 and variance > 50:
                return {
                    'success': True,
                    'confidence': min(green_ratio * 100, 80),  # Cap at 80%
                    'method': 'fallback',
                    'message': f"Leaf characteristics detected (green ratio: {green_ratio:.1%}). Proceeding with analysis."
                }
            else:
                # For debugging - show actual values
                print(f"DEBUG: Fallback - green_ratio = {green_ratio:.3f}, variance = {variance:.1f}")
                return {
                    'success': False,
                    'method': 'fallback',
                    'message': "upload the leaf images only"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': "upload the leaf images only"
            }
