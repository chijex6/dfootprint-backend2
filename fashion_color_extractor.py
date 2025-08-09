from PIL import Image, ImageFilter
import numpy as np
import webcolors
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import json
import requests
import time
from typing import Dict, List, Tuple, Optional

class FashionColorExtractor:
    def __init__(self):
        # Extended fashion color names for better naming
        self.fashion_colors = {
            # Blacks & Grays
            (0, 0, 0): "Jet Black",
            (36, 36, 36): "Charcoal",
            (64, 64, 64): "Graphite",
            (128, 128, 128): "Steel Gray",
            (169, 169, 169): "Silver",
            (211, 211, 211): "Platinum",
            
            # Whites & Creams
            (255, 255, 255): "Pure White",
            (255, 250, 240): "Ivory",
            (250, 235, 215): "Cream",
            (245, 245, 220): "Beige",
            (240, 248, 255): "Ghost White",
            
            # Blues
            (0, 0, 139): "Midnight Blue",
            (25, 25, 112): "Navy",
            (70, 130, 180): "Steel Blue",
            (100, 149, 237): "Cornflower Blue",
            (135, 206, 235): "Sky Blue",
            (173, 216, 230): "Powder Blue",
            (0, 191, 255): "Electric Blue",
            
            # Reds
            (139, 0, 0): "Crimson",
            (178, 34, 34): "Fire Brick",
            (220, 20, 60): "Cherry Red",
            (255, 0, 0): "True Red",
            (255, 20, 147): "Hot Pink",
            (255, 192, 203): "Blush Pink",
            (250, 128, 114): "Coral",
            
            # Greens
            (0, 100, 0): "Forest Green",
            (34, 139, 34): "Emerald",
            (50, 205, 50): "Lime Green",
            (144, 238, 144): "Mint Green",
            (152, 251, 152): "Sage",
            (128, 128, 0): "Olive",
            
            # Browns
            (139, 69, 19): "Chocolate",
            (160, 82, 45): "Cognac",
            (210, 180, 140): "Tan",
            (222, 184, 135): "Camel",
            (245, 222, 179): "Nude",
            (139, 90, 43): "Espresso",
            
            # Purples
            (75, 0, 130): "Royal Purple",
            (138, 43, 226): "Amethyst",
            (147, 112, 219): "Lilac",
            (221, 160, 221): "Lavender",
            (218, 112, 214): "Orchid",
            
            # Yellows
            (255, 215, 0): "Golden",
            (255, 255, 0): "Canary Yellow",
            (255, 250, 205): "Lemon",
            (240, 230, 140): "Champagne",
            (255, 228, 181): "Vanilla",
            
            # Fashion-specific colors
            (95, 158, 160): "Teal",
            (255, 127, 80): "Tangerine",
            (255, 105, 180): "Fuchsia",
            (32, 178, 170): "Turquoise",
            (255, 20, 147): "Magenta",
            (255, 69, 0): "Vermillion",
        }
        
        # Color name API configuration
        self.api_base_url = "https://www.thecolorapi.com/id"
        self.max_retries = 2
        self.retry_delay = 0.5
    
    def get_color_name_from_api(self, hex_color: str) -> Optional[str]:
        """Get color name from TheColorAPI.com"""
        try:
            hex_clean = hex_color.lstrip('#')
            url = f"{self.api_base_url}?hex={hex_clean}"
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(url, timeout=3)
                    response.raise_for_status()
                    
                    data = response.json()
                    if data and 'name' in data and 'value' in data['name']:
                        return data['name']['value']
                    
                except requests.RequestException:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
                    
        except Exception:
            pass
        
        return None
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background using edge detection and color clustering.
        Assumes the main subject (fashion item) is in the center.
        """
        # Convert to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create mask for the center region (likely the fashion item)
        height, width = img_cv.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create elliptical mask favoring center
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x) / (width * 0.3)) ** 2 + ((y - center_y) / (height * 0.3)) ** 2 <= 1
        
        # Apply Gaussian blur to soften edges
        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        
        # Use grabCut algorithm for better segmentation
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Create rectangle around center area
        rect = (width//4, height//4, width//2, height//2)
        mask_gc = np.zeros((height, width), np.uint8)
        
        try:
            cv2.grabCut(img_cv, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask_gc = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
            
            # Combine masks
            final_mask = mask & mask_gc.astype(bool)
        except:
            # Fallback to simple center mask
            final_mask = mask
        
        # Apply mask
        result = img_array.copy()
        result[~final_mask] = [255, 255, 255]  # Set background to white
        
        return Image.fromarray(result)
    
    def get_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using K-means clustering.
        """
        # Resize for faster processing
        image = image.resize((150, 150))
        
        # Convert to numpy array and reshape
        data = np.array(image).reshape(-1, 3)
        
        # Remove white/near-white pixels (likely background)
        non_white_mask = np.sum(data, axis=1) < 700  # Adjust threshold as needed
        data = data[non_white_mask]
        
        if len(data) == 0:
            return [(128, 128, 128)]  # Return gray if no non-white pixels
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=min(n_colors, len(data)), random_state=42, n_init=10)
        kmeans.fit(data)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get color frequencies
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        # Sort colors by frequency
        sorted_colors = []
        for i in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
            # Convert numpy integers to regular Python integers
            color_tuple = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            sorted_colors.append(color_tuple)
        
        return sorted_colors
    
    def closest_fashion_color(self, rgb: Tuple[int, int, int]) -> str:
        """
        Find closest fashion color name - NOW WITH API ENHANCEMENT!
        """
        # First try the API for better naming
        hex_color = webcolors.rgb_to_hex(rgb)
        api_name = self.get_color_name_from_api(hex_color)
        if api_name:
            return api_name
        
        # Fallback to original logic
        min_distance = float('inf')
        closest_name = None
        
        # Check custom fashion colors first
        for fashion_rgb, name in self.fashion_colors.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb, fashion_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        # Fallback to CSS3 colors if needed
        try:
            css_name = self.closest_css3_colour(rgb)
            if min_distance > 10000:  # If fashion color match is poor
                return css_name.replace('_', ' ').title()
        except:
            pass
        
        return closest_name or "Unknown"
    
    def closest_css3_colour(self, requested_colour: Tuple[int, int, int]) -> str:
        """Find the closest named CSS3 colour for an RGB value."""
        min_distance = None
        closest_name = None
        
        for name in webcolors.CSS3_HEX_TO_NAMES.values():
            try:
                r_c, g_c, b_c = webcolors.name_to_rgb(name)
                distance = (r_c - requested_colour[0]) ** 2 + \
                          (g_c - requested_colour[1]) ** 2 + \
                          (b_c - requested_colour[2]) ** 2
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_name = name
            except:
                continue
        
        return closest_name or "unknown"
    
    def analyze_image(self, image_path: str, remove_bg: bool = True) -> Dict:
        """
        Main function to analyze image and extract color information.
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Remove background if requested
            if remove_bg:
                image = self.remove_background(image)
            
            # Get dominant colors
            dominant_colors = self.get_dominant_colors(image, n_colors=3)
            
            # Analyze each dominant color
            color_analysis = []
            for i, rgb in enumerate(dominant_colors):
                # Ensure RGB values are regular Python integers
                rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                
                hex_color = webcolors.rgb_to_hex(rgb)
                fashion_name = self.closest_fashion_color(rgb)
                
                color_info = {
                    'rank': i + 1,
                    'rgb': rgb,  # Now guaranteed to be regular Python ints
                    'hex': hex_color,
                    'name': fashion_name
                }
                color_analysis.append(color_info)
            
            # Main result (most dominant color)
            main_color = color_analysis[0]
            
            return {
                'success': True,
                'dominant_color': main_color,
                'color_palette': color_analysis,
                'image_processed': 'background_removed' if remove_bg else 'original'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dominant_color': None,
                'color_palette': []
            }

# Custom JSON encoder to handle numpy types (backup solution)
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Express.js integration function
def extract_color_for_api(image_path: str, remove_background: bool = True) -> str:
    """
    Function specifically designed for Express.js API integration.
    Returns JSON string for easy parsing in Node.js.
    """
    extractor = FashionColorExtractor()
    result = extractor.analyze_image(image_path, remove_background)
    # Use custom encoder as backup to handle any remaining numpy types
    return json.dumps(result, indent=2, cls=NumpyJSONEncoder)

# Command line usage
if __name__ == "__main__":
    import sys
    
    # Default image path or get from command line
    image_path = sys.argv[1] if len(sys.argv) > 1 else "22.jpg"
    
    extractor = FashionColorExtractor()
    result = extractor.analyze_image(image_path, remove_bg=True)
    
    if result['success']:
        print(f"✅ Image Analysis Complete!")
        print(f"📸 Processing: {result['image_processed']}")
        print(f"\n🎨 Dominant Color:")
        main = result['dominant_color']
        print(f"   • Name: {main['name']}")
        print(f"   • RGB: {main['rgb']}")
        print(f"   • HEX: {main['hex']}")
        
        print(f"\n🌈 Color Palette:")
        for color in result['color_palette']:
            print(f"   {color['rank']}. {color['name']} - {color['hex']}")
    else:
        print(f"❌ Error: {result['error']}")