#!/usr/bin/env python3
"""
Python wrapper script for Express.js integration.
This script handles command-line arguments and returns JSON output.
"""

import sys
import argparse
from fashion_color_extractor import FashionColorExtractor, extract_color_for_api

def main():
    parser = argparse.ArgumentParser(description='Extract dominant colors from fashion images')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--keep-background', action='store_true', 
                       help='Keep background instead of removing it')
    parser.add_argument('--colors', type=int, default=3,
                       help='Number of dominant colors to extract (default: 3)')
    parser.add_argument('--output-format', choices=['json', 'simple'], default='json',
                       help='Output format (default: json)')
    
    args = parser.parse_args()
    
    try:
        # Determine if we should remove background
        remove_bg = not args.keep_background
        
        if args.output_format == 'json':
            # Use the API-ready function
            result = extract_color_for_api(args.image_path, remove_bg)
            print(result)
        else:
            # Simple output for debugging
            extractor = FashionColorExtractor()
            result = extractor.analyze_image(args.image_path, remove_bg)
            
            if result['success']:
                main_color = result['dominant_color']
                print(f"Dominant Color: {main_color['name']} ({main_color['hex']})")
                for i, color in enumerate(result['color_palette'][:3]):
                    print(f"  {i+1}. {color['name']} - {color['hex']}")
            else:
                print(f"Error: {result['error']}")
                sys.exit(1)
    
    except FileNotFoundError:
        error_result = {
            'success': False,
            'error': f'Image file not found: {args.image_path}',
            'dominant_color': None,
            'color_palette': []
        }
        if args.output_format == 'json':
            import json
            print(json.dumps(error_result))
        else:
            print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'dominant_color': None,
            'color_palette': []
        }
        if args.output_format == 'json':
            import json
            print(json.dumps(error_result))
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()