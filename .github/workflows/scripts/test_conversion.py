import os
import sys
import argparse
from flux_controlnet_integration import DickiesSketchConverter

def main():
    parser = argparse.ArgumentParser(description='Test sketch to photorealistic conversion')
    parser.add_argument('--sketch', type=str, required=True, help='Path to sketch image')
    parser.add_argument('--prompt', type=str, default="workwear jacket with pockets, high quality stitching", 
                        help='Text prompt to guide image generation')
    parser.add_argument('--use_replicate', action='store_true', help='Use Replicate API (requires API token)')
    args = parser.parse_args()
    
    if not os.path.exists(args.sketch) and not args.sketch.startswith(('http://', 'https://')):
        print(f"Error: Sketch file {args.sketch} not found")
        sys.exit(1)
    
    converter = DickiesSketchConverter()
    result_path = converter.convert_sketch_to_realistic(
        args.prompt, 
        args.sketch, 
        use_replicate=args.use_replicate
    )
    
    if result_path:
        print(f"Successfully generated photorealistic image at: {result_path}")
    else:
        print("Failed to generate photorealistic image")
        sys.exit(1)

if __name__ == "__main__":
    main()
