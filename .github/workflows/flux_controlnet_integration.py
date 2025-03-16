import os
import sys
import torch
import replicate
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Configuration
REPLICATE_API_TOKEN = "YOUR_REPLICATE_API_TOKEN"  # Replace with your token
DICKIES_MODEL_VERSION = "dkk222/ai_test:1551cfdfd757021cabae03b09a4d44ed10f86c3929e530d0b0763daa88be6738"
OUTPUT_DIR = "output_images"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up Replicate API
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
replicate_client = replicate.Client()

class DickiesSketchConverter:
    def __init__(self):
        # Load the ControlNet for line art
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=torch.float16
        )
        
        # We'll use a base SD model and later incorporate results from your fine-tuned model
        self.base_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )
        
        # Improve inference speed and quality
        self.base_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.base_pipeline.scheduler.config
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.base_pipeline = self.base_pipeline.to("cuda")
            
    def preprocess_sketch(self, sketch_path):
        """Load and prepare the sketch for ControlNet"""
        if sketch_path.startswith(('http://', 'https://')):
            response = requests.get(sketch_path)
            sketch_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            sketch_image = Image.open(sketch_path).convert("RGB")
        
        # Resize if needed (ControlNet expects specific dimensions)
        sketch_image = sketch_image.resize((512, 512))
        
        return sketch_image
    
    def generate_with_replicate(self, prompt, sketch_path):
        """Generate using your fine-tuned Replicate model"""
        sketch_image = self.preprocess_sketch(sketch_path)
        
        # Save the preprocessed sketch temporarily
        temp_sketch_path = "temp_sketch.png"
        sketch_image.save(temp_sketch_path)
        
        # Run inference with your Replicate model
        output = replicate.run(
            DICKIES_MODEL_VERSION,
            input={
                "prompt": f"photorealistic Dickies men's jacket, highly detailed, {prompt}",
                "image": open(temp_sketch_path, "rb"),
                "negative_prompt": "low quality, blurry, distorted, unrealistic",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "controlnet_conditioning_scale": 0.8
            }
        )
        
        # Download the result image
        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]  # Most Replicate models return a list of URLs
            response = requests.get(image_url)
            result_image = Image.open(BytesIO(response.content))
            return result_image
        else:
            raise Exception("Failed to generate image with Replicate model")
    
    def generate_with_controlnet(self, prompt, sketch_path):
        """Generate using local ControlNet pipeline (backup method)"""
        sketch_image = self.preprocess_sketch(sketch_path)
        
        result = self.base_pipeline(
            prompt=f"photorealistic Dickies men's jacket, highly detailed, {prompt}",
            image=sketch_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,
            negative_prompt="low quality, blurry, distorted, unrealistic"
        ).images[0]
        
        return result
    
    def enhance_details(self, image):
        """Optional post-processing to enhance image details"""
        # This is a simple example - you might want to use more sophisticated methods
        from PIL import ImageEnhance
        
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_image = enhancer.enhance(1.5)  # Increase sharpness by 50%
        
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        return enhanced_image
    
    def convert_sketch_to_realistic(self, prompt, sketch_path, use_replicate=True, enhance=True):
        """Main method to convert a sketch to a photorealistic image"""
        try:
            if use_replicate:
                result_image = self.generate_with_replicate(prompt, sketch_path)
            else:
                result_image = self.generate_with_controlnet(prompt, sketch_path)
            
            if enhance:
                result_image = self.enhance_details(result_image)
            
            # Save the result
            filename = os.path.basename(sketch_path) if not sketch_path.startswith(('http://', 'https://')) else "result.png"
            output_path = os.path.join(OUTPUT_DIR, f"realistic_{filename}")
            result_image.save(output_path)
            
            print(f"Photorealistic image saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            # Fallback to the other method if one fails
            if use_replicate:
                print("Falling back to local ControlNet...")
                return self.convert_sketch_to_realistic(prompt, sketch_path, use_replicate=False)
            return None

# Example usage
if __name__ == "__main__":
    converter = DickiesSketchConverter()
    
    # Example - you'd replace this with your actual sketch path
    sketch_path = "path/to/your/sketch.png"
    prompt = "workwear jacket with pockets, high quality stitching, durable fabric"
    
    output_image = converter.convert_sketch_to_realistic(prompt, sketch_path)
