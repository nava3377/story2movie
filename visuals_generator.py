# File: visuals_generator.py
import torch
import gc
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

def generate_start_image(prompt, pipe, output_path="start_frame.png"):
    """Generates a single still image from a text prompt using a pre-loaded model."""
    print(f"🎨 Generating start image for: '{prompt[:40]}...'")
    image = pipe(prompt=prompt, height=512, width=512).images[0]
    image.save(output_path)
    return output_path

def generate_video_from_image(start_image_path, pipe, output_path="silent_video.mp4"):
    """Animates a still image into a short video clip using a pre-loaded model."""
    print(f"🎬 Animating image: {start_image_path}...")
    image = Image.open(start_image_path).convert("RGB")
    frames = pipe(image, num_frames=16, decode_chunk_size=4).frames[0]
    export_to_video(frames, output_path, fps=7)
    return output_path