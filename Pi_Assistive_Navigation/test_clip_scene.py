#!/usr/bin/env python3
"""
Test CLIP scene classification
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

def test_clip_scene():
    """Test CLIP scene classification"""
    print("Testing CLIP Scene Classification...")
    
    # Load CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    
    # Scene labels from the config
    scene_labels = [
        'forest', 'park', 'beach', 'urban street', 'residential area',
        'indoor hallway', 'corridor', 'staircase', 'museum gallery',
        'store', 'plaza', 'parking lot'
    ]
    
    # Test with sample images
    test_images = [
        '../test_output.jpg',
        '../annotated.jpg', 
        '../Testing stuff/Ash_Tree_-_geograph.org.uk_-_590710.jpg'
    ]
    
    for img_path in test_images:
        try:
            print(f"\nTesting with: {img_path}")
            
            # Load and process image
            image = Image.open(img_path)
            print(f"  Image size: {image.size}")
            
            # Prepare inputs
            text_inputs = processor(
                text=scene_labels,
                return_tensors='pt',
                padding=True
            )
            image_inputs = processor(
                images=image,
                return_tensors='pt'
            )
            
            # Move to device
            for k, v in image_inputs.items():
                image_inputs[k] = v.to(device)
            for k, v in text_inputs.items():
                text_inputs[k] = v.to(device)
            
            # Inference
            with torch.no_grad():
                image_emb = model.get_image_features(**image_inputs)
                text_emb = model.get_text_features(**text_inputs)
                
                # Normalize
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                
                # Scores
                scores = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
                scores = scores[0].cpu().tolist()
            
            # Get top predictions
            print("  Scene classification results:")
            for i, (label, score) in enumerate(zip(scene_labels, scores)):
                if score > 0.1:  # Only show scores above 10%
                    print(f"    {label}: {score:.3f}")
            
            # Top prediction
            max_idx = scores.index(max(scores))
            confidence = scores[max_idx]
            print(f"  Top prediction: {scene_labels[max_idx]} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_clip_scene()

