from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_symbols(image):
    """
    Splits image into tiles and classifies them using CLIP.
    Assumes components are spatially separated enough for patching to work.
    """

    labels = ["CT", "PT", "Relay", "Switchgear", "Cable", "Ground", "Unknown"]
    image = image.convert("RGB")
    w, h = image.size
    tile_size = 150  # Adjustable

    symbols = []

    for top in range(0, h, tile_size):
        for left in range(0, w, tile_size):
            box = (left, top, min(left + tile_size, w), min(top + tile_size, h))
            cropped = image.crop(box)

            inputs = clip_processor(text=labels, images=cropped, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().tolist()

            best_idx = torch.tensor(probs).argmax().item()
            confidence = probs[best_idx]
            if confidence > 0.8 and labels[best_idx] != "Unknown":
                symbols.append({
                    "label": labels[best_idx],
                    "confidence": round(confidence, 2),
                    "position": (left, top)
                })

    return symbols
