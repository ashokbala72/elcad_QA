from PIL import Image
import easyocr
import numpy as np

def extract_layout_text(image):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_np)

    text_blocks = []
    positions = []

    for box, text, conf in results:
        if conf > 0.5:
            text_blocks.append(text.strip())
            positions.append(box[0])  # top-left coordinate

    return {
        "text_blocks": text_blocks,
        "positions": positions
    }
