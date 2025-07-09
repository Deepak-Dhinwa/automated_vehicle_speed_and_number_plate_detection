from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import cv2
import os

# Load TrOCR fine-tuned model
trocr_path = "models/trocr_numberplate_finetuned"

processor = TrOCRProcessor.from_pretrained(trocr_path)
trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trocr_model.to(device)

def read_plate_text(plate_crop):
    """
    Given plate crop (BGR image), returns the detected plate text.
    """
    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    plate_pil = Image.fromarray(plate_rgb)

    inputs = processor(images=plate_pil, return_tensors="pt").to(device)
    generated_ids = trocr_model.generate(**inputs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text
