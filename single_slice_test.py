from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")
image = Image.open("/cs/home/psaas6/cognid_project/axial_slice_075.png").convert("RGB")
inputs = processor("Describe this image", image, return_tensors="pt")
print(inputs.keys())
