#!/usr/bin/env python3
import sys
import os
from datetime import datetime
from PIL import Image

# Add LLaVA-Med repo path
sys.path.append("/cs/home/psaas6/LLaVA-Med")

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def load_llava_med():
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    print("🔄 Loading LLaVA-Med...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map="auto"
    )
    print("✔️ Model loaded.")
    return tokenizer, model, image_processor, context_len

def analyze_brain_slice():
    tokenizer, model, image_processor, context_len = load_llava_med()
    device = next(model.parameters()).device

    image_path = "/cs/home/psaas6/cognid_project/axial_slice_075.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)

    prompt = (
        "You are a neuroradiologist with expertise in neurodegenerative disorders. "
        "Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities "
        "consistent with Alzheimer's disease or other neurodegenerative disorders.\n\n"
        "Provide:\n"
        "1. Anatomical structure descriptions\n"
        "2. Ventricular size/morphology\n"
        "3. Cortical thickness/atrophy\n"
        "4. Signal abnormalities or lesions\n"
        "5. Overall impression and differential diagnosis\n\n"
        "Respond in a formal radiology report format."
    )

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    print("🧠 Generating radiology report...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=[image_tensor],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    report = output.split("ASSISTANT:")[-1].strip()

    print("\n" + "=" * 80)
    print("🧾 LLaVA-Med RADIOLOGY REPORT")
    print("=" * 80)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(report)
    print("=" * 80)

if __name__ == "__main__":
    analyze_brain_slice()
