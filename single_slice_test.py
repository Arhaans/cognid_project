#!/usr/bin/env python3
import sys
import os
from datetime import datetime
from PIL import Image

# Add LLaVA-Med to path
sys.path.append("/cs/home/psaas6/LLaVA-Med")

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def load_llava_med():
    """Load LLaVA-Med using the repo's builder"""
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    try:
        print("🔄 Loading model on GPU with device_map='auto'...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device_map="auto"
        )
        print("✔️ Model loaded with GPU")
    except Exception as e:
        print(f"⚠️ GPU load failed ({e}), falling back to CPU...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device_map="cpu"
        )
        print("✔️ Model loaded on CPU")

    return tokenizer, model, image_processor, context_len

def analyze_brain_slice():
    tokenizer, model, image_processor, context_len = load_llava_med()
    device = next(model.parameters()).device

    # Load image
    image_path = "/cs/home/psaas6/cognid_project/axial_slice_075.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)

    # Prompt
    prompt = """You are a neuroradiologist with expertise in neurodegenerative disorders. Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities consistent with Alzheimer's disease or any neurodegenerative disease.

Please provide a detailed radiological report that includes:
1. Description of visible anatomical structures
2. Assessment of ventricular size and morphology
3. Evaluation of cortical thickness and any atrophy
4. Identification of any abnormal signal intensities or lesions
5. Overall impression and differential diagnosis considerations

Format your response as a formal radiological report."""

    # Prepare prompt
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    print("🤖 Generating report...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=[image_tensor],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output.split("ASSISTANT:")[-1].strip()

    print("\n" + "=" * 80)
    print("🧠 NEURORADIOLOGY REPORT - LLaVA-Med")
    print("=" * 80)
    print(f"Image: axial_slice_075.png")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(response)
    print("=" * 80)

if __name__ == "__main__":
    analyze_brain_slice()
