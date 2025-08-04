#!/usr/bin/env python3

import os
from datetime import datetime
from PIL import Image
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def load_llava_med():
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    model_name = get_model_name_from_path(model_path)

    print(f"🔄 Loading LLaVA-Med from: {model_path} (model_name: {model_name})")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base="microsoft/llava-med-v1.5-mistral-7b",  # ✅ critical fix
        model_name=model_name,
        device_map="auto"
    )

    print(f"✅ Model loaded. Context length: {context_len}")
    print(f"🔍 Vision tower: {getattr(model.config, 'vision_tower', 'None')}")

    return tokenizer, model, image_processor, context_len

def analyze_brain_slice():
    tokenizer, model, image_processor, context_len = load_llava_med()
    device = next(model.parameters()).device
    print(f"🖥️ Running on device: {device}")

    image_path = "/cs/home/psaas6/cognid_project/axial_slice_075.png"
    print(f"📷 Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    print("🖼️ Processing image...")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    print(f"✅ Image tensor shape: {image_tensor.shape}")

    # Prompt setup
    print("📝 Building conversation prompt...")
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
    print(f"📜 Final prompt text:\n{prompt_text[:300]}...")  # Preview prompt

    print("🧪 Tokenizing prompt...")
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    print(f"✅ Input token shape (before batch dim): {input_ids.shape[1:]}")

    print("🧠 Generating radiology report...")
    try:
        print(f"⚙️ Calling model.generate() with:\n   input_ids.shape: {input_ids.shape}\n   image_tensor.shape: {image_tensor.shape}")
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

    except Exception as e:
        print(f"❌ model.generate() failed: {e}")

if __name__ == "__main__":
    analyze_brain_slice()
