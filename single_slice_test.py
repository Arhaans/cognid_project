#!/usr/bin/env python3
import sys
import os
from datetime import datetime
from PIL import Image

# Add LLaVA-Med repo to path
sys.path.append("/cs/home/psaas6/LLaVA-Med")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

def load_llava_med():
    """Load LLaVA-Med with device fallback"""
    model_id = "microsoft/llava-med-v1.5-mistral-7b"
    try:
        print("🔄 Loading model with device_map='auto' (FP16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        print("✔️ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load on GPU: {e}. Trying CPU fallback...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        print("✔️ Loaded model on CPU")
    return model, tokenizer, processor

def analyze_brain_slice():
    """Run analysis on brain MRI slice"""
    model, tokenizer, processor = load_llava_med()
    device = next(model.parameters()).device

    # Image path
    slice_path = "/cs/home/psaas6/cognid_project/axial_slice_075.png"
    if not os.path.exists(slice_path):
        print(f"❌ File not found: {slice_path}")
        return

    image = Image.open(slice_path).convert("RGB")

    prompt = """You are a neuroradiologist with expertise in neurodegenerative disorders. Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities consistent with Alzheimer's disease or any neurodegenerative disease. 

Please provide a detailed radiological report that includes:
1. Description of visible anatomical structures
2. Assessment of ventricular size and morphology
3. Evaluation of cortical thickness and any atrophy
4. Identification of any abnormal signal intensities or lesions
5. Overall impression and differential diagnosis considerations

Format your response as a formal radiological report."""

    print("🖼️ Preprocessing with processor...")
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    print("🤖 Generating report...")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1024)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("ASSISTANT:")[-1].strip()

    print("\n" + "=" * 80)
    print("🧠 NEURORADIOLOGY REPORT - LLaVA-Med Analysis")
    print("=" * 80)
    print(f"Patient ID: CogNID010_1")
    print(f"Image: axial_slice_075.png")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print("\nRADIOLOGICAL ANALYSIS:")
    print(response)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_brain_slice()
